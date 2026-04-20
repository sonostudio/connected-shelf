"""
detect_pi.py — Raspberry Pi side
---------------------------------
Runs YOLOv8 inference on the OAK-D Pro and sends detection results
to the Mac over OSC.

OSC messages sent:
  /detection   sku_label(str)  confidence(float)  x1 x2 y1 y2 (floats, 0-1 normalised)
  /no_detection  (no args)

Run from project root:
    poetry run python src/detect_pi.py
"""

import time
import logging
import yaml
import numpy as np
import depthai as dai
from pathlib import Path
from pythonosc import udp_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Pi] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_labels(path):
    p = Path(path)
    if not p.exists():
        logger.warning(f"Labels file not found: {path}")
        return []
    labels = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    logger.info(f"Loaded {len(labels)} labels: {', '.join(labels)}")
    return labels


# ---------------------------------------------------------------------------
# YOLOv8 host-side decoding
# (The blob is run through dai.node.NeuralNetwork, not YoloDetectionNetwork,
#  because YOLOv8 uses an anchor-free flat tensor output that NeuralNetwork
#  passes through raw — we decode it here instead.)
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolov8(output_tensor, num_classes, conf_threshold):
    """
    Decode the raw [1, 4 + num_classes, 8400] YOLOv8 output tensor.

    YOLOv8 packs all 8400 anchor-free predictions into a single tensor:
      rows 0-3  : cx, cy, w, h  (normalised 0-1)
      rows 4-N  : raw class logits (no objectness score)

    Returns a list of dicts:
      {label, confidence, x1, y1, x2, y2}   (all coords normalised 0-1)
    """
    # tensor shape after squeeze: (4 + num_classes, 8400)
    tensor = np.array(output_tensor).reshape((4 + num_classes, -1))

    cx   = tensor[0]          # (8400,)
    cy   = tensor[1]
    w    = tensor[2]
    h    = tensor[3]
    logits = tensor[4:]        # (num_classes, 8400)

    # Class scores via sigmoid (YOLOv8 uses sigmoid, not softmax)
    scores = sigmoid(logits)   # (num_classes, 8400)

    # Per-prediction: best class and its score
    class_ids  = np.argmax(scores, axis=0)   # (8400,)
    confidences = scores[class_ids, np.arange(scores.shape[1])]  # (8400,)

    # Filter by threshold
    mask = confidences >= conf_threshold
    if not np.any(mask):
        return []

    cx   = cx[mask];   cy = cy[mask]
    w    = w[mask];    h  = h[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    # cx/cy/w/h → x1/y1/x2/y2
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    detections = []
    for i in range(len(class_ids)):
        detections.append({
            "label":      int(class_ids[i]),
            "confidence": float(confidences[i]),
            "x1": float(np.clip(x1[i], 0, 1)),
            "y1": float(np.clip(y1[i], 0, 1)),
            "x2": float(np.clip(x2[i], 0, 1)),
            "y2": float(np.clip(y2[i], 0, 1)),
        })

    return detections


def nms(detections, iou_threshold=0.45):
    """
    Simple class-aware non-maximum suppression.
    Works on the list of dicts returned by decode_yolov8.
    """
    if not detections:
        return []

    # Group by class
    by_class = {}
    for d in detections:
        by_class.setdefault(d["label"], []).append(d)

    kept = []
    for label, dets in by_class.items():
        dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if _iou(best, d) < iou_threshold]

    return kept


def _iou(a, b):
    ix1 = max(a["x1"], b["x1"]);  iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]);  iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# DepthAI pipeline
# ---------------------------------------------------------------------------

def build_pipeline(model_path, camera_fps):
    """
    Build a DepthAI pipeline with ColorCamera → NeuralNetwork.
    We use NeuralNetwork (not YoloDetectionNetwork) so the raw
    tensor reaches the host for Python-side decoding.
    """
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 640)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(camera_fps)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(model_path))
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    cam.preview.link(nn.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    nn.out.link(xout_nn.input)

    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    det_cfg  = cfg["detection"]
    osc_cfg  = cfg["osc"]

    model_path   = det_cfg["model_path"]
    labels_path  = det_cfg["labels_path"]
    conf_thresh  = det_cfg["confidence_threshold"]
    camera_fps   = det_cfg["camera_fps"]
    send_interval = osc_cfg["send_interval"]

    labels = load_labels(labels_path)
    num_classes = len(labels)

    if not Path(model_path).exists():
        logger.error(f"Model blob not found: {model_path}")
        return

    # OSC client — sends to the Mac
    osc = udp_client.SimpleUDPClient(osc_cfg["mac_ip"], osc_cfg["port"])
    logger.info(f"OSC → {osc_cfg['mac_ip']}:{osc_cfg['port']}")

    logger.info("Building DepthAI pipeline…")
    pipeline = build_pipeline(model_path, camera_fps)

    with dai.Device(pipeline) as device:
        logger.info(f"Connected: {device.getDeviceName()}  USB: {device.getUsbSpeed().name}")

        q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

        last_send = 0.0

        logger.info("Detection loop running. Ctrl-C to stop.")
        try:
            while True:
                packet = q_nn.get()
                raw    = packet.getFirstLayerFp16()

                detections = decode_yolov8(raw, num_classes, conf_thresh)
                detections = nms(detections)

                now = time.time()
                if now - last_send < send_interval:
                    continue
                last_send = now

                if not detections:
                    osc.send_message("/no_detection", [])
                    logger.debug("→ /no_detection")
                    continue

                # Pick highest-confidence detection
                best = max(detections, key=lambda d: d["confidence"])
                label_name = labels[best["label"]] if best["label"] < len(labels) else f"class_{best['label']}"

                osc.send_message(
                    "/detection",
                    [
                        label_name,
                        best["confidence"],
                        best["x1"], best["y1"],
                        best["x2"], best["y2"],
                    ],
                )
                logger.info(
                    f"→ /detection  {label_name}  {best['confidence']:.2f}"
                    f"  [{best['x1']:.2f},{best['y1']:.2f},{best['x2']:.2f},{best['y2']:.2f}]"
                )

        except KeyboardInterrupt:
            logger.info("Stopped by user.")


if __name__ == "__main__":
    main()
