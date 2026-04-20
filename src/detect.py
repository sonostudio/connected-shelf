"""
detect_pi.py — Raspberry Pi side
---------------------------------
Runs YOLOv8 inference on the OAK-D Pro and sends detection results
to the Mac over OSC.

OSC messages sent:
  /detection   sku_label(str)  confidence(float)  x1 y1 x2 y2 (floats, 0-1 normalised)
  /no_detection  (no args)

Run from project root:
    poetry run python src/detect_pi.py

Controls (when preview is visible):
  p — toggle preview window on/off
  q — quit
"""

import time
import logging
import yaml
import numpy as np
import cv2
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
      {label, confidence, x1, y1, x2, y2}  (all coords normalised 0-1)
    """
    # tensor shape after squeeze: (4 + num_classes, 8400)
    tensor = np.array(output_tensor).reshape((4 + num_classes, -1))

    cx     = tensor[0]       # (8400,)
    cy     = tensor[1]
    w      = tensor[2]
    h      = tensor[3]
    logits = tensor[4:]      # (num_classes, 8400)

    # Class scores via sigmoid (YOLOv8 uses sigmoid, not softmax)
    scores = sigmoid(logits)  # (num_classes, 8400)

    # Per-prediction: best class and its score
    class_ids   = np.argmax(scores, axis=0)                      # (8400,)
    confidences = scores[class_ids, np.arange(scores.shape[1])]  # (8400,)

    # Filter by threshold
    mask = confidences >= conf_threshold
    if not np.any(mask):
        return []

    cx          = cx[mask];  cy = cy[mask]
    w           = w[mask];   h  = h[mask]
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
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Preview drawing helpers
# ---------------------------------------------------------------------------

# A small palette — cycles if there are more classes than colours
_PALETTE = [
    (0, 255, 0),    # green
    (255, 80, 0),   # blue-ish
    (0, 80, 255),   # red-ish
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
]


def draw_detections(frame, detections, labels):
    """
    Draw bounding boxes and labels onto a copy of frame.
    Coords in detections are normalised 0-1.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    for det in detections:
        x1 = int(det["x1"] * w);  y1 = int(det["y1"] * h)
        x2 = int(det["x2"] * w);  y2 = int(det["y2"] * h)

        color      = _PALETTE[det["label"] % len(_PALETTE)]
        label_name = labels[det["label"]] if det["label"] < len(labels) else f"class_{det['label']}"
        text       = f"{label_name} {det['confidence']:.0%}"

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)

        # Label text (black on coloured background)
        cv2.putText(out, text, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return out


def draw_overlay(frame, fps, num_detections, osc_host, osc_port):
    """Draw FPS / status info in the top-right corner."""
    lines = [
        f"FPS: {fps:.1f}",
        f"Detections: {num_detections}",
        f"OSC -> {osc_host}:{osc_port}",
    ]
    x = frame.shape[1] - 230
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return frame


# ---------------------------------------------------------------------------
# DepthAI pipeline
# ---------------------------------------------------------------------------

def build_pipeline(model_path, camera_fps, include_rgb_stream):
    """
    Build a DepthAI pipeline with ColorCamera → NeuralNetwork.

    include_rgb_stream: if True, also expose an XLinkOut for the raw RGB
    preview frame so we can display it on the host. This must be decided
    before the pipeline starts — it cannot be changed at runtime.
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

    if include_rgb_stream:
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam.preview.link(xout_rgb.input)

    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    det_cfg  = cfg["detection"]
    osc_cfg  = cfg["osc"]

    model_path    = det_cfg["model_path"]
    labels_path   = det_cfg["labels_path"]
    conf_thresh   = det_cfg["confidence_threshold"]
    camera_fps    = det_cfg["camera_fps"]
    send_interval = osc_cfg["send_interval"]

    # Preview is enabled via config; can be toggled off at runtime with [p].
    # Toggling it ON at runtime only works if it was already enabled at startup
    # (because the rgb XLinkOut must be in the pipeline before the device starts).
    show_preview = det_cfg.get("preview", False)

    labels      = load_labels(labels_path)
    num_classes = len(labels)

    if not Path(model_path).exists():
        logger.error(f"Model blob not found: {model_path}")
        return

    # OSC client — sends to the Mac
    osc = udp_client.SimpleUDPClient(osc_cfg["mac_ip"], osc_cfg["port"])
    logger.info(f"OSC → {osc_cfg['mac_ip']}:{osc_cfg['port']}")

    # Build pipeline — include the rgb stream if preview is requested
    logger.info("Building DepthAI pipeline…")
    pipeline = build_pipeline(model_path, camera_fps, include_rgb_stream=show_preview)

    with dai.Device(pipeline) as device:
        logger.info(f"Connected: {device.getDeviceName()}  USB: {device.getUsbSpeed().name}")

        q_nn  = device.getOutputQueue("nn",  maxSize=4, blocking=False)
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False) if show_preview else None

        if show_preview:
            cv2.namedWindow("Detection Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection Preview", 640, 640)
            logger.info("Preview window open.")

        logger.info("Detection loop running. Keys: [p] toggle preview  [q] quit")

        last_send   = 0.0
        last_fps_t  = time.time()
        frame_count = 0
        fps         = 0.0
        last_dets   = []   # cache so the preview stays useful between NN packets

        try:
            while True:
                # ---- neural network output ---------------------------------
                packet = q_nn.get()
                raw    = packet.getFirstLayerFp16()

                detections = decode_yolov8(raw, num_classes, conf_thresh)
                detections = nms(detections)
                last_dets  = detections

                # ---- OSC send (rate-limited) --------------------------------
                now = time.time()
                if now - last_send >= send_interval:
                    last_send = now

                    if not detections:
                        osc.send_message("/no_detection", [])
                        logger.debug("→ /no_detection")
                    else:
                        best = max(detections, key=lambda d: d["confidence"])
                        label_name = (
                            labels[best["label"]]
                            if best["label"] < len(labels)
                            else f"class_{best['label']}"
                        )
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
                            f"  [{best['x1']:.2f},{best['y1']:.2f},"
                            f"{best['x2']:.2f},{best['y2']:.2f}]"
                        )

                # ---- FPS counter -------------------------------------------
                frame_count += 1
                if frame_count % 30 == 0:
                    fps        = 30 / (time.time() - last_fps_t)
                    last_fps_t = time.time()

                # ---- preview -----------------------------------------------
                if show_preview and q_rgb is not None:
                    in_rgb = q_rgb.tryGet()
                    if in_rgb is not None:
                        frame = in_rgb.getCvFrame()
                        frame = draw_detections(frame, last_dets, labels)
                        frame = draw_overlay(frame, fps, len(last_dets),
                                             osc_cfg["mac_ip"], osc_cfg["port"])
                        cv2.imshow("Detection Preview", frame)

                # ---- keyboard controls -------------------------------------
                # waitKey(1) is needed even without a window so OpenCV
                # processes its internal event queue.
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    logger.info("Quit requested.")
                    break

                elif key == ord("p"):
                    show_preview = not show_preview
                    if show_preview:
                        if q_rgb is None:
                            # rgb stream was never added to the pipeline —
                            # user needs to enable it in config and restart.
                            logger.warning(
                                "Preview was disabled at startup so the RGB stream "
                                "was not included in the pipeline. "
                                "Set preview.enabled: true in config and restart."
                            )
                            show_preview = False
                        else:
                            cv2.namedWindow("Detection Preview", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("Detection Preview", 640, 640)
                            logger.info("Preview ON")
                    else:
                        cv2.destroyWindow("Detection Preview")
                        logger.info("Preview OFF")

        except KeyboardInterrupt:
            logger.info("Stopped by user.")

        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()