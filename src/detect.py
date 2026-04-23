"""
detect.py — Raspberry Pi side
------------------------------
Runs YOLOv8 inference on the OAK-D Pro and sends detection results
to the Mac over OSC.

OSC messages sent:
  /detection   sku_label(str)  confidence(float)  x1 y1 x2 y2 (floats, 0-1 normalised)
  /no_detection  (no args)

Run from project root:
    poetry run python src/detect.py

Controls (when preview: true in config):
  p — toggle preview window
  Ctrl-C — quit
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolov8(output_tensor, num_classes, conf_threshold):
    tensor      = np.array(output_tensor).reshape((4 + num_classes, -1))
    # cx/cy/w/h are in pixel space (0-640), normalise to 0-1
    cx = tensor[0] / 640; cy = tensor[1] / 640
    w  = tensor[2] / 640; h  = tensor[3] / 640
    # Values are already post-sigmoid (0-1) from the blob converter.
    # Do NOT apply sigmoid again — it would squash 1.0 down to 0.731.
    scores      = tensor[4:]
    class_ids   = np.argmax(scores, axis=0)
    confidences = scores[class_ids, np.arange(scores.shape[1])]
    mask        = confidences >= conf_threshold
    if not np.any(mask):
        return []
    cx = cx[mask]; cy = cy[mask]; w = w[mask]; h = h[mask]
    confidences = confidences[mask]; class_ids = class_ids[mask]
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    return [{"label": int(class_ids[i]), "confidence": float(confidences[i]),
             "x1": float(np.clip(x1[i], 0, 1)), "y1": float(np.clip(y1[i], 0, 1)),
             "x2": float(np.clip(x2[i], 0, 1)), "y2": float(np.clip(y2[i], 0, 1))}
            for i in range(len(class_ids))]


def nms(detections, iou_threshold=0.45):
    if not detections:
        return []
    by_class = {}
    for d in detections:
        by_class.setdefault(d["label"], []).append(d)
    kept = []
    for _, dets in by_class.items():
        dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if _iou(best, d) < iou_threshold]
    return kept


def _iou(a, b):
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = ((a["x2"]-a["x1"])*(a["y2"]-a["y1"]) +
             (b["x2"]-b["x1"])*(b["y2"]-b["y1"]) - inter)
    return inter / union if union > 0 else 0.0


_PALETTE = [
    (0, 255, 0), (255, 80, 0), (0, 80, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]


def draw_detections(frame, detections, labels):
    out = frame.copy()
    h, w = out.shape[:2]
    for det in detections:
        x1 = int(det["x1"]*w); y1 = int(det["y1"]*h)
        x2 = int(det["x2"]*w); y2 = int(det["y2"]*h)
        color = _PALETTE[det["label"] % len(_PALETTE)]
        name  = labels[det["label"]] if det["label"] < len(labels) else f"class_{det['label']}"
        text  = f"{name} {det['confidence']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1-th-bl-4), (x1+tw, y1), color, -1)
        cv2.putText(out, text, (x1, y1-bl-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return out


def draw_overlay(frame, fps, num_dets, osc_host, osc_port):
    lines = [f"FPS: {fps:.1f}", f"Detections: {num_dets}",
             f"OSC -> {osc_host}:{osc_port}"]
    x = frame.shape[1] - 230
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, 24+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return frame


def build_pipeline(model_path, camera_fps, include_preview):
    """
    cam.preview  -> nn.input  (640x640, feeds the neural network)
    cam.video    -> xout_video (resized to 640x640 for display)

    Using cam.video for display instead of cam.preview means the two
    streams are independent — the NN getting preview frames does not
    starve the display stream.
    """
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 640)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(camera_fps)

    if include_preview:
        # Resize the video stream to match the preview/NN size for display
        cam.setVideoSize(640, 640)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(model_path))
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    cam.preview.link(nn.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    nn.out.link(xout_nn.input)

    if include_preview:
        xout_video = pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName("video")
        cam.video.link(xout_video.input)

    return pipeline


def main():
    cfg      = load_config()
    det_cfg  = cfg["detection"]
    osc_cfg  = cfg["osc"]

    model_path        = det_cfg["model_path"]
    labels_path       = det_cfg["labels_path"]
    conf_thresh       = det_cfg["confidence_threshold"]
    pre_filter_thresh = det_cfg.get("pre_filter_threshold", conf_thresh)
    camera_fps        = det_cfg["camera_fps"]
    send_interval     = osc_cfg["send_interval"]
    show_preview      = det_cfg.get("preview", False)

    labels      = load_labels(labels_path)
    num_classes = len(labels)

    if not Path(model_path).exists():
        logger.error(f"Model blob not found: {model_path}")
        return

    osc = udp_client.SimpleUDPClient(osc_cfg["mac_ip"], osc_cfg["port"])
    logger.info(f"OSC -> {osc_cfg['mac_ip']}:{osc_cfg['port']}")

    logger.info("Building DepthAI pipeline...")
    pipeline = build_pipeline(model_path, camera_fps, include_preview=show_preview)

    with dai.Device(pipeline) as device:
        logger.info(f"Connected: {device.getDeviceName()}  USB: {device.getUsbSpeed().name}")

        q_nn    = device.getOutputQueue("nn",    maxSize=4, blocking=False)
        q_video = device.getOutputQueue("video", maxSize=4, blocking=False) if show_preview else None

        if show_preview:
            cv2.namedWindow("Detection Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection Preview", 640, 640)
            logger.info("Preview window open. [p] toggle  Ctrl-C to quit")

        last_send   = 0.0
        last_fps_t  = time.time()
        frame_count = 0
        fps         = 0.0
        last_dets   = []

        try:
            while True:
                # ---- NN output (blocking — drives the loop) ----------------
                in_nn = q_nn.get()
                raw   = in_nn.getFirstLayerFp16()

                detections = decode_yolov8(raw, num_classes, pre_filter_thresh)
                detections = nms(detections)
                last_dets  = detections

                # ---- OSC (rate-limited, gated by conf_thresh) --------------
                now = time.time()
                if now - last_send >= send_interval:
                    last_send = now
                    strong = [d for d in last_dets if d["confidence"] >= conf_thresh]
                    if not strong:
                        osc.send_message("/no_detection", [])
                        logger.debug("-> /no_detection")
                    else:
                        best = max(strong, key=lambda d: d["confidence"])
                        label_name = (labels[best["label"]]
                                      if best["label"] < len(labels)
                                      else f"class_{best['label']}")
                        osc.send_message("/detection",
                                         [label_name, best["confidence"],
                                          best["x1"], best["y1"],
                                          best["x2"], best["y2"]])
                        logger.info(f"-> /detection  {label_name}  {best['confidence']:.2f}")

                # ---- FPS ---------------------------------------------------
                frame_count += 1
                if frame_count % 30 == 0:
                    fps        = 30 / (time.time() - last_fps_t)
                    last_fps_t = time.time()

                # ---- preview (cam.video — independent of cam.preview/NN) --
                if show_preview and q_video is not None:
                    in_video = q_video.tryGet()
                    if in_video is not None:
                        frame = in_video.getCvFrame()
                        frame = draw_detections(frame, last_dets, labels)
                        frame = draw_overlay(frame, fps, len(last_dets),
                                             osc_cfg["mac_ip"], osc_cfg["port"])
                        cv2.imshow("Detection Preview", frame)

                # ---- keyboard ----------------------------------------------
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    show_preview = not show_preview
                    if show_preview:
                        if q_video is None:
                            logger.warning("Video stream not in pipeline — "
                                           "set detection.preview: true and restart.")
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