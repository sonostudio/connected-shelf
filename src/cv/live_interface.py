import cv2
import depthai as dai
import numpy as np
import time

# --- CONFIGURATION ---
BLOB_PATH = "models/best_openvino_2022.1_6shave.blob"  # Your compiled model
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5  # For removing overlapping boxes

# MATCH YOUR LABELS EXACTLY
LABELS = [
    "linger",
    "first_image",
    "blyertspenna",
    "lone",
    "clinical_nugetts",
    "revue_diapo_002"
]


# ---------------------

def decode_yolov8(output_tensor, conf_thresh, iou_thresh):
    """
    Decodes the raw YOLOv8 output (1, 4+nc, 2100) into useful boxes.
    """
    # 1. Reshape: OAK sends a flat list. We need (Classes+4, 2100)
    # 320x320 model -> 2100 predictions
    num_classes = len(LABELS)
    num_anchors = 2100

    # The tensor comes in flattened. Reshape it to [1, Channels, Anchors]
    # Channels = 4 box coords + Num_Classes
    expected_len = (4 + num_classes) * num_anchors

    if len(output_tensor) != expected_len:
        # Safety check if model size is different
        return [], [], []

    output = np.array(output_tensor).reshape(4 + num_classes, num_anchors)

    # 2. Transpose to [2100, Channels] for easier iteration
    output = output.transpose()

    # 3. Filter low confidence predictions
    # Box coords are the first 4 columns (cx, cy, w, h)
    boxes = output[:, :4]
    # Class scores are the rest
    scores = output[:, 4:]

    # Find the maximum score for each anchor and which class it belongs to
    class_ids = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)

    # Create a mask for high-confidence detections
    mask = max_scores >= conf_thresh

    filtered_boxes = boxes[mask]
    filtered_scores = max_scores[mask]
    filtered_class_ids = class_ids[mask]

    if len(filtered_boxes) == 0:
        return [], [], []

    # 4. Convert (cx, cy, w, h) to (x, y, w, h) for OpenCV NMS
    # Note: These are normalized 0-1
    final_boxes = []
    for box in filtered_boxes:
        cx, cy, w, h = box
        x = int((cx - w / 2) * 320)  # Scale to preview size
        y = int((cy - h / 2) * 320)
        w = int(w * 320)
        h = int(h * 320)
        final_boxes.append([x, y, w, h])

    # 5. Non-Maximum Suppression (Remove overlaps)
    indices = cv2.dnn.NMSBoxes(final_boxes, filtered_scores.tolist(), conf_thresh, iou_thresh)

    clean_boxes = []
    clean_labels = []
    clean_scores = []

    if len(indices) > 0:
        for i in indices.flatten():
            clean_boxes.append(final_boxes[i])
            clean_labels.append(LABELS[filtered_class_ids[i]])
            clean_scores.append(filtered_scores[i])

    return clean_boxes, clean_labels, clean_scores


# --- PIPELINE SETUP ---
pipeline = dai.Pipeline()

# 1. Camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(320, 320)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)
camRgb.setIspScale(1, 3)  # Fix for OAK-D Tiny/Lite

# 2. Neural Network (Standard, not "YoloDetectionNetwork")
# We use the generic node so we can handle the raw data ourselves
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(BLOB_PATH)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

camRgb.preview.link(nn.input)

# 3. Outputs
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("nn")
nn.out.link(xoutNN.input)

# --- MAIN LOOP ---
print("🚀 Connected! Running Host-Side Decoding...")

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None

    while True:
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

        if inDet is not None:
            # Get the raw layer data
            # "output0" is the standard name for YOLOv8 exports
            try:
                raw_data = inDet.getLayerFp16("output0")

                # Run our Python decoder
                boxes, labels, scores = decode_yolov8(raw_data, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

                if frame is not None:
                    # Scale boxes from 320x320 back to frame size (640x360)
                    scale_x = frame.shape[1] / 320
                    scale_y = frame.shape[0] / 320

                    for i, box in enumerate(boxes):
                        x, y, w, h = box
                        # Scale up
                        x = int(x * scale_x)
                        y = int(y * scale_y)
                        w = int(w * scale_x)
                        h = int(h * scale_y)

                        label = labels[i]
                        conf = f"{int(scores[i] * 100)}%"

                        # Draw
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf}", (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except Exception as e:
                # If layer name is wrong, it will print available layers
                print(f"Layer Error: {e}")
                print(f"Available layers: {inDet.getAllLayerNames()}")

        if frame is not None:
            cv2.imshow("Connected Shelf - Live", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()