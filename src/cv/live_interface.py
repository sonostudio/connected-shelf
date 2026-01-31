import cv2
import depthai as dai
import numpy as np

# Configuration
BLOB_PATH = "models/20260120_6c_v1.blob"
CONFIDENCE_THRESHOLD = 0.8
IOU_THRESHOLD = 0.5
LABELS = ['eva', 'firstimage', 'linger', 'midwinter', 'revuediapo', 'satoshi', 'toiletpaper']


def decode_from_map(output_tensor):
    """
    Decodes raw tensor output into bounding boxes, labels, and scores.
    """
    num_anchors = 2100
    total_size = len(output_tensor)
    num_channels = total_size // num_anchors

    planar = np.array(output_tensor).reshape(num_channels, num_anchors)
    output = planar.transpose()

    boxes = output[:, :4]
    scores = output[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)

    mask = max_scores >= CONFIDENCE_THRESHOLD
    filtered_boxes = boxes[mask]
    filtered_scores = max_scores[mask]
    filtered_class_ids = class_ids[mask]

    if len(filtered_boxes) == 0:
        return [], [], []

    final_boxes = []
    for box in filtered_boxes:
        cx, cy, w, h = box
        final_boxes.append([int(cx - (w / 2)), int(cy - (h / 2)), int(w), int(h)])

    indices = cv2.dnn.NMSBoxes(final_boxes, filtered_scores.tolist(), CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

    clean_boxes, clean_labels, clean_scores = [], [], []
    if len(indices) > 0:
        for i in indices.flatten():
            clean_boxes.append(final_boxes[i])
            clean_labels.append(LABELS[filtered_class_ids[i]])
            clean_scores.append(filtered_scores[i])

    return clean_boxes, clean_labels, clean_scores


def create_pipeline():
    pipeline = dai.Pipeline()

    # Camera Setup
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(320, 320)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.setFps(30)
    camRgb.setIspScale(1, 3)

    # Neural Network Setup
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(BLOB_PATH)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    camRgb.preview.link(nn.input)

    # XLink Outputs
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutNN.setStreamName("nn")
    nn.out.link(xoutNN.input)

    return pipeline


def main():
    print("Connected! Running Final Decoder...")
    pipeline = create_pipeline()

    try:
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
                    try:
                        raw_data = inDet.getLayerFp16("output0")
                        boxes, labels, scores = decode_from_map(raw_data)

                        if frame is not None:
                            scale_x = frame.shape[1] / 320
                            scale_y = frame.shape[0] / 320

                            for i, box in enumerate(boxes):
                                x, y, w, h = [int(val * (scale_x if j % 2 == 0 else scale_y)) for j, val in
                                              enumerate(box)]
                                label_text = f"{labels[i]} {int(scores[i] * 100)}%"

                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                            2)
                    except Exception as e:
                        print(f"Inference Error: {e}")

                if frame is not None:
                    cv2.imshow("Connected Shelf - Live", frame)

                if cv2.waitKey(1) == ord('q'):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
