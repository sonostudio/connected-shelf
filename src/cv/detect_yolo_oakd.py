import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import time
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model files
MODEL_BLOB_PATH = "model/model_shoes.blob"
LABELS_PATH = "model/labels.txt"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.9
IOU_THRESHOLD = 0.5

# Camera settings
CAMERA_FPS = 30
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 640

# Display settings
SHOW_FPS = True
SHOW_LABELS = True
SHOW_CONFIDENCE = True
SHOW_DEPTH = False

# Colors for bounding boxes (BGR format)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_labels(labels_path):
    """Load class labels from file"""
    labels = []
    if Path(labels_path).exists():
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"✓ Loaded {len(labels)} classes: {', '.join(labels)}")
    else:
        print(f"⚠ Labels file not found: {labels_path}")
        print("  Using generic labels (class_0, class_1, ...)")
    return labels


def create_yolo_pipeline(blob_path, num_classes=None, labels_path=None):
    """
    Create DepthAI pipeline with YOLO detection network

    Args:
        blob_path: Path to model blob
        num_classes: Number of classes (auto-detected from labels if not provided)
        labels_path: Path to labels file for auto-detection
    """
    if num_classes is None and labels_path is not None:
        try:
            with open(labels_path, 'r') as f:
                num_classes = len([line.strip() for line in f.readlines() if line.strip()])
            print(f"Auto-detected {num_classes} classes from labels file")
        except:
            num_classes = 80  # Default fallback
            print(f"Warning: Could not read labels file, using default {num_classes} classes")
    elif num_classes is None:
        num_classes = 80

    pipeline = dai.Pipeline()

    # ========================================================================
    # COLOR CAMERA
    # ========================================================================
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(CAMERA_FPS)

    # ========================================================================
    # YOLO DETECTION NETWORK
    # ========================================================================
    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    detection_nn.setBlobPath(str(blob_path))
    detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    detection_nn.setIouThreshold(IOU_THRESHOLD)

    # YOLOv8 specific settings
    detection_nn.setNumClasses(num_classes)
    detection_nn.setCoordinateSize(4)
    detection_nn.setAnchors([])  # YOLOv8 doesn't use anchors

    # YOLOv8 uses anchor-free detection with a single output layer
    # For 640x640 input: 80x80 + 40x40 + 20x20 = 6400 + 1600 + 400 = 8400 predictions
    # We need to tell DepthAI about this output layer
    detection_nn.setAnchorMasks({"side8400": []})  # Empty list for anchor-free

    detection_nn.setNumInferenceThreads(2)
    detection_nn.input.setBlocking(False)

    # Link camera to detection network
    cam_rgb.preview.link(detection_nn.input)

    # ========================================================================
    # STEREO DEPTH
    # ========================================================================
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    # Mono camera settings
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setCamera("left")
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setCamera("right")

    # Stereo depth settings
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)

    # Link mono cameras to stereo
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # ========================================================================
    # OUTPUTS
    # ========================================================================

    # RGB output
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # Detection output
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    detection_nn.out.link(xout_nn.input)

    # Depth output
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def get_depth_at_point(depth_frame, x, y):
    """
    Get depth value at specific pixel coordinates
    Returns depth in meters, or None if invalid
    """
    if depth_frame is None:
        return None

    h, w = depth_frame.shape
    if 0 <= x < w and 0 <= y < h:
        depth_mm = depth_frame[y, x]
        if depth_mm > 0:
            return depth_mm / 1000.0  # Convert to meters
    return None


def draw_detection(frame, detection, label, color, depth_frame=None):
    """
    Draw bounding box and label for a detection
    """
    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]

    # Calculate bounding box coordinates
    x1 = int(detection.xmin * frame_w)
    y1 = int(detection.ymin * frame_h)
    x2 = int(detection.xmax * frame_w)
    y2 = int(detection.ymax * frame_h)

    # Ensure coordinates are within frame
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w - 1))
    y2 = max(0, min(y2, frame_h - 1))

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Prepare label text
    label_parts = []

    if SHOW_LABELS:
        label_parts.append(label)

    if SHOW_CONFIDENCE:
        label_parts.append(f"{detection.confidence:.1%}")

    # Add depth information
    if SHOW_DEPTH and depth_frame is not None:
        # Get depth at center of bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        depth = get_depth_at_point(depth_frame, center_x, center_y)

        if depth is not None:
            label_parts.append(f"{depth:.2f}m")

    label_text = " | ".join(label_parts)

    # Draw label background
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )

    cv2.rectangle(
        frame,
        (x1, y1 - text_h - baseline - 5),
        (x1 + text_w, y1),
        color,
        -1
    )

    # Draw label text
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),  # Black text
        2
    )

    return frame


def create_depth_visualization(depth_frame):
    """
    Create a colored depth visualization
    """
    # Normalize depth for visualization
    depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Apply histogram equalization for better contrast
    depth_vis = cv2.equalizeHist(depth_vis)

    # Apply color map
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    return depth_colored


def add_info_overlay(frame, fps, num_detections, status_text=""):
    """
    Add information overlay to frame
    """
    overlay_y = 30
    line_height = 30

    if SHOW_FPS:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        overlay_y += line_height

    cv2.putText(
        frame,
        f"Detections: {num_detections}",
        (10, overlay_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    overlay_y += line_height

    if status_text:
        cv2.putText(
            frame,
            status_text,
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    return frame


# ============================================================================
# MAIN DETECTION FUNCTION
# ============================================================================

def run_detection(save_video=False, output_path="output.mp4",
                  model_path=None, labels_path=None,
                  conf_threshold=None, iou_threshold=None, show_depth=None):
    # Use provided parameters or fall back to defaults
    model_path = model_path or MODEL_BLOB_PATH
    labels_path = labels_path or LABELS_PATH
    conf_threshold = conf_threshold if conf_threshold is not None else CONFIDENCE_THRESHOLD
    iou_threshold = iou_threshold if iou_threshold is not None else IOU_THRESHOLD
    show_depth_setting = show_depth if show_depth is not None else SHOW_DEPTH

    print("\n" + "="*70)
    print("OAK-D Pro Object Detection")
    print("="*70)

    if not Path(model_path).exists():
        print(f"\n✗ Model not found: {model_path}")
        print("\nPlease ensure:")
        print("  1. Training is complete")
        print("  2. Model has been converted to blob format")
        print("  3. MODEL_BLOB_PATH points to the correct file")
        return

    print(f"Model: {model_path}")

    labels = load_labels(labels_path)

    print(f"\nSettings:")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  IOU threshold: {iou_threshold}")
    print(f"  Camera FPS: {CAMERA_FPS}")
    print(f"  Show depth: {show_depth_setting}")

    print(f"\nControls:")
    print(f"  'q' - Quit")
    print(f"  's' - Save current frame")
    print(f"  'd' - Toggle depth display")
    print(f"  'c' - Toggle confidence display")
    print(f"  '+' - Increase confidence threshold")
    print(f"  '-' - Decrease confidence threshold")

    print("\n" + "="*70)

    print("\nCreating pipeline...")
    pipeline = create_yolo_pipeline(model_path, labels_path=labels_path)

    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, CAMERA_FPS,
                                       (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        print(f"Recording to: {output_path}")

    print("Connecting to OAK-D Pro...")

    with dai.Device(pipeline) as device:
        print(f"✓ Connected: {device.getDeviceName()}")
        print(f"✓ USB Speed: {device.getUsbSpeed().name}")

        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_det = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        # FPS calculation
        frame_count = 0
        fps_start_time = time.time()
        fps = 0

        show_depth = show_depth_setting
        show_confidence = SHOW_CONFIDENCE
        current_threshold = conf_threshold

        print("\n✓ Detection started!\n")

        try:
            while True:
                # Get RGB frame
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()

                # Get detections
                in_det = q_det.get()
                detections = in_det.detections

                # Get depth frame
                depth_frame = None
                depth_colored = None
                if show_depth:
                    in_depth = q_depth.get()
                    depth_frame = in_depth.getFrame()
                    depth_colored = create_depth_visualization(depth_frame)

                # Filter detections by current threshold
                filtered_detections = [
                    det for det in detections
                    if det.confidence >= current_threshold
                ]

                # Draw detections
                for i, detection in enumerate(filtered_detections):
                    class_id = detection.label
                    if labels and class_id < len(labels):
                        label = labels[class_id]
                    else:
                        label = f"class_{class_id}"

                    color = COLORS[class_id % len(COLORS)]
                    frame = draw_detection(frame, detection, label, color, depth_frame)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()

                status = f"Threshold: {current_threshold:.2f}"
                frame = add_info_overlay(frame, fps, len(filtered_detections), status)

                # Save to video if enabled
                if video_writer is not None:
                    video_writer.write(frame)

                cv2.imshow("OAK-D Pro Detection", frame)

                if show_depth and depth_colored is not None:
                    cv2.imshow("Depth", depth_colored)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nStopping detection...")
                    break

                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Saved: {filename}")

                elif key == ord('d'):
                    # Toggle depth display
                    show_depth = not show_depth
                    if not show_depth:
                        cv2.destroyWindow("Depth")
                    print(f"Depth display: {'ON' if show_depth else 'OFF'}")

                elif key == ord('c'):
                    # Toggle confidence display
                    show_confidence = not show_confidence
                    print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")

                elif key == ord('+') or key == ord('='):
                    # Increase threshold
                    current_threshold = min(1.0, current_threshold + 0.05)
                    print(f"Confidence threshold: {current_threshold:.2f}")

                elif key == ord('-') or key == ord('_'):
                    # Decrease threshold
                    current_threshold = max(0.0, current_threshold - 0.05)
                    print(f"Confidence threshold: {current_threshold:.2f}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Cleanup
            if video_writer is not None:
                video_writer.release()
                print(f"✓ Video saved: {output_path}")

            cv2.destroyAllWindows()
            print("✓ Detection stopped")
            print(f"Total frames processed: {frame_count}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OAK-D Pro Object Detection with YOLOv8"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_BLOB_PATH,
        help=f"Path to model blob file (default: {MODEL_BLOB_PATH})"
    )

    parser.add_argument(
        "--labels",
        type=str,
        default=LABELS_PATH,
        help=f"Path to labels file (default: {LABELS_PATH})"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help=f"IOU threshold (default: {IOU_THRESHOLD})"
    )

    parser.add_argument(
        "--depth",
        action="store_true",
        help="Enable depth display"
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save detection video"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video path (default: output.mp4)"
    )

    args = parser.parse_args()

    run_detection(
        save_video=args.save_video,
        output_path=args.output,
        model_path=args.model,
        labels_path=args.labels,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        show_depth=args.depth
    )


if __name__ == "__main__":
    main()
