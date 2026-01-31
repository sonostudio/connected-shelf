import cv2
import depthai as dai
import os
import warnings
from datetime import datetime

# Configuration
DATASET_DIR = "../data/raw/videos"
LABEL_ID = "0"


def create_pipeline():
    pipeline = dai.Pipeline()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(320, 320)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.setFps(30)
    cam.setIspScale(1, 3)

    cam.initialControl.setContrast(5)
    cam.initialControl.setSharpness(5)
    cam.initialControl.setSaturation(0)

    # Output to host
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    return pipeline


def run_recorder():
    os.makedirs(DATASET_DIR, exist_ok=True)
    pipeline = create_pipeline()

    print(f"OAK-D Video Recorder initialized.")
    print(f"Target Label ID: '{LABEL_ID}'")
    print(f"Controls: [R] to Start/Stop Recording, [Q] to Quit")

    try:
        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            recording = False
            video_writer = None

            while True:
                in_rgb = q_rgb.tryGet()

                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()

                    if recording and video_writer is not None:
                        video_writer.write(frame)

                    display_frame = frame.copy()

                    if recording:
                        cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(display_frame, "REC", (50, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    cv2.imshow("Data Capture - OAK-D", display_frame)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        if recording:
                            print("💾 Saving final video...")
                            video_writer.release()
                        break

                    elif key == ord('r'):
                        if not recording:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{DATASET_DIR}/{LABEL_ID}_{timestamp}.mp4"
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (320, 320))
                            recording = True
                            print(f"Recording started: {filename}")
                        else:
                            recording = False
                            video_writer.release()
                            video_writer = None
                            print(f"⏹️  Recording saved.")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recorder()
