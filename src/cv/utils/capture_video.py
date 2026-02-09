import cv2
import depthai as dai
import os
import warnings
from datetime import datetime

# --- CONFIG ---
DATASET_DIR = "../data/raw/"
LABEL_ID = "0"

os.makedirs(DATASET_DIR, exist_ok=True)

# --- PIPELINE SETUP ---
pipeline = dai.Pipeline()
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define Color Camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(320, 320)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam.setFps(30)
cam.setIspScale(1, 3)

# --- ENHANCED VISION SETTINGS ---
cam.initialControl.setContrast(5)
cam.initialControl.setSharpness(5)
cam.initialControl.setSaturation(0)

# XLinkOut (Output to host)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")

cam.preview.link(xout.input)

print(f"📷 OAK-D Video Recorder initialized.")
print(f"🎯 Target Label ID: '{LABEL_ID}'")
print(f"⌨️  Controls: [R] to Start/Stop Recording, [Q] to Quit")

# Connect to device
try:
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        recording = False
        video_writer = None

        while True:
            in_rgb = q_rgb.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

                # --- RECORDING LOGIC ---
                if recording and video_writer is not None:
                    video_writer.write(frame)

                # --- DISPLAY LOGIC ---
                # Create a copy so we don't draw the "REC" text onto the saved video
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

                elif key == ord('r'):  # 'R' key toggles recording
                    if not recording:
                        # START RECORDING
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{DATASET_DIR}/{LABEL_ID}_{timestamp}.mp4"

                        fourcc = cv2.VideoWriter_fourcc(*'avc1')
                        video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (320, 320))

                        recording = True
                        print(f"🔴 Recording started: {filename}")
                    else:
                        # STOP RECORDING
                        recording = False
                        video_writer.release()
                        video_writer = None
                        print(f"⏹️  Recording saved.")

except Exception as e:
    print(f"\n🔴 ERROR: {e}")

cv2.destroyAllWindows()
