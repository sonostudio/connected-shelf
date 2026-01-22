import cv2
import depthai as dai
import time
import os
import warnings
from datetime import datetime

# --- CONFIG ---
DATASET_DIR = "../data/raw/dataset/raw_images"
LABEL_ID = "5"

os.makedirs(DATASET_DIR, exist_ok=True)

# --- PIPELINE SETUP ---
pipeline = dai.Pipeline()
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define Color Camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setPreviewSize(640, 640)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# XLinkOut (Output to host)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")

cam.preview.link(xout.input)

print(f"📷 OAK-D Camera initialized.")
print(f"🎯 Target Label ID: '{LABEL_ID}'")
print(f"⌨️  Controls: [SPACE] to capture, [Q] to quit")

# Connect to device
try:
    with dai.Device(pipeline) as device:
        print(f"ℹ️  DepthAI Version: {dai.__version__}")

        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            in_rgb = q_rgb.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                cv2.imshow("Data Capture - OAK-D", frame)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    break
                elif key == 32:  # SPACE bar
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{DATASET_DIR}/{LABEL_ID}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✅ Saved: {filename}")

                    cv2.imshow("Data Capture - OAK-D", cv2.bitwise_not(frame))
                    cv2.waitKey(50)

except AttributeError as e:
    print(f"\n🔴 ERROR: {e}")
except Exception as e:
    print(f"\n🔴 UNEXPECTED ERROR: {e}")

cv2.destroyAllWindows()