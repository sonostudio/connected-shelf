from ultralytics import YOLO
import cv2

# Load model
model = YOLO('../cv/runs/detect/runs/train/oakd_detection/weights/best.pt')

# Test on an image
results = model('../cv/connected-shelf-object-detection-4/valid/images/0_20260130_161633_linger_mp4-0064_jpg.rf.80c134daa053247433b6fd1a7011bcc3.jpg')

# Show results
results[0].show()
