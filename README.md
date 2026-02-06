# connected-shelf
## Train YOLOv8 model
```commandline
yolo detect train \
data=src/cv/data/processed/roboflow_dataset/data.yaml \
model=yolov8n.pt \
epochs=50 \
imgsz=320 \
device=mps \
batch=16 \
project=src/cv/runs \
name=detect \
exist_ok=True
```

## Convert output to blob format
```commandline
yolo export model=src/cv/runs/run_3/detect/weights/best.pt format=onnx imgsz=320 opset=12
blobconverter --onnx src/cv/runs/classify/train/weights/best.onnx \
--shaves 6 \
--output-dir src/cv/models \
--no-cache
```


