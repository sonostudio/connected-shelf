import blobconverter
import shutil
import os
from pathlib import Path
from ultralytics import YOLO

path_weights = Path("../runs/detect/runs/train/oakd_detection/weights/")
model = YOLO(f"{path_weights}/best.pt")

# Export to ONNX with explicit settings
onnx_path = model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False
)

print(f"ONNX model saved: {onnx_path}")

# Convert ONNX to blob
print("Converting to blob format...")
blob_path = blobconverter.from_onnx(
    model=f"{path_weights}/best.onnx",
    data_type="FP16",
    shaves=6,
    use_cache=False,
    optimizer_params=[
        "--data_type=FP16",
        "--reverse_input_channels",
        "--scale_values=[255,255,255]"
    ]
)
print(f"Blob created: {blob_path}")

os.makedirs("../model", exist_ok=True)
shutil.copy(blob_path, "../model/model.blob")
print("✓ Model ready at: model/model.blob")
