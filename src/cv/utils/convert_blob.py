"""
convert_blob.py — run on Mac
Converts best.pt -> ONNX -> blob with --reverse_input_channels explicitly set.

python src/cv/convert_blob.py
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

PT_PATH = "runs/detect/runs/train/oakd_detection/weights/best.pt"

# ---- Load and confirm model ------------------------------------------------
model = YOLO(PT_PATH)
imgsz = model.overrides.get("imgsz", 640)
print(f"Model: {PT_PATH}")
print(f"  imgsz={imgsz}  nc={model.model.nc}  names={list(model.names.values())}")

# ---- Export ONNX -----------------------------------------------------------
print(f"\nExporting ONNX at imgsz={imgsz}...")
onnx_path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
print(f"  -> {onnx_path}")

# ---- Convert to blob -------------------------------------------------------
print("\nConverting to blob...")
print("  Using --reverse_input_channels to fix BGR->RGB mismatch\n")

import blobconverter

blob_path = blobconverter.from_onnx(
    model=str(onnx_path),
    data_type="FP16",
    shaves=6,
    use_cache=False,
    optimizer_params=[
        "--reverse_input_channels",
        "--mean_values=[123.675,116.28,103.53]",
        "--scale_values=[58.395,57.12,57.375]",
    ],
)
print(f"  -> {blob_path}")

# ---- Save ------------------------------------------------------------------
Path("model").mkdir(exist_ok=True)
shutil.copy(blob_path, "model/model.blob")

with open("model/labels.txt", "w") as f:
    for name in model.names.values():
        f.write(f"{name}\n")

print(f"\n✓ model/model.blob updated")
print(f"✓ model/labels.txt updated")
print(f"\nCopy both files to the Pi, then run:")
print(f"  poetry run python src/test/test_model.py")