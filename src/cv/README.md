# Complete Guide: Object Detection with Roboflow & OAK-D Pro

## End-to-End Pipeline from Dataset to Deployment

This guide walks you through the complete process of creating a custom object detection system using Roboflow for dataset management and OAK-D Pro camera for deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Dataset Preparation (Roboflow)](#phase-1-dataset-preparation-roboflow)
4. [Phase 2: Local Model Training](#phase-2-local-model-training)
5. [Phase 3: Model Conversion for OAK-D Pro](#phase-3-model-conversion-for-oak-d-pro)
6. [Phase 4: Deployment & Detection](#phase-4-deployment--detection)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [Advanced Topics](#advanced-topics)

---

## Overview

### What You'll Build

A complete object detection system that:
- Detects custom objects in real-time (30 FPS)
- Provides depth information for each detected object
- Runs entirely on-device (no cloud/API needed)
- Works offline with no recurring costs

### Technology Stack

- **Dataset Management**: Roboflow (free tier)
- **Training**: YOLOv8 (Ultralytics, open-source)
- **Hardware**: OAK-D Pro camera
- **Framework**: DepthAI SDK
- **Languages**: Python

### Time Requirements

- Dataset preparation: 1-4 hours (depends on dataset size)
- Model training: 30 minutes - 2 hours (depends on hardware)
- Model conversion: 5-10 minutes
- Deployment setup: 10 minutes
- **Total**: ~2-6 hours for complete pipeline

---

## Prerequisites

### Hardware

**Required:**
- OAK-D Pro camera
- Computer with:
  - 8GB+ RAM
  - 10GB+ free storage
  - USB 3.0 port

### Accounts

**Roboflow Account** (free):
1. Go to https://roboflow.com
2. Sign up for free account
3. Get your API key from Settings → Roboflow API

---

## Phase 1: Dataset Preparation (Roboflow)

### Step 1.1: Create Project in Roboflow

1. **Log in** to Roboflow
2. Click **"Create New Project"**
3. Configure:
   - **Project Name**: e.g., "connected-shelf-object-detection"
   - **Project Type**: Object Detection
   - **Annotation Group**: Choose your use case
4. Click **"Create Project"**

### Step 1.2: Upload Videos

1. Click **"Upload"** button
2. Select video files from your computer
3. Roboflow will automatically extract frames from your videos
4. Review and select the frames you want to keep

**Best Practices:**
- Use videos with variety: different angles, lighting, backgrounds
- Aim for 100-500 frames minimum (Roboflow extracts these from videos)
- Higher is better: 1000+ frames = excellent results
- Video/frame quality: 640x640 to 1920x1080 pixels
- Record videos that capture your objects in realistic conditions

### Step 1.3: Annotate Frames

Roboflow's annotation tool is intuitive - simply draw bounding boxes around objects and label them.

**Annotation Best Practices:**
- Box should tightly fit object
- Include entire object (don't cut off parts)
- Consistent labeling across all frames
- Label all instances in each frame

### Step 1.4: Generate Dataset Version

1. **Click "Generate"** (after annotation)
2. **Configure Preprocessing**:
   - **Auto-Orient**: ✓ Recommended
   - **Resize**: 640x640 (standard for YOLO)
   - **Grayscale**: ✗ (unless needed)

3. **Configure Augmentation**:
   
   **Recommended Settings:**
   - **Flip**: Horizontal (if objects can be flipped)
   - **Rotation**: ±15° (for varied angles)
   - **Brightness**: ±15% (for lighting variation)
   - **Exposure**: ±15% (for camera variance)
   - **Blur**: Up to 1px (for motion blur)

4. **Train/Valid/Test Split**:
   - Training: 70%
   - Validation: 20%
   - Test: 10%

5. **Click "Generate"**

### Step 1.5: Review Dataset

1. **Check "Health Check"** tab
   - Class balance
   - Image dimensions
   - Annotation quality

2. **View Examples**
   - Verify augmentations look good
   - Check for annotation errors

3. **Fix Issues** if needed:
   - Re-annotate problematic images
   - Adjust augmentation settings
   - Re-generate version

**Your dataset is now ready for download!**

---

## Phase 2: Local Model Training

### Step 2.1: Install Dependencies

This project uses Poetry for dependency management. The repository includes a `pyproject.toml` file.

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies from pyproject.toml
poetry install

# Activate the virtual environment
poetry shell
```

### Step 2.2: Download Dataset from Roboflow

Use the provided `train.py` script:

1. **Edit configuration** in `train.py`:
   ```python
   API_KEY = "your_api_key_here"
   WORKSPACE = "your_workspace"
   PROJECT = "connected-shelf-object-detection"
   VERSION = 4  # Your dataset version
   ```

2. **Run the script**:
   ```bash
   python train.py
   ```

The script will:
- Download dataset automatically
- Set up training
- Guide you through the process

**Dataset Structure:**
```
dataset/
├── data.yaml          # Dataset configuration
├── train/
│   ├── images/        # Training images
│   └── labels/        # Training annotations
├── valid/
│   ├── images/        # Validation images
│   └── labels/        # Validation annotations
└── test/
    ├── images/        # Test images
    └── labels/        # Test annotations
```

### Step 2.3: Configure Training

**Choose Model Size:**

| Model | Parameters | Size | Speed | Accuracy | Best For |
|-------|-----------|------|-------|----------|----------|
| YOLOv8n | 3.2M | 6MB | Fastest | Good | Edge devices, real-time |
| YOLOv8s | 11.2M | 22MB | Fast | Better | Balanced use cases |
| YOLOv8m | 25.9M | 52MB | Medium | Best | Maximum accuracy |
| YOLOv8l | 43.7M | 88MB | Slow | Excellent | GPU deployment |
| YOLOv8x | 68.2M | 136MB | Slowest | Best | Research, accuracy priority |

**Recommendation for OAK-D Pro**: YOLOv8n or YOLOv8s

### Step 2.4: Train the Model
**Training Parameters Explained:**

- `epochs=100`: How many times to iterate through dataset
  - More epochs = better learning (but can overfit)
  - Start with 100, increase if needed
  
- `imgsz=640`: Input image size
  - 640x640 is standard
  - Larger = more accurate but slower
  
- `batch=16`: Images per training step
  - Larger = faster but needs more GPU memory
  - If out of memory, reduce to 8, 4, or even 1
  
- `patience=50`: Stop if no improvement for N epochs
  - Prevents overfitting and saves time
  
- `device='0'`: Hardware to use
  - `'0'` = First GPU (NVIDIA)
  - `'cpu'` = CPU only (slower)
  - `'mps'` = Mac M1/M2/M3 acceleration

**Monitor Training:**

Training creates real-time plots in:
```
runs/train/oakd_detection/
├── weights/
│   ├── best.pt              # Best model checkpoint
│   ├── last.pt              # Latest checkpoint
├── results.png              # Training metrics over time
├── confusion_matrix.png     # Class confusion
├── F1_curve.png            # F1 score by confidence
├── PR_curve.png            # Precision-Recall
└── val_batch0_labels.jpg   # Sample predictions
```

Open `results.png` to see:
- Loss curves (should decrease)
- mAP (should increase)
- Precision/Recall

**Training Tips:**

1. **Watch for Overfitting**:
   - Training mAP much higher than validation mAP
   - Solution: More data, more augmentation, early stopping

2. **Watch for Underfitting**:
   - Both training and validation mAP are low
   - Solution: Train longer, bigger model, better data

3. **Optimal Training**:
   - Both metrics improving together
   - Validation mAP close to training mAP
   - This is what you want!

### Step 2.5: Evaluate Results

**Good Results:**
- mAP50 > 0.9 (90%+)
- mAP50-95 > 0.7 (70%+)
- Precision > 0.85 (85%+)
- Recall > 0.85 (85%+)

**If results are poor:**
- Train for more epochs (200-300)
- Use larger model (yolov8s or yolov8m)
- Improve dataset (more images, better annotations)
- Adjust augmentation settings

---

## Phase 3: Model Conversion for OAK-D Pro

**Note:** If you used `train_local.py` for training, it should have already completed these conversion steps automatically. This section explains what happens during conversion and is useful for troubleshooting or manual conversion if needed.

### Step 3.1: Export to ONNX

Exports your model to ONNX format.

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/train/oakd_detection/weights/best.pt')

# Export to ONNX
onnx_path = model.export(
    format='onnx',
    imgsz=640,
    simplify=True
)

print(f"ONNX model saved: {onnx_path}")
```

**Output location:**
```
runs/train/oakd_detection/weights/best.onnx
```

### Step 3.2: Convert ONNX to Blob

Converts ONNX to blob format for OAK-D Pro.

```python
import blobconverter
import shutil
import os

# Convert ONNX to blob
print("Converting to blob format...")
blob_path = blobconverter.from_onnx(
    model="runs/train/oakd_detection/weights/best.onnx",
    data_type="FP16",      # Half precision
    shaves=6,              # Number of SHAVE cores
    use_cache=True         # Cache for faster re-conversion
)

print(f"Blob created: {blob_path}")

# Copy to model directory
os.makedirs("model", exist_ok=True)
shutil.copy(blob_path, "model/model.blob")

print("✓ Model ready at: model/model.blob")
```

**Conversion Settings Explained:**

- **FP16 vs FP32**: 
  - FP16 = Half precision, 2x faster, smaller file
  - FP32 = Full precision, slightly more accurate
  - Recommendation: FP16

- **Shaves**: 
  - More shaves = faster inference
  - OAK-D Pro has 16 shaves
  - Use 6 for good balance
  - Can try 8-12 for more speed

### Step 3.3: Create Labels File

Creates a labels file from your dataset.

```python
# This is done automatically by train_local.py
import yaml
import os

# Read class names from data.yaml
with open('dataset/data.yaml', 'r') as f:
    data = yaml.safe_load(f)

class_names = data['names']

# Create labels file
os.makedirs("model", exist_ok=True)
with open('model/labels.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print(f"✓ Labels saved: model/labels.txt")
print(f"Classes: {', '.join(class_names)}")
```

**After `train.py` completes**, you should have:
```
model/
├── model.blob          # For OAK-D Pro inference
└── labels.txt          # Class names
```

---

## Phase 4: Deployment & Detection

### Step 4.1: Verify OAK-D Pro Connection

```bash
python test_oakd_setup.py
```

**Expected Output:**
```
======================================================================
OAK-D Pro Setup Verification
======================================================================

[1/5] Checking depthai installation...
✓ depthai version: 2.x.x

[2/5] Checking OpenCV installation...
✓ OpenCV version: 4.x.x

[3/5] Checking other dependencies...
✓ NumPy version: 1.x.x
✓ Requests version: 2.x.x

[4/5] Searching for OAK-D Pro devices...
✓ Found 1 device(s):
  - 14442C10D1XXXXXX (USB3)

[5/5] Testing device connection...
✓ Successfully connected to: OAK-D Pro
  - Product Name: OAK-D Pro
  - Connected cameras: [RGB, LEFT, RIGHT]

======================================================================
✓ All checks passed! Your OAK-D Pro is ready to use.
======================================================================
```

**If errors occur**, see [Troubleshooting](#troubleshooting) section.

### Step 4.2: Configure Detection Script

Edit `detect_yolo_oakd.py`:

```python
# Update these paths
MODEL_BLOB_PATH = "model/model.blob"
LABELS_PATH = "model/labels.txt"

# Adjust detection parameters
CONFIDENCE_THRESHOLD = 0.5  # Lower = more detections
IOU_THRESHOLD = 0.5         # Overlap threshold

# Camera settings
CAMERA_FPS = 30
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 640

# Display options
SHOW_FPS = True
SHOW_LABELS = True
SHOW_CONFIDENCE = True
SHOW_DEPTH = True
```

### Step 4.3: Run Detection

**Basic Command:**
```bash
python detect_yolo_oakd.py
```

**With Options:**
```bash
# Custom confidence threshold
python detect_yolo_oakd.py --conf 0.4

# Disable depth display
python detect_yolo_oakd.py --no-depth

# Save video recording
python detect_yolo_oakd.py --save-video --output detections.mp4

# Custom model path
python detect_yolo_oakd.py --model path/to/model.blob --labels path/to/labels.txt
```

### Step 4.4: Real-Time Controls

While detection is running:

| Key | Function |
|-----|----------|
| `q` | Quit application |
| `s` | Save current frame as image |
| `d` | Toggle depth display on/off |
| `c` | Toggle confidence display |
| `+` or `=` | Increase confidence threshold |
| `-` or `_` | Decrease confidence threshold |

### Step 4.5: Interpret Results

**On-Screen Display:**

```
┌─────────────────────────────────────────┐
│ FPS: 28.5                               │
│ Detections: 3                           │
│ Threshold: 0.50                         │
│                                         │
│  ┌──────────────────────┐               │
│  │ product_a | 98.5% |  │               │
│  │         0.45m        │               │
│  └──────────────────────┘               │
│                                         │
└─────────────────────────────────────────┘
```

**Information Shown:**
- **Class name**: What object was detected
- **Confidence**: How sure the model is (%)
- **Depth**: Distance to object (meters)
- **FPS**: Frames per second
- **Threshold**: Current confidence threshold

### Step 4.6: Fine-Tune Detection

**Adjust Confidence Threshold:**

Too many false positives?
- Press `+` to increase threshold
- Or set `--conf 0.7` for higher confidence

Missing detections?
- Press `-` to decrease threshold
- Or set `--conf 0.3` for lower confidence

**Adjust IOU (in code):**
- Higher IOU = fewer overlapping boxes
- Lower IOU = more boxes kept

---

## Troubleshooting

### Training Issues

**Problem: "CUDA out of memory"**
```python
# Reduce batch size
model.train(..., batch=8)  # or 4, or 1
```

**Problem: "Overfitting"**
- Add more training data
- Increase augmentation in Roboflow
- Enable early stopping: `patience=50`
- Reduce model size

**Problem: "Low mAP"**
- Improve dataset quality
- Train longer
- Use larger model
- Check annotations are correct

### Conversion Issues

**Problem: "Blob conversion failed"**
- Use online converter: https://blobconverter.luxonis.com/
- Check ONNX file exists
- Try different OpenVINO version
- Reinstall blobconverter: `pip install --upgrade blobconverter`

**Problem: "Model too large for OAK"**
- Use smaller model (yolov8n)
- Reduce input size: `imgsz=416`
- Use FP16 instead of FP32

---

## File Organization

```
your-project/
├── dataset/                          # From Roboflow
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
│
├── runs/train/oakd_detection/       # Training outputs
│   ├── weights/
│   │   ├── best.pt                  # PyTorch model
│   │   ├── best.onnx                # ONNX export
│   │   └── last.pt                  # Last checkpoint
│   ├── results.png                  # Training curves
│   └── confusion_matrix.png
│
├── model/                           # Deployment files
│   ├── model.blob                   # For OAK-D Pro
│   └── labels.txt                   # Class names
│
├── train_local.py                   # Training script
├── detect_yolo_oakd.py              # Detection script
├── test_oakd_setup.py               # Hardware test
└── requirements.txt                 # Dependencies
```
