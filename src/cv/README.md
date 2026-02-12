# Object Detection with Roboflow & OAK-D Pro

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
   - **Resize**: 320x320 (matches model input size)
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
src/cv/data/project-name-version/  # From Roboflow (rename to match your download)
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

Run training via the YOLO CLI:

```bash
yolo detect train \
  data=src/cv/data/project-name-version/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=320 \
  device=mps \
  batch=16 \
  project=src/cv/runs \
  name=detect \
  exist_ok=True
```

**Training Parameters Explained:**

- `epochs=50`: How many times to iterate through dataset
  - More epochs = better learning (but can overfit)
  - Start with 50, increase if needed
  
- `imgsz=320`: Input image size
  - 320x320 matches the OAK-D capture size in this project
  - Larger = more accurate but slower
  
- `batch=16`: Images per training step
  - Larger = faster but needs more GPU memory
  - If out of memory, reduce to 8, 4, or even 1
  
- `patience=50`: Stop if no improvement for N epochs
  - Prevents overfitting and saves time
  
- `device`: Hardware to use
  - `mps` = Mac Apple Silicon (M1/M2/M3) acceleration
  - `0` = First GPU (NVIDIA)
  - `cpu` = CPU only (slower)

**Monitor Training:**

Training creates real-time plots in:
```
src/cv/runs/detect/
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
- Train for more epochs (100-200)
- Use larger model (yolov8s or yolov8m)
- Improve dataset (more images, better annotations)
- Adjust augmentation settings

---

## Phase 3: Model Conversion for OAK-D Pro

The trained PyTorch model (`.pt`) needs to be converted to `.blob` format before it can run on the OAK-D Pro. This is done using the Luxonis online converter, which handles the full conversion internally.

### Step 3.1: Convert to Blob (Online Converter)

Blob conversion is done using the **Luxonis online converter** at:

👉 **https://tools.luxonis.com**

**Steps:**

1. Go to https://tools.luxonis.com
2. Upload your `best.pt` file from `src/cv/runs/detect/weights/`
3. Configure conversion settings:
   - **Data Type**: FP16
   - **Shaves**: 6
4. Click **Convert**
5. Download the resulting `.blob` file
6. Place it at `src/cv/models/model.blob`

**Conversion Settings Explained:**

- **FP16 vs FP32**: 
  - FP16 = Half precision, 2x faster, smaller file
  - FP32 = Full precision, slightly more accurate
  - Recommendation: FP16

- **Shaves**: 
  - More shaves = faster inference
  - OAK-D Pro has 16 shaves
  - Use 6 for a good balance
  - Can try 8-12 for more speed

### Step 3.2: Create Labels File

Use the provided utility script at `src/cv/utils/create_labels.py`.

Before running, update the dataset path in the script to match your Roboflow download:

```python
# create_labels.py
with open('src/cv/data/project-name-version/data.yaml', 'r') as f:
```

Then run it:

```bash
python src/cv/utils/create_labels.py
```

**After conversion completes**, you should have:
```
src/cv/models/
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
MODEL_BLOB_PATH = "src/cv/models/model.blob"
LABELS_PATH = "src/cv/models/labels.txt"

# Adjust detection parameters
CONFIDENCE_THRESHOLD = 0.5  # Lower = more detections
IOU_THRESHOLD = 0.5         # Overlap threshold

# Camera settings
CAMERA_FPS = 30
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 320

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

# Enable depth display
python detect_yolo_oakd.py --depth

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
```bash
# Reduce batch size
yolo detect train ... batch=8  # or 4, or 1
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
- Use the online converter: https://tools.luxonis.com
- Check that `best.pt` exists in `src/cv/runs/detect/weights/`
- Try a different number of shaves

**Problem: "Model too large for OAK"**
- Use smaller model (yolov8n)
- Reduce input size: `imgsz=320` (already the default in this project)
- Use FP16 instead of FP32

---

## File Organization

```
connected-shelf/
├── src/
│   └── cv/
│       ├── data/
│       │   └── project-name-version/          # From Roboflow
│       │       ├── data.yaml
│       │           ├── train/
│       │           ├── valid/
│       │           └── test/
│       ├── runs/
│       │   └── detect/                # Training outputs
│       │       ├── weights/
│       │       │   ├── best.pt        # PyTorch model
│       │       │   └── last.pt        # Last checkpoint
│       │       ├── results.png        # Training curves
│       │       └── confusion_matrix.png
│       └── models/                    # Deployment files
│           ├── model.blob             # For OAK-D Pro
│           └── labels.txt            # Class names
│
├── videos/                            # Raw capture videos
├── config/                            # Configuration files
├── pyproject.toml                     # Poetry dependencies
├── train.py                           # Training script
├── detect_yolo_oakd.py                # Detection script
└── test_oakd_setup.py                 # Hardware test
```