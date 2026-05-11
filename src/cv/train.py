import os
import shutil

print("=" * 60)
print("Local Model Training for OAK-D Pro")
print("=" * 60)
print()
print("This script will help you:")
print("1. Download your dataset from Roboflow (free)")
print("2. Fine-tune a YOLO model locally")
print("3. Convert to OAK-D Pro format")
print()

# Configuration
API_KEY = ""
WORKSPACE = ""
PROJECT = ""
VERSION = 2
DATASET_PATH = ""  # Set this to skip Roboflow download, e.g. "dataset/your-project-2"

# Fine-tuning configuration
FINETUNE_WEIGHTS = "src/cv/runs/detect/runs/train/oakd_detection2/weights/best.pt"


def download_dataset():
    """
    Download dataset from Roboflow (this IS available on free plan)
    """
    print("\n" + "=" * 60)
    print("Step 1: Download Dataset from Roboflow")
    print("=" * 60)

    try:
        from roboflow import Roboflow

        print("\nConnecting to Roboflow...")
        rf = Roboflow(api_key=API_KEY)

        print(f"Accessing workspace: {WORKSPACE}")
        project = rf.workspace(WORKSPACE).project(PROJECT)

        print(f"Downloading version {VERSION}...")
        dataset = project.version(VERSION).download("yolov8")

        print(f"\n✓ Dataset downloaded successfully!")
        print(f"Location: {dataset.location}")

        return dataset.location

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nManual alternative:")
        print("1. Go to app.roboflow.com")
        print(f"2. Your project → Version {VERSION}")
        print("3. Click 'Download Dataset' (this IS free)")
        print("4. Choose format: 'YOLOv8'")
        print("5. Download and extract to 'dataset/' folder")
        print("6. Set DATASET_PATH in this script to the extracted folder path")
        return None


def install_ultralytics():
    """
    Install YOLOv8 (Ultralytics)
    """
    print("\n" + "=" * 60)
    print("Step 2: Install YOLOv8")
    print("=" * 60)

    try:
        import ultralytics
        print("✓ Ultralytics already installed")
        return True
    except ImportError:
        print("\nInstalling Ultralytics (YOLOv8)...")
        os.system("pip install ultralytics")

        try:
            import ultralytics
            print("✓ Installation successful!")
            return True
        except ImportError:
            print("✗ Installation failed")
            return False


def train_model(dataset_path, finetune_weights=None):
    """
    Fine-tune or train YOLOv8 model on your dataset
    """
    print("\n" + "=" * 60)
    print("Step 3: Train Model")
    print("=" * 60)

    from ultralytics import YOLO

    # Find data.yaml
    data_yaml = None
    if dataset_path:
        data_yaml = os.path.join(dataset_path, "data.yaml")
    else:
        for root, dirs, files in os.walk("."):
            if "data.yaml" in files:
                data_yaml = os.path.join(root, "data.yaml")
                break

    if not data_yaml or not os.path.exists(data_yaml):
        print("✗ data.yaml not found!")
        print("Please ensure your dataset is downloaded.")
        return None

    print(f"\nUsing dataset: {data_yaml}")

    # Determine whether fine-tuning or training from scratch
    if finetune_weights and os.path.exists(finetune_weights):
        print(f"\n✓ Fine-tuning from: {finetune_weights}")
        print("  Model size is inherited from the existing weights (yolov8s)")
        model = YOLO(finetune_weights)
        run_name = "oakd_finetune"
        epochs = 50
        lr0 = 0.001   # 10x lower than default to preserve learned features
        lrf = 0.01
        warmup_epochs = 3
    else:
        print("\n⚠️  Fine-tune weights not found, falling back to training from scratch")
        print("\nSelect model size:")
        print("1. YOLOv8n (nano - fastest, 6MB)")
        print("2. YOLOv8s (small - balanced, 22MB)")
        print("3. YOLOv8m (medium - accurate, 52MB)")

        choice = input("\nEnter choice (1-3) [default: 2]: ").strip() or "2"
        model_sizes = {
            "1": "yolov8n.pt",
            "2": "yolov8s.pt",
            "3": "yolov8m.pt"
        }
        model_file = model_sizes.get(choice, "yolov8s.pt")
        print(f"\nUsing model: {model_file}")
        model = YOLO(model_file)
        run_name = "oakd_detection"
        epochs = 100
        lr0 = 0.01    # default learning rate for training from scratch
        lrf = 0.01
        warmup_epochs = 3

    print(f"\nTraining configuration:")
    print(f"  - Epochs:         {epochs}")
    print(f"  - Image size:     640")
    print(f"  - Batch size:     16 (auto-adjusted based on GPU)")
    print(f"  - Learning rate:  {lr0} (initial), {lrf} (final)")
    print(f"  - Warmup epochs:  {warmup_epochs}")
    print(f"  - Device:         auto (GPU if available, else CPU)")

    input("\nPress Enter to start training...")

    print("\n🚀 Starting training...")
    if finetune_weights and os.path.exists(finetune_weights):
        print("Fine-tuning typically converges faster than training from scratch.")
    else:
        print("This may take 30min - 2 hours depending on your hardware")
    print("You can stop anytime with Ctrl+C and use the best checkpoint")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        lr0=lr0,
        lrf=lrf,
        warmup_epochs=warmup_epochs,
        name=run_name,
        project="runs/train"
    )

    print("\n✓ Training complete!")

    best_model = f"runs/detect/runs/train/{run_name}/weights/best.pt"

    if os.path.exists(best_model):
        print(f"Best model: {best_model}")
        return best_model
    else:
        print("✗ Model file not found")
        return None


def export_to_onnx(model_path):
    """
    Export trained model to ONNX format
    """
    print("\n" + "=" * 60)
    print("Step 4: Export to ONNX")
    print("=" * 60)

    from ultralytics import YOLO

    if not model_path or not os.path.exists(model_path):
        print("✗ Model file not found!")
        return None

    print(f"\nExporting: {model_path}")

    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=640)

    print(f"\n✓ ONNX export complete!")
    print(f"File: {onnx_path}")

    return onnx_path


def convert_to_blob(onnx_path):
    """
    Convert ONNX to OAK-D Pro blob format
    """
    print("\n" + "=" * 60)
    print("Step 5: Convert to Blob")
    print("=" * 60)

    if not onnx_path or not os.path.exists(onnx_path):
        print("✗ ONNX file not found!")
        return None

    try:
        import blobconverter
    except ImportError:
        print("\nInstalling blobconverter...")
        os.system("pip install blobconverter")
        import blobconverter

    print(f"\nConverting: {onnx_path}")
    print("This may take several minutes...")

    try:
        blob_path = blobconverter.from_onnx(
            model=onnx_path,
            data_type="FP16",
            shaves=6,
            use_cache=True
        )

        print(f"\n✓ Blob conversion complete!")
        print(f"File: {blob_path}")

        os.makedirs("model", exist_ok=True)
        final_path = "model/model.blob"
        shutil.copy(blob_path, final_path)
        print(f"Copied to: {final_path}")

        return final_path

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        print("\nAlternative: Use online converter")
        print("1. Go to https://blobconverter.luxonis.com/")
        print(f"2. Upload: {onnx_path}")
        print("3. Settings: FP16, 6 shaves, OpenVINO 2021.4")
        print("4. Download and save as model/model.blob")
        return None


def create_labels_file(dataset_path):
    """
    Create labels.txt file from dataset
    """
    print("\n" + "=" * 60)
    print("Creating labels file")
    print("=" * 60)

    import yaml

    data_yaml = None
    if dataset_path:
        data_yaml = os.path.join(dataset_path, "data.yaml")
    else:
        for root, dirs, files in os.walk("."):
            if "data.yaml" in files:
                data_yaml = os.path.join(root, "data.yaml")
                break

    if not data_yaml or not os.path.exists(data_yaml):
        print("✗ data.yaml not found")
        return False

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    names = data.get('names', [])

    if not names:
        print("✗ No class names found in data.yaml")
        return False

    os.makedirs("model", exist_ok=True)
    labels_path = "model/labels.txt"

    with open(labels_path, 'w') as f:
        for name in names:
            f.write(f"{name}\n")

    print(f"✓ Created: {labels_path}")
    print(f"Classes: {', '.join(names)}")

    return True


def main():
    """
    Main training pipeline
    """
    print("\n⚠️  IMPORTANT:")
    print("Make sure you have configured:")
    print("  - API_KEY (or DATASET_PATH for manual download)")
    print("  - WORKSPACE")
    print("  - PROJECT")
    print(f"  - FINETUNE_WEIGHTS (currently: {FINETUNE_WEIGHTS})")
    print()

    # Check fine-tune weights
    if FINETUNE_WEIGHTS and os.path.exists(FINETUNE_WEIGHTS):
        print(f"✓ Fine-tune weights found: {FINETUNE_WEIGHTS}")
        finetune_weights = FINETUNE_WEIGHTS
    else:
        print(f"⚠️  Fine-tune weights not found at: {FINETUNE_WEIGHTS}")
        print("   Will fall back to training from scratch if not resolved.")
        finetune_weights = None

    # Step 1: Resolve dataset path
    if DATASET_PATH:
        if not os.path.exists(DATASET_PATH):
            print(f"✗ DATASET_PATH not found: '{DATASET_PATH}'")
            return
        print(f"✓ Using manually specified dataset: {DATASET_PATH}")
        dataset_path = DATASET_PATH
    else:
        if not API_KEY:
            print("✗ Please set either API_KEY or DATASET_PATH in this script first!")
            return
        dataset_path = download_dataset()

    if not dataset_path:
        print("\n⚠️  No dataset available.")
        print("Either set DATASET_PATH to your extracted dataset folder,")
        print("or fix the Roboflow download issue and try again.")
        return

    # Step 2: Install Ultralytics
    if not install_ultralytics():
        return

    # Create labels file
    create_labels_file(dataset_path)

    # Step 3: Train model
    print("\n" + "=" * 60)
    choice = input("Start training now? (y/n): ").strip().lower()

    if choice != 'y':
        print("\nYou can train later by running:")
        print("  python train.py")
        return

    model_path = train_model(dataset_path, finetune_weights)

    if not model_path:
        return

    # Step 4: Export to ONNX
    onnx_path = export_to_onnx(model_path)

    if not onnx_path:
        return

    # Step 5: Convert to blob
    blob_path = convert_to_blob(onnx_path)

    if blob_path:
        print("\n" + "=" * 60)
        print("✓ SUCCESS! Your model is ready!")
        print("=" * 60)
        print(f"\nModel files created:")
        print(f"  - {blob_path}")
        print(f"  - model/labels.txt")
        print(f"\nYou can now run:")
        print(f"  python detect.py")


if __name__ == "__main__":
    main()