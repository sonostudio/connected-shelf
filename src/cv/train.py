import os

print("="*60)
print("Local Model Training for OAK-D Pro")
print("="*60)
print()
print("This script will help you:")
print("1. Download your dataset from Roboflow (free)")
print("2. Train a YOLO model locally")
print("3. Convert to OAK-D Pro format")
print()

# Configuration
API_KEY = "wBdl9uQ6INwB93C7nlKn"
WORKSPACE = "sono-studio"
PROJECT = "connected-shelf-object-detection"
VERSION = 2

def download_dataset():
    """
    Download dataset from Roboflow (this IS available on free plan)
    """
    print("\n" + "="*60)
    print("Step 1: Download Dataset from Roboflow")
    print("="*60)
    
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
        print("2. Your project → Version 4")
        print("3. Click 'Download Dataset' (this IS free)")
        print("4. Choose format: 'YOLOv8'")
        print("5. Download and extract to 'dataset/' folder")
        return None


def install_ultralytics():
    """
    Install YOLOv8 (Ultralytics)
    """
    print("\n" + "="*60)
    print("Step 2: Install YOLOv8")
    print("="*60)
    
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


def train_model(dataset_path):
    """
    Train YOLOv8 model on your dataset
    """
    print("\n" + "="*60)
    print("Step 3: Train Model")
    print("="*60)
    
    from ultralytics import YOLO
    
    # Find data.yaml
    data_yaml = None
    if dataset_path:
        data_yaml = os.path.join(dataset_path, "data.yaml")
    else:
        # Search for it
        for root, dirs, files in os.walk("."):
            if "data.yaml" in files:
                data_yaml = os.path.join(root, "data.yaml")
                break
    
    if not data_yaml or not os.path.exists(data_yaml):
        print("✗ data.yaml not found!")
        print("Please ensure your dataset is downloaded.")
        return None
    
    print(f"\nUsing dataset: {data_yaml}")
    
    # Training configuration
    print("\nSelect model size:")
    print("1. YOLOv8n (nano - fastest, 6MB)")
    print("2. YOLOv8s (small - balanced, 22MB)")
    print("3. YOLOv8m (medium - accurate, 52MB)")
    
    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"
    
    model_sizes = {
        "1": "yolov8n.pt",
        "2": "yolov8s.pt", 
        "3": "yolov8m.pt"
    }
    
    model_file = model_sizes.get(choice, "yolov8n.pt")
    
    print(f"\nUsing model: {model_file}")
    print("\nTraining configuration:")
    print("  - Epochs: 100 (you can change this)")
    print("  - Image size: 640")
    print("  - Batch size: 16 (auto-adjusted based on GPU)")
    print("  - Device: auto (GPU if available, else CPU)")
    
    input("\nPress Enter to start training...")
    
    # Load model
    model = YOLO(model_file)
    
    # Train
    print("\n🚀 Starting training...")
    print("This may take 30min - 2 hours depending on your hardware")
    print("You can stop anytime with Ctrl+C and use the best checkpoint")
    
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name="oakd_detection",
        project="runs/train"
    )
    
    print("\n✓ Training complete!")
    
    # Get best model path
    best_model = "runs/detect/runs/train/oakd_detection/weights/best.pt"
    
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
    print("\n" + "="*60)
    print("Step 4: Export to ONNX")
    print("="*60)
    
    from ultralytics import YOLO
    
    if not model_path or not os.path.exists(model_path):
        print("✗ Model file not found!")
        return None
    
    print(f"\nExporting: {model_path}")
    
    model = YOLO(model_path)
    
    # Export to ONNX
    onnx_path = model.export(format="onnx", imgsz=640)
    
    print(f"\n✓ ONNX export complete!")
    print(f"File: {onnx_path}")
    
    return onnx_path


def convert_to_blob(onnx_path):
    """
    Convert ONNX to OAK-D Pro blob format
    """
    print("\n" + "="*60)
    print("Step 5: Convert to Blob")
    print("="*60)
    
    if not onnx_path or not os.path.exists(onnx_path):
        print("✗ ONNX file not found!")
        return None
    
    # Install blobconverter
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
        
        # Copy to model directory
        os.makedirs("model", exist_ok=True)
        import shutil
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
    print("\n" + "="*60)
    print("Creating labels file")
    print("="*60)
    
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
    
    # Create labels file
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
    print("  - API_KEY")
    print("  - WORKSPACE")
    print("  - PROJECT")
    print()
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("✗ Please set your API_KEY in this script first!")
        return
    
    # Step 1: Download dataset
    dataset_path = download_dataset()
    
    if not dataset_path:
        print("\n⚠️  Manual download required")
        print("After downloading, run this script again")
        return
    
    # Step 2: Install Ultralytics
    if not install_ultralytics():
        return
    
    # Create labels file
    create_labels_file(dataset_path)
    
    # Step 3: Train model
    print("\n" + "="*60)
    choice = input("Start training now? (y/n): ").strip().lower()
    
    if choice != 'y':
        print("\nYou can train later by running:")
        print("  python train_local.py")
        return
    
    model_path = train_model(dataset_path)
    
    if not model_path:
        return
    
    # Step 4: Export to ONNX
    onnx_path = export_to_onnx(model_path)
    
    if not onnx_path:
        return
    
    # Step 5: Convert to blob
    blob_path = convert_to_blob(onnx_path)
    
    if blob_path:
        print("\n" + "="*60)
        print("✓ SUCCESS! Your model is ready!")
        print("="*60)
        print(f"\nModel files created:")
        print(f"  - {blob_path}")
        print(f"  - model/labels.txt")
        print(f"\nYou can now run:")
        print(f"  python detect_oakd_local.py")
        print(f"\nMake sure to set:")
        print(f"  USE_PRETRAINED_YOLO = False")


if __name__ == "__main__":
    main()
