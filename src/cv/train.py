from ultralytics import YOLO

# Configuration
DATASET_YAML = "data/data.yaml"


def train_model():
    model = YOLO('yolov8n.pt')
    print(f"Starting training with dataset: {DATASET_YAML}")

    results = model.train(
        data=DATASET_YAML,

        # Training Duration
        epochs=100,
        patience=50,  # Stop if no improvement for 50 epochs

        # Image Settings
        imgsz=320,
        batch=16,

        # Hardware
        device='mps',  # Apple Silicon GPU acceleration
        workers=4,

        # Mosaic
        mosaic=0.0,  # TURN OFF! (Default is 1.0)
        close_mosaic=0,

        # Color noise
        hsv_h=0.0,  # Disable Hue shift
        hsv_s=0.0,  # Disable Saturation shift
        hsv_v=0.0,  # Disable Brightness shift

        # Geometry
        degrees=10.0,
        translate=0.1,
        scale=0.0,  # Disable scaling (keep size consistent)

        # Saving
        project='runs',
        name='detect',
        exist_ok=True
    )

    print("Training Complete!")


if __name__ == '__main__':
    train_model()
