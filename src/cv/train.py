from ultralytics import YOLO

def main():
    print("🧠 Loading YOLOv8 Nano model...")
    model = YOLO('yolov8n.pt')

    print("🚀 Starting training on Apple Silicon (MPS)...")
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='mps',
        plots=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1
    )

    print("📦 Exporting to ONNX...")
    success = model.export(format='onnx', imgsz=320, opset=12)
    print(f"Export finished: {success}")

if __name__ == '__main__':
    main()
