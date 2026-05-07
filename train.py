from ultralytics import YOLO

def main():
    print("Starting custom training pipeline...")
    # Load base model
    model = YOLO("yolov8s.pt")
    
    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="train",
        exist_ok=True
    )
    print("Training complete. Models saved to runs/detect/train/weights/")

if __name__ == "__main__":
    main()
