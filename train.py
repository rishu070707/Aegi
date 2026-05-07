import wandb
import mlflow
from ultralytics import YOLO

def main():
    print("Initializing Experiment Tracking...")
    
    # 1. Initialize Weights & Biases
    wandb.init(
        project="weapon_detection_research",
        name="fold_1_sgd",
        config={
            "optimizer": "SGD",
            "lr0": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "cos_lr": True,
            "epochs": 100,
            "batch": 16,
            "imgsz": 640
        }
    )
    
    # 2. Initialize MLflow
    mlflow.set_experiment("weapon_detection_pipeline")
    mlflow.start_run(run_name="baseline_yolov8s_augmented")
    mlflow.log_params(wandb.config)
    
    print("Starting custom training pipeline with W&B and MLflow...")
    # Load base model
    model = YOLO("yolov8s.pt")
    
    # Train the model with hyperparameters from the paper
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        mosaic=0.9,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        project="runs/detect",
        name="train_custom",
        exist_ok=True
    )
    
    # Log metrics to MLflow
    if hasattr(results, "box"):
        mlflow.log_metric("mAP50", results.box.map50)
        mlflow.log_metric("Precision", results.box.mp)
        mlflow.log_metric("Recall", results.box.mr)
        
    mlflow.end_run()
    wandb.finish()
    print("Training complete. Trackers synchronized.")

if __name__ == "__main__":
    main()
