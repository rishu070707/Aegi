import os
from pathlib import Path
from ultralytics import YOLO

def main():
    print("Starting 5-fold cross validation for Weapon Detection Dataset...")
    
    # Normally, you would use Ultralytics K-Fold validation or a custom loop:
    data_path = Path("data.yaml")
    if not data_path.exists():
        print("Error: data.yaml not found.")
        return
        
    folds = 5
    epochs = 50
    model_name = "yolov8s.pt"
    
    metrics = []
    
    for i in range(1, folds + 1):
        print(f"\n--- Training Fold {i}/{folds} ---")
        fold_yaml = f"dataset/fold{i}/data_fold{i}.yaml"
        if not os.path.exists(fold_yaml):
            print(f"Warning: {fold_yaml} not found. Ensure dataset splits exist for fold {i}.")
            continue
            
        model = YOLO(model_name)
        results = model.train(
            data=fold_yaml,
            epochs=epochs,
            imgsz=640,
            batch=16,
            project="runs/crossval",
            name=f"fold_{i}",
            exist_ok=True
        )
        
        # Collect metrics from results
        metrics.append({
            "Fold": i,
            "mAP50": results.box.map50,
            "Precision": results.box.mp,
            "Recall": results.box.mr
        })
        
    print("\n--- Cross Validation Summary ---")
    if not metrics:
        print("No folds were trained. Please prepare dataset fold YAML files.")
    else:
        for m in metrics:
            print(m)

if __name__ == "__main__":
    main()
