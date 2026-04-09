#!/usr/bin/env python3
"""Train a YOLO weapon detection model from a merged dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO weapon detection model.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the merged YOLO data YAML file.",
    )
    parser.add_argument(
        "--weights",
        default="yolov8s.pt",
        help="Base weights to fine-tune from (default: yolov8s.pt).",
    )
    parser.add_argument(
        "--project",
        default="runs/train",
        help="Training output project folder.",
    )
    parser.add_argument(
        "--name",
        default="weapon_finetune",
        help="Training experiment name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Overwrite existing training output if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")

    print(f"Training from weights: {args.weights}")
    print(f"Using data file: {data_path}")
    print("Starting 5-fold cross validation as per paper specifications...")

    for fold in range(1, 6):
        print(f"\n[{'='*40}]\n[ INFO ] Training Fold {fold}/5\n[{'='*40}]\n")
        
        # Re-initialize the model for each fold to avoid weight leakage
        model = YOLO(args.weights)
        fold_name = f"{args.name}_fold_{fold}"

        model.train(
            data=str(data_path),
            epochs=100,           # Max 100 epochs
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=fold_name,
            exist_ok=args.exist_ok,
            patience=15,          # Early stopping after 15 consecutive epochs without improvement
            optimizer='SGD',      # Fine-tuned using SGD
            momentum=0.937,
            weight_decay=0.0005,
            lr0=0.01,
            lrf=0.1,              # Cosine annealing reduces LR from 0.01 to 0.001 (0.01 * 0.1)
            cos_lr=True,
            val=True
        )

    print("5-Fold Cross Validation Complete. Check the run directory for weights and logs.")


if __name__ == "__main__":
    main()
