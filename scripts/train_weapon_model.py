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

    model = YOLO(args.weights)
    print(f"Training from weights: {args.weights}")
    print(f"Using data file: {data_path}")

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    print("Training complete. Check the run directory for weights and logs.")


if __name__ == "__main__":
    main()
