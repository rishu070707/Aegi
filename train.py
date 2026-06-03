"""
Weapon Detection Training Script
=================================
Paper: "Real-Time Multi-Class Weapon Detection System Using YOLOv8"
Authors: Shambhavi Trivedi, Aekeesh Jaiswal, Vaishnavi Singh
Institution: SRM Institute of Science and Technology, Delhi-NCR

Training Protocol (as per paper Section V):
- 3 model variants: YOLOv8n, YOLOv8s, YOLOv8m (ablation study, Table II)
- Dataset: Custom multi-source (Google Open Images + Roboflow + Kaggle + CCTV captures)
- 4 classes: Handgun, Knife, Rifle, Shotgun
- Optimizer: SGD, momentum=0.937, weight_decay=5e-4
- LR: cosine annealing 0.01 -> 0.001
- Batch: 16, imgsz: 640, max epochs: 100
- Early stopping: patience=15
- NMS: IoU=0.45, conf=0.25 at evaluation

IMPORTANT: Run this AFTER downloading the custom dataset via the Colab notebook.
           The dataset should be placed in ./dataset/ with train/val/test splits.
"""

import os
import csv
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")


# ─────────────────────────────────────────────────────────────────
# PAPER-EXACT HYPERPARAMETERS (Section V-B)
# ─────────────────────────────────────────────────────────────────
PAPER_HYPERPARAMS = dict(
    # Optimizer (SGD as specified in paper)
    optimizer="SGD",
    lr0=0.01,                  # initial LR
    lrf=0.1,                   # final LR = lr0 * lrf → 0.001 (cosine annealing)
    momentum=0.937,
    weight_decay=5e-4,
    cos_lr=True,               # cosine annealing schedule

    # Training
    epochs=100,
    batch=16,
    imgsz=640,
    patience=15,               # early stopping patience (15 non-improving epochs)

    # Augmentations (Section IV-A)
    mosaic=0.9,
    mixup=0.15,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,                 # ±50% random scaling
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    erasing=0.3,               # random erasing probability (simulates occlusion)
    degrees=0.0,               # rotation (not mentioned in paper, keep 0)

    # Evaluation thresholds
    conf=0.25,
    iou=0.45,
)

# ─────────────────────────────────────────────────────────────────
# 3 MODEL VARIANTS FOR ABLATION STUDY (Table II in paper)
# ─────────────────────────────────────────────────────────────────
ABLATION_MODELS = [
    {
        "name": "YOLOv8n",
        "weights": "yolov8n.pt",
        "description": "Nano variant - fastest, baseline comparison"
    },
    {
        "name": "YOLOv8s",
        "weights": "yolov8s.pt",
        "description": "Small variant - SELECTED MODEL (best speed-accuracy tradeoff per paper)"
    },
    {
        "name": "YOLOv8m",
        "weights": "yolov8m.pt",
        "description": "Medium variant - highest accuracy but slower"
    },
]

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_YAML    = PROJECT_ROOT / "data.yaml"
RESULTS_DIR  = PROJECT_ROOT / "runs" / "ablation"
RESULTS_CSV  = PROJECT_ROOT / "ablation_results.csv"
LOG_FILE     = PROJECT_ROOT / "training_log.json"


def verify_dataset():
    """Check that custom dataset exists and has images before training."""
    required = [
        PROJECT_ROOT / "dataset" / "train" / "images",
        PROJECT_ROOT / "dataset" / "valid" / "images",
        PROJECT_ROOT / "dataset" / "test"  / "images",
    ]
    print("\n[Dataset Check]")
    all_ok = True
    for path in required:
        count = len(list(path.glob("*.*"))) if path.exists() else 0
        status = "✓" if count > 10 else "✗ MISSING/EMPTY"
        print(f"  {status}  {path.relative_to(PROJECT_ROOT)}  ({count} images)")
        if count <= 10:
            all_ok = False

    if not all_ok:
        print("\n  ⚠  Dataset is missing or too small.")
        print("  ➜  Run the Google Colab notebook first to download & prepare the dataset.")
        print("  ➜  Then copy the dataset/ folder here and re-run train.py\n")
        return False
    return True


def verify_weights(weights_path: str) -> bool:
    """Check that the base weights file exists."""
    path = PROJECT_ROOT / weights_path
    if not path.exists():
        print(f"  ⚠  Weights not found: {weights_path}")
        print(f"  ➜  Ultralytics will auto-download them on first use.")
    return True  # Ultralytics auto-downloads


def train_single_model(model_cfg: dict, fold_yaml: str = None, run_name: str = None) -> dict:
    """
    Train one YOLOv8 variant with paper-exact hyperparameters.

    Args:
        model_cfg:  Entry from ABLATION_MODELS list
        fold_yaml:  Path to fold-specific YAML (for cross-val), or None for full dataset
        run_name:   Name for the training run directory

    Returns:
        dict with training metrics
    """
    data_yaml  = fold_yaml if fold_yaml else str(DATA_YAML)
    model_name = model_cfg["name"]
    weights    = model_cfg["weights"]
    run_id     = run_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}  ({model_cfg['description']})")
    print(f"  Weights : {weights}")
    print(f"  Data    : {data_yaml}")
    print(f"  Run     : {run_id}")
    print(f"{'='*60}")

    verify_weights(weights)
    model = YOLO(weights)

    start_time = time.time()

    results = model.train(
        data=data_yaml,
        project=str(RESULTS_DIR),
        name=run_id,
        exist_ok=True,

        # ── Exact paper hyperparameters ──────────────────────────
        epochs=PAPER_HYPERPARAMS["epochs"],
        batch=PAPER_HYPERPARAMS["batch"],
        imgsz=PAPER_HYPERPARAMS["imgsz"],
        patience=PAPER_HYPERPARAMS["patience"],

        optimizer=PAPER_HYPERPARAMS["optimizer"],
        lr0=PAPER_HYPERPARAMS["lr0"],
        lrf=PAPER_HYPERPARAMS["lrf"],
        momentum=PAPER_HYPERPARAMS["momentum"],
        weight_decay=PAPER_HYPERPARAMS["weight_decay"],
        cos_lr=PAPER_HYPERPARAMS["cos_lr"],

        # ── Augmentations ────────────────────────────────────────
        mosaic=PAPER_HYPERPARAMS["mosaic"],
        mixup=PAPER_HYPERPARAMS["mixup"],
        fliplr=PAPER_HYPERPARAMS["fliplr"],
        translate=PAPER_HYPERPARAMS["translate"],
        scale=PAPER_HYPERPARAMS["scale"],
        hsv_h=PAPER_HYPERPARAMS["hsv_h"],
        hsv_s=PAPER_HYPERPARAMS["hsv_s"],
        hsv_v=PAPER_HYPERPARAMS["hsv_v"],
        erasing=PAPER_HYPERPARAMS["erasing"],
        degrees=PAPER_HYPERPARAMS["degrees"],

        # ── Output / misc ────────────────────────────────────────
        verbose=True,
        save=True,
        save_period=10,     # save checkpoint every 10 epochs
        plots=True,
        val=True,
    )

    elapsed = time.time() - start_time

    # ── Extract metrics ──────────────────────────────────────────
    metrics_out = {
        "model":       model_name,
        "run_id":      run_id,
        "map50":       float(results.box.map50)   if hasattr(results, "box") else None,
        "map50_95":    float(results.box.map)     if hasattr(results, "box") else None,
        "precision":   float(results.box.mp)      if hasattr(results, "box") else None,
        "recall":      float(results.box.mr)      if hasattr(results, "box") else None,
        "elapsed_sec": round(elapsed, 1),
        "best_weights": str(RESULTS_DIR / run_id / "weights" / "best.pt"),
    }

    print(f"\n  ✓  {model_name} done in {elapsed/60:.1f} min")
    print(f"     mAP@50     : {metrics_out['map50']}")
    print(f"     mAP@50:95  : {metrics_out['map50_95']}")
    print(f"     Precision  : {metrics_out['precision']}")
    print(f"     Recall     : {metrics_out['recall']}")

    return metrics_out


def run_ablation_study():
    """
    Train YOLOv8n, YOLOv8s, YOLOv8m on the full dataset.
    Reproduces Table II (ablation study) from the paper.
    """
    print("\n" + "="*60)
    print("  ABLATION STUDY: YOLOv8n vs YOLOv8s vs YOLOv8m")
    print("  (Reproducing Table II from paper)")
    print("="*60)

    if not verify_dataset():
        return

    all_metrics = []
    for model_cfg in ABLATION_MODELS:
        run_name = f"ablation_{model_cfg['name'].lower()}"
        metrics  = train_single_model(model_cfg, run_name=run_name)
        all_metrics.append(metrics)

    # ── Save results to CSV ──────────────────────────────────────
    if all_metrics:
        keys = ["model", "map50", "map50_95", "precision", "recall", "elapsed_sec", "best_weights"]
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow({k: m.get(k) for k in keys})
        print(f"\n  ✓  Ablation results saved → {RESULTS_CSV}")

    # ── Print comparison table ────────────────────────────────────
    print("\n" + "="*60)
    print(f"  {'Model':<12} {'mAP@50':>8} {'mAP@50:95':>10} {'Precision':>10} {'Recall':>8}")
    print("  " + "-"*56)
    for m in all_metrics:
        print(f"  {m['model']:<12} "
              f"{m['map50'] or 'N/A':>8.4f} "
              f"{m['map50_95'] or 'N/A':>10.4f} "
              f"{m['precision'] or 'N/A':>10.4f} "
              f"{m['recall'] or 'N/A':>8.4f}")
    print("="*60)

    # ── Save training log ────────────────────────────────────────
    log = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "ablation_study",
        "hyperparams": PAPER_HYPERPARAMS,
        "results": all_metrics
    }
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  ✓  Training log saved → {LOG_FILE}")

    # Identify best model
    valid = [m for m in all_metrics if m["map50"] is not None]
    if valid:
        best = max(valid, key=lambda x: x["map50"])
        print(f"\n  ★  Best model: {best['model']}  (mAP@50 = {best['map50']:.4f})")
        print(f"     Per paper, YOLOv8s should achieve mAP@50 ≈ 0.961")

    return all_metrics


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Weapon Detection — Ablation Study Training              ║")
    print("║  YOLOv8n / YOLOv8s / YOLOv8m  |  Paper: Table II        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    print("  ⚠  REQUIREMENTS:")
    print("  1. Custom dataset must be downloaded via the Colab notebook")
    print("  2. dataset/ folder must have train/valid/test splits")
    print("  3. GPU strongly recommended (paper used NVIDIA Tesla T4)")
    print()
    run_ablation_study()
