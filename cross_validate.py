"""
5-Fold Cross-Validation Script
================================
Paper: "Real-Time Multi-Class Weapon Detection System Using YOLOv8"
Authors: Shambhavi Trivedi, Aekeesh Jaiswal, Vaishnavi Singh
Institution: SRM Institute of Science and Technology, Delhi-NCR

Cross-Validation Protocol (Section III-C & V-B of paper):
- Model: YOLOv8s (selected from ablation study, Table II)
- 5-fold cross-validation with 80/20 train/val split per fold
- Max 100 epochs per fold, early stopping patience=15
- All metrics averaged across folds (mean ± std)
- Reports: mAP@50, mAP@50:95, per-class P/R/F1

Run AFTER:
  1. Colab notebook has prepared dataset + fold YAMLs
  2. train.py ablation study has confirmed YOLOv8s is best model
"""

import os
import csv
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
import statistics

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")


# ─────────────────────────────────────────────────────────────────
# PAPER-EXACT HYPERPARAMETERS (must match train.py exactly)
# ─────────────────────────────────────────────────────────────────
PAPER_HYPERPARAMS = dict(
    optimizer="SGD",
    lr0=0.01,
    lrf=0.1,                   # cosine decay to lr0*lrf = 0.001
    momentum=0.937,
    weight_decay=5e-4,
    cos_lr=True,
    epochs=100,
    batch=16,
    imgsz=640,
    patience=15,               # early stopping: 15 consecutive non-improving epochs
    mosaic=0.9,
    mixup=0.15,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    erasing=0.3,
    degrees=0.0,
    conf=0.25,                 # NMS confidence threshold
    iou=0.45,                  # NMS IoU threshold
)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
NUM_FOLDS           = 5
MODEL_WEIGHTS       = "yolov8s.pt"        # selected model from ablation study
PROJECT_ROOT        = Path(__file__).parent
FOLD_DIR            = PROJECT_ROOT / "dataset"
RUNS_DIR            = PROJECT_ROOT / "runs" / "crossval"
RESULTS_CSV         = PROJECT_ROOT / "cross_validation_results.csv"
CLASS_METRICS_CSV   = PROJECT_ROOT / "class_metrics.csv"
LOG_FILE            = PROJECT_ROOT / "crossval_log.json"

CLASSES = ["Handgun", "Knife", "Rifle", "Shotgun"]


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def check_fold_yamls() -> list:
    """
    Verify that fold YAML files exist.
    These are created by the Colab notebook in:
      dataset/fold{i}/data_fold{i}.yaml
    Returns list of valid fold YAML paths.
    """
    valid_folds = []
    print("\n[Fold YAML Check]")
    for i in range(1, NUM_FOLDS + 1):
        yaml_path = FOLD_DIR / f"fold{i}" / f"data_fold{i}.yaml"
        if yaml_path.exists():
            valid_folds.append((i, str(yaml_path)))
            print(f"  ✓  Fold {i}: {yaml_path}")
        else:
            print(f"  ✗  Fold {i}: NOT FOUND — {yaml_path}")
            print(f"       ➜  Run the Colab notebook to generate fold splits")

    if not valid_folds:
        print("\n  ⚠  No fold YAMLs found.")
        print("  ➜  Run the Google Colab notebook to prepare 5-fold dataset splits first.")
    return valid_folds


def extract_metrics(results) -> dict:
    """
    Safely extract metrics from YOLO training results.
    Returns a dict with mAP@50, mAP@50:95, precision, recall, F1.
    """
    metrics = {
        "map50":     None,
        "map50_95":  None,
        "precision": None,
        "recall":    None,
        "f1":        None,
    }
    if not hasattr(results, "box"):
        return metrics
    try:
        metrics["map50"]     = float(results.box.map50)
        metrics["map50_95"]  = float(results.box.map)
        metrics["precision"] = float(results.box.mp)
        metrics["recall"]    = float(results.box.mr)
        # F1 = 2 * P * R / (P + R)
        p = metrics["precision"]
        r = metrics["recall"]
        if p and r and (p + r) > 0:
            metrics["f1"] = 2 * p * r / (p + r)
    except Exception as e:
        print(f"  ⚠  Could not extract some metrics: {e}")
    return metrics


def extract_per_class_metrics(results) -> dict:
    """
    Extract per-class AP@50 values from YOLO results.
    Maps class index → class name.
    """
    per_class = {cls: None for cls in CLASSES}
    try:
        if hasattr(results, "box") and hasattr(results.box, "ap_class_index"):
            indices = results.box.ap_class_index
            ap50s   = results.box.ap50          # per-class AP@50
            for idx, ap in zip(indices, ap50s):
                if idx < len(CLASSES):
                    per_class[CLASSES[idx]] = float(ap)
    except Exception as e:
        print(f"  ⚠  Per-class metrics not available: {e}")
    return per_class


def mean_std(values: list) -> tuple:
    """Return (mean, std) for a list, ignoring None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None
    m = statistics.mean(valid)
    s = statistics.stdev(valid) if len(valid) > 1 else 0.0
    return m, s


# ─────────────────────────────────────────────────────────────────
# MAIN CROSS-VALIDATION LOOP
# ─────────────────────────────────────────────────────────────────

def run_cross_validation():
    """
    Run 5-fold cross-validation with YOLOv8s.
    Reproduces the validation methodology described in Section III-C.
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Weapon Detection — 5-Fold Cross-Validation              ║")
    print("║  Model: YOLOv8s  |  Folds: 5  |  Max Epochs: 100        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    valid_folds = check_fold_yamls()
    if not valid_folds:
        return

    all_metrics     = []    # list of per-fold metric dicts
    all_per_class   = []    # list of per-fold per-class dicts
    fold_log        = []    # detailed log for JSON

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    for fold_num, fold_yaml in valid_folds:
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_num} / {NUM_FOLDS}")
        print(f"  Data  : {fold_yaml}")
        print(f"  Model : {MODEL_WEIGHTS}")
        print(f"{'─'*60}")

        run_name   = f"fold_{fold_num}"
        start_time = time.time()

        # Fresh model instance for each fold (no weight leakage between folds)
        model = YOLO(MODEL_WEIGHTS)

        results = model.train(
            data=fold_yaml,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,

            # ── Paper-exact hyperparameters ─────────────────────
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

            # ── Augmentations ───────────────────────────────────
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

            verbose=True,
            save=True,
            save_period=0,    # only save best + last during CV
            plots=True,
            val=True,
        )

        elapsed = time.time() - start_time
        metrics = extract_metrics(results)
        per_cls = extract_per_class_metrics(results)

        metrics["fold"]        = fold_num
        metrics["elapsed_sec"] = round(elapsed, 1)
        metrics["best_weights"]= str(RUNS_DIR / run_name / "weights" / "best.pt")

        all_metrics.append(metrics)
        all_per_class.append(per_cls)

        # Per-fold summary
        print(f"\n  ✓  Fold {fold_num} complete ({elapsed/60:.1f} min)")
        print(f"     mAP@50     : {metrics['map50']:.4f}" if metrics['map50'] else "     mAP@50     : N/A")
        print(f"     mAP@50:95  : {metrics['map50_95']:.4f}" if metrics['map50_95'] else "     mAP@50:95  : N/A")
        print(f"     Precision  : {metrics['precision']:.4f}" if metrics['precision'] else "     Precision  : N/A")
        print(f"     Recall     : {metrics['recall']:.4f}" if metrics['recall'] else "     Recall     : N/A")
        for cls in CLASSES:
            ap = per_cls.get(cls)
            print(f"     AP@50 {cls:<10}: {ap:.4f}" if ap else f"     AP@50 {cls:<10}: N/A")

        fold_log.append({
            "fold":      fold_num,
            "metrics":   metrics,
            "per_class": per_cls,
        })

    # ─────────────────────────────────────────────────────────────
    # AGGREGATE RESULTS ACROSS ALL FOLDS
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  CROSS-VALIDATION SUMMARY (Mean ± Std across 5 folds)")
    print(f"{'═'*60}")

    metric_keys = ["map50", "map50_95", "precision", "recall", "f1"]
    summary = {}
    for key in metric_keys:
        vals = [m.get(key) for m in all_metrics]
        m, s = mean_std(vals)
        summary[key] = {"mean": m, "std": s, "values": vals}
        label = key.upper().replace("_", "@")
        if m is not None:
            print(f"  {label:<15}: {m:.4f} ± {s:.4f}")
        else:
            print(f"  {label:<15}: N/A")

    # Per-class summary
    print(f"\n  Per-Class AP@50 (averaged across folds):")
    print(f"  {'Class':<12} {'Mean AP@50':>10} {'Std':>8}")
    print("  " + "─"*32)
    class_summary = {}
    for cls in CLASSES:
        vals = [pc.get(cls) for pc in all_per_class]
        m, s = mean_std(vals)
        class_summary[cls] = {"mean": m, "std": s}
        if m is not None:
            print(f"  {cls:<12} {m:>10.4f} {s:>8.4f}")
        else:
            print(f"  {cls:<12} {'N/A':>10}")

    # Expected per paper (Table III): Handgun=0.959, Knife=0.948, Rifle=0.974, Shotgun=0.963
    print("\n  Expected per paper (Table III):")
    expected = {"Handgun": 0.959, "Knife": 0.948, "Rifle": 0.974, "Shotgun": 0.963}
    for cls, exp in expected.items():
        print(f"  {cls:<12} expected AP@50 ≈ {exp}")

    # ─────────────────────────────────────────────────────────────
    # SAVE RESULTS TO CSV
    # ─────────────────────────────────────────────────────────────

    # Per-fold results
    fieldnames = ["fold", "map50", "map50_95", "precision", "recall", "f1",
                  "elapsed_sec"] + [f"ap50_{cls.lower()}" for cls in CLASSES] + ["best_weights"]

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m, pc in zip(all_metrics, all_per_class):
            row = {k: m.get(k) for k in fieldnames if k in m}
            for cls in CLASSES:
                row[f"ap50_{cls.lower()}"] = pc.get(cls)
            writer.writerow(row)

        # Append mean and std rows
        mean_row = {"fold": "MEAN"}
        std_row  = {"fold": "STD"}
        for key in metric_keys:
            mean_row[key] = summary[key]["mean"]
            std_row[key]  = summary[key]["std"]
        for cls in CLASSES:
            mean_row[f"ap50_{cls.lower()}"] = class_summary[cls]["mean"]
            std_row[f"ap50_{cls.lower()}"]  = class_summary[cls]["std"]
        writer.writerow(mean_row)
        writer.writerow(std_row)

    print(f"\n  ✓  Per-fold results saved → {RESULTS_CSV}")

    # Per-class metrics
    with open(CLASS_METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "mean_ap50", "std_ap50"])
        writer.writeheader()
        for cls in CLASSES:
            writer.writerow({
                "class":     cls,
                "mean_ap50": class_summary[cls]["mean"],
                "std_ap50":  class_summary[cls]["std"],
            })
    print(f"  ✓  Per-class metrics saved → {CLASS_METRICS_CSV}")

    # Full JSON log
    log = {
        "timestamp":     datetime.now().isoformat(),
        "experiment":    "5_fold_cross_validation",
        "model":         MODEL_WEIGHTS,
        "num_folds":     NUM_FOLDS,
        "hyperparams":   PAPER_HYPERPARAMS,
        "fold_results":  fold_log,
        "summary":       {
            k: {"mean": summary[k]["mean"], "std": summary[k]["std"]}
            for k in metric_keys
        },
        "per_class_summary": class_summary,
    }
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  ✓  Detailed log saved → {LOG_FILE}")

    # ─────────────────────────────────────────────────────────────
    # IDENTIFY BEST FOLD MODEL (highest mAP@50)
    # ─────────────────────────────────────────────────────────────
    valid = [m for m in all_metrics if m["map50"] is not None]
    if valid:
        best = max(valid, key=lambda x: x["map50"])
        print(f"\n  ★  Best fold: Fold {best['fold']}  (mAP@50 = {best['map50']:.4f})")
        print(f"     Best weights: {best['best_weights']}")
        print(f"\n  Paper target: mAP@50 ≥ 0.95  |  Claimed: 0.961")
        if summary["map50"]["mean"] and summary["map50"]["mean"] >= 0.95:
            print(f"  ✓  TARGET MET — Mean mAP@50 = {summary['map50']['mean']:.4f}")
        elif summary["map50"]["mean"]:
            print(f"  ✗  Below target — Mean mAP@50 = {summary['map50']['mean']:.4f}")

    print(f"\n{'═'*60}")
    print("  Cross-validation complete.")
    print(f"{'═'*60}\n")

    return all_metrics, summary


if __name__ == "__main__":
    print("\n  ⚠  PRE-REQUISITES:")
    print("  1. Custom dataset downloaded via Colab notebook")
    print("  2. Fold YAMLs at: dataset/fold{1-5}/data_fold{1-5}.yaml")
    print("  3. YOLOv8s ablation study completed (train.py)")
    print()
    run_cross_validation()
