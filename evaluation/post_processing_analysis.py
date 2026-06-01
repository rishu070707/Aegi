"""
evaluation/post_processing_analysis.py — Post-Processing Ablation Study Analysis

Generates ablation study metrics showing the contribution of each post-processing
module to false positive reduction and overall detection performance.

Data source priority:
  1. evaluation/ablation_real_results.csv  — real experimental results (preferred)
  2. Hardcoded placeholder values          — fallback only (see WARNING below)

CSV format expected (headers):
  module,precision,recall,f1,fp_rate,fp_reduction_pct
"""

import os
import csv
import json
from pathlib import Path


# ── Path resolution ────────────────────────────────────────────────────────────
_EVAL_DIR = Path(__file__).parent
_REAL_CSV  = _EVAL_DIR / "ablation_real_results.csv"


def generate_ablation_study() -> dict:
    """
    Generate ablation study data for post-processing modules.

    Returns a dict with keys:
      - 'source'  : 'real' | 'placeholder'
      - 'modules' : list of dicts, one per pipeline configuration
      - 'summary' : high-level summary statistics
    """
    # ── 1. Try to load real experimental results ───────────────────────────────
    if _REAL_CSV.exists():
        try:
            modules = []
            with open(_REAL_CSV, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    modules.append({
                        "module":           row.get("module", "Unknown"),
                        "precision":        float(row.get("precision", 0)),
                        "recall":           float(row.get("recall", 0)),
                        "f1":               float(row.get("f1", 0)),
                        "fp_rate":          float(row.get("fp_rate", 0)),
                        "fp_reduction_pct": float(row.get("fp_reduction_pct", 0)),
                    })

            summary = _compute_summary(modules)
            return {
                "source":  "real",
                "csv_path": str(_REAL_CSV),
                "modules": modules,
                "summary": summary,
            }

        except Exception as exc:
            print(f"[AblationStudy] WARNING: Failed to read real CSV ({_REAL_CSV}): {exc}")
            print("[AblationStudy] Falling back to placeholder data.")

    # ── 2. Fallback: placeholder / simulated values ────────────────────────────
    # WARNING: Using placeholder data. Run real ablation experiment to generate actual results.
    print(
        "WARNING: Using placeholder data. Run real ablation experiment to generate actual results."
    )

    # Simulating the post-processing ablation study metrics
    # These values are PLACEHOLDERS — replace by running the full ablation pipeline
    # and saving results to evaluation/ablation_real_results.csv
    modules = [
        {
            "module":           "Baseline (No Post-Processing)",
            "precision":        0.72,
            "recall":           0.89,
            "f1":               0.80,
            "fp_rate":          0.28,
            "fp_reduction_pct": 0.0,
        },
        {
            "module":           "+ Temporal Consistency Filter (N=5, K=3, τ=0.30)",
            "precision":        0.81,
            "recall":           0.87,
            "f1":               0.84,
            "fp_rate":          0.19,
            "fp_reduction_pct": 32.1,
        },
        {
            "module":           "+ Confidence Stabilization (EMA α=0.4)",
            "precision":        0.84,
            "recall":           0.86,
            "f1":               0.85,
            "fp_rate":          0.16,
            "fp_reduction_pct": 42.9,
        },
        {
            "module":           "+ Context-Aware Risk Scoring (w1=0.5, w2=0.3, w3=0.2)",
            "precision":        0.87,
            "recall":           0.85,
            "f1":               0.86,
            "fp_rate":          0.13,
            "fp_reduction_pct": 53.6,
        },
        {
            "module":           "+ Smart ROI Monitoring",
            "precision":        0.90,
            "recall":           0.84,
            "f1":               0.87,
            "fp_rate":          0.10,
            "fp_reduction_pct": 64.3,
        },
        {
            "module":           "+ Alert Cooldown (Δt=5s)",
            "precision":        0.928,
            "recall":           0.83,
            "f1":               0.877,
            "fp_rate":          0.072,
            "fp_reduction_pct": 74.3,
        },
    ]

    summary = _compute_summary(modules)
    return {
        "source":  "placeholder",
        "csv_path": None,
        "modules": modules,
        "summary": summary,
    }


def _compute_summary(modules: list) -> dict:
    """Derive high-level summary stats from the modules list."""
    if not modules:
        return {}
    baseline = modules[0]
    final    = modules[-1]
    return {
        "baseline_precision":   baseline["precision"],
        "final_precision":      final["precision"],
        "baseline_fp_rate":     baseline["fp_rate"],
        "final_fp_rate":        final["fp_rate"],
        "total_fp_reduction_pct": final["fp_reduction_pct"],
        "module_count":         len(modules),
    }


def print_ablation_table(data: dict | None = None) -> None:
    """Pretty-print the ablation study results to stdout."""
    if data is None:
        data = generate_ablation_study()

    source_label = "REAL EXPERIMENTAL DATA" if data["source"] == "real" else "PLACEHOLDER DATA"
    print(f"\n{'=' * 70}")
    print(f"  POST-PROCESSING ABLATION STUDY  [{source_label}]")
    if data.get("csv_path"):
        print(f"  Source: {data['csv_path']}")
    print(f"{'=' * 70}")
    print(f"  {'Module':<52} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FP↓%':>7}")
    print(f"  {'-' * 52} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7}")
    for m in data["modules"]:
        print(
            f"  {m['module']:<52} "
            f"{m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} "
            f"{m['fp_reduction_pct']:>6.1f}%"
        )
    s = data.get("summary", {})
    print(f"\n  Total FP reduction: {s.get('total_fp_reduction_pct', 0):.1f}%")
    print(f"  Final precision:    {s.get('final_precision', 0):.3f}")
    print(f"{'=' * 70}\n")


def save_ablation_json(out_path: str | None = None) -> str:
    """Run ablation study and save results as JSON. Returns path to saved file."""
    data = generate_ablation_study()
    if out_path is None:
        out_path = str(_EVAL_DIR / "ablation_results_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[AblationStudy] Results saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    print_ablation_table()
