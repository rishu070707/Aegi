#!/usr/bin/env python3
"""Prepare a merged weapon detection dataset from multiple YOLO-format sources.

This script is intended to help you combine multiple downloaded weapon datasets
into a single unified training set for YOLO model fine-tuning.

Usage:
  python scripts/prepare_weapon_dataset.py \
    --sources ./data/openimages ./data/ari-dasci ./data/roboflow-top \
    --output ./data/weapon_combined \
    --target-classes Handgun,Knife,Rifle,Shotgun \
    --split 0.85
"""

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise RuntimeError(
        "PyYAML is required for this script. Install it with: pip install PyYAML"
    ) from exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_ALIAS_MAP = {
    "Handgun": [
        "handgun",
        "pistol",
        "glock",
        "revolver",
        "firearm",
        "gun",
        "hand gun",
        "semi-automatic",
        "semi automatic",
        "automatic",
    ],
    "Shotgun": [
        "shotgun",
    ],
    "Rifle": [
        "rifle",
        "sniper",
        "carbine",
        "assault",
        "longgun",
        "long gun",
        "ar15",
        "m4",
        "m16",
    ],
    "Knife": [
        "knife",
        "blade",
        "dagger",
        "machete",
        "scissors",
        "cutlass",
    ],

}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO-format weapon datasets into one unified dataset."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="One or more dataset root folders already downloaded from the source sites.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for the merged dataset.",
    )
    parser.add_argument(
        "--target-classes",
        default="Handgun,Knife,Rifle,Shotgun",
        help="Comma-separated final class list to keep in the merged dataset.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.85,
        help="Train/validation split ratio (default: 0.85).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_name_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines


def resolve_source_class_names(source_root: Path) -> list[str]:
    yaml_candidates = [source_root / "data.yaml", source_root / "dataset.yaml"]
    for yaml_path in yaml_candidates:
        if yaml_path.exists():
            data = load_yaml(yaml_path)
            names = data.get("names") or data.get("classes")
            if isinstance(names, dict):
                return [str(v) for k, v in sorted(names.items(), key=lambda i: int(i[0]))]
            if isinstance(names, list):
                return [str(v) for v in names]
    for name_file in ["classes.txt", "obj.names", "names.txt"]:
        path = source_root / name_file
        if path.exists():
            return read_name_list(path)
    return []


def normalize_class_name(raw: str) -> str | None:
    raw_normal = raw.lower().strip()
    for standard, aliases in DEFAULT_ALIAS_MAP.items():
        if any(alias in raw_normal for alias in aliases):
            return standard
    return None


def resolve_label_name(source_name: str, target_classes: list[str]) -> str | None:
    normalized = normalize_class_name(source_name)
    if normalized is None:
        return None
    if normalized in target_classes:
        return normalized
    # allow exact match even if not in alias list
    if source_name in target_classes:
        return source_name
    return None


def find_image_files(root: Path) -> list[Path]:
    candidates = []
    for ext in IMAGE_EXTENSIONS:
        candidates.extend(root.rglob(f"*{ext}"))
    return [p for p in candidates if p.is_file()]


def gather_image_label_pairs(source_root: Path) -> tuple[list[Path], Path | None]:
    yaml_candidates = [source_root / "data.yaml", source_root / "dataset.yaml"]
    if any(candidate.exists() for candidate in yaml_candidates):
        # If YAML already describes splits, we will use images from the dataset
        return find_image_files(source_root), None

    label_root = None
    for candidate in [source_root / "labels", source_root / "labels/train", source_root / "labels/val"]:
        if candidate.exists():
            label_root = source_root / "labels"
            break
    if label_root is None:
        return find_image_files(source_root), None
    return find_image_files(source_root), label_root


def build_class_mapping(source_names: list[str], target_classes: list[str]) -> dict[int, int]:
    mapping = {}
    for idx, raw in enumerate(source_names):
        label_name = resolve_label_name(raw, target_classes)
        if label_name is not None and label_name in target_classes:
            mapping[idx] = target_classes.index(label_name)
    return mapping


def load_yolo_label_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_label_line(line: str, class_map: dict[int, int]) -> str | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        source_idx = int(parts[0])
    except ValueError:
        return None
    target_idx = class_map.get(source_idx)
    if target_idx is None:
        return None
    return " ".join([str(target_idx)] + parts[1:5])


def copy_dataset_images_and_labels(
    image_paths: list[Path],
    source_root: Path,
    label_root: Path | None,
    class_map: dict[int, int],
    source_name: str,
    output_root: Path,
    split: str,
) -> tuple[int, int]:
    image_dest = output_root / "images" / split
    label_dest = output_root / "labels" / split
    image_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for image_path in image_paths:
        rel = image_path.relative_to(source_root)
        dest_image_name = f"{source_name}__{rel.name}"
        dest_image_path = image_dest / dest_image_name
        shutil.copy2(image_path, dest_image_path)

        if label_root is None:
            skipped += 1
            continue

        label_path = label_root / image_path.with_suffix(".txt").name
        if not label_path.exists():
            label_path = label_root / rel.with_suffix(".txt")
        if not label_path.exists():
            skipped += 1
            continue

        lines = load_yolo_label_file(label_path)
        converted = [convert_label_line(line, class_map) for line in lines]
        converted = [line for line in converted if line is not None]
        if not converted:
            skipped += 1
            continue

        label_output = label_dest / dest_image_path.with_suffix(".txt").name
        label_output.write_text("\n".join(converted), encoding="utf-8")
        copied += 1
    return copied, skipped


def create_data_yaml(output_root: Path, target_classes: list[str]) -> None:
    data_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "names": target_classes,
    }
    yaml_path = output_root / "weapon_data.yaml"
    yaml.safe_dump(data_yaml, yaml_path)
    print(f"Wrote data YAML: {yaml_path}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output).resolve()
    target_classes = [c.strip() for c in args.target_classes.split(",") if c.strip()]
    if not target_classes:
        raise ValueError("At least one target class is required.")

    output_images_train = output_root / "images" / "train"
    output_images_val = output_root / "images" / "val"
    output_labels_train = output_root / "labels" / "train"
    output_labels_val = output_root / "labels" / "val"
    output_images_train.mkdir(parents=True, exist_ok=True)
    output_images_val.mkdir(parents=True, exist_ok=True)
    output_labels_train.mkdir(parents=True, exist_ok=True)
    output_labels_val.mkdir(parents=True, exist_ok=True)

    all_pairs: list[tuple[Path, Path | None, str, dict[int, int]]] = []
    summary = {
        "sources": {},
        "classes": target_classes,
        "images": 0,
        "labels": 0,
        "skipped_images": 0,
    }

    for source_path in args.sources:
        source_root = Path(source_path).resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Source dataset not found: {source_root}")

        source_label_names = resolve_source_class_names(source_root)
        if not source_label_names:
            raise RuntimeError(
                f"Could not resolve class names for source {source_root}. "
                "Please ensure the dataset contains data.yaml or classes.txt/obj.names."
            )

        class_map = build_class_mapping(source_label_names, target_classes)
        if not class_map:
            raise RuntimeError(
                f"Source {source_root} has no classes that map to the target class set: {target_classes}"
            )

        source_image_paths, label_root = gather_image_label_pairs(source_root)
        source_name = sanitize_name(source_root.name)
        random.Random(args.seed).shuffle(source_image_paths)

        if label_root is None:
            raise RuntimeError(
                f"Source {source_root} does not contain a recognizable labels folder. "
                "Please prepare the dataset in YOLO format first."
            )

        cutoff = int(len(source_image_paths) * args.split)
        train_paths = source_image_paths[:cutoff]
        val_paths = source_image_paths[cutoff:]

        train_copied, train_skipped = copy_dataset_images_and_labels(
            train_paths,
            source_root,
            label_root,
            class_map,
            source_name,
            output_root,
            "train",
        )
        val_copied, val_skipped = copy_dataset_images_and_labels(
            val_paths,
            source_root,
            label_root,
            class_map,
            source_name,
            output_root,
            "val",
        )

        summary["sources"][source_name] = {
            "images_found": len(source_image_paths),
            "train_copied": train_copied,
            "val_copied": val_copied,
            "train_skipped": train_skipped,
            "val_skipped": val_skipped,
        }
        summary["images"] += train_copied + val_copied
        summary["labels"] += train_copied + val_copied
        summary["skipped_images"] += train_skipped + val_skipped

    create_data_yaml(output_root, target_classes)
    stats_path = output_root / "merge_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Merged dataset written to {output_root}")
    print(f"Total image / label pairs: {summary['images']}")
    print(f"Skipped due to missing or unmapped labels: {summary['skipped_images']}")


if __name__ == "__main__":
    main()
