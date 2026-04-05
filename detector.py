# -*- coding: utf-8 -*-
"""
detector.py  --  Weapon detector (Hugging Face weights only)

Loads Subh775/Threat-Detection-YOLOv8n weights as weapon_model.pt (see download_model.py).
If the file is missing, attempts a one-time download via huggingface_hub.

No secondary COCO model — all weapon inference uses this checkpoint only.
"""

import cv2
import os
import re
import shutil
import time
import numpy as np
from ultralytics import YOLO

HF_WEAPON_REPO = "Subh775/Threat-Detection-YOLOv8n"
HF_WEAPON_FILE = "weights/best.pt"


def _normalize_weapon_label(label: str) -> str:
    clean = label.replace("_", " ").replace("-", " ")
    clean = re.sub(r"\s+", " ", clean.strip())
    return clean.title()


def _name_to_weapon(name: str):
    """Map raw YOLO class name to a weapon label, or None if not a weapon class."""
    n = name.lower().strip()
    if n in {"hand", "human", "person", "people"}:
        return None

    if "weapon" in n or "firearm" in n or "gun" in n or "pistol" in n or "revolver" in n:
        return _normalize_weapon_label(n)

    if any(k in n for k in ("rifle", "sniper", "assault", "carbine", "longgun", "shotgun", "ak47", "ar15", "m4", "m16")):
        return _normalize_weapon_label(n)

    if any(k in n for k in ("knife", "blade", "dagger", "machete", "scissors", "sword", "cutlass", "bayonet")):
        return "Knife"

    if any(k in n for k in ("grenade", "explosive", "explosion", "bomb", "c4", "dynamite", "ied", "rocket", "missile")):
        return "Explosive"

    return None


RISK_COLORS = {
    "High": (0, 0, 255),
    "Medium": (0, 165, 255),
    "Low": (0, 220, 90),
}

# HF Threat-Detection-YOLOv8n often scores class "Gun" much lower than "grenade"/"knife"
# on the same frame; a dedicated low-threshold pass on that class recovers the firearm.
GUN_CONF_FLOOR = 0.01


def _resolve_gun_class_id(names: dict) -> int:
    for k, v in names.items():
        if str(v).lower().strip() == "gun":
            return int(k)
    return 0


def _iou_xyxy(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _drop_orphan_grenade_if_gun_present(dets: list, iou_need: float = 0.25) -> list:
    """
    When the model fires true class ``Gun``, separate high-conf ``grenade`` boxes are
    often the magazine/ammo tray. Drop those grenade→Handgun dets unless they overlap
    the pistol enough to be the same object.
    """
    guns = [d for d in dets if d.get("coco_name") == "Gun"]
    if not guns:
        return dets
    out = []
    for d in dets:
        if str(d.get("coco_name")).lower() != "grenade":
            out.append(d)
            continue
        best = max((_iou_xyxy(d["bbox"], g["bbox"]) for g in guns), default=0.0)
        if best >= iou_need:
            out.append(d)
    return out


def _merge_prefer_true_gun(primary: list, gun_only: list, iou_merge: float = 0.32) -> list:
    """Keep mislabeled boxes unless a true 'Gun' box overlaps — then prefer Gun."""
    out = list(primary)
    for g in gun_only:
        best_i = None
        best_iou = 0.0
        for i, d in enumerate(out):
            v = _iou_xyxy(d["bbox"], g["bbox"])
            if v >= iou_merge and v > best_iou:
                best_iou = v
                best_i = i
        if best_i is not None:
            d = out[best_i]
            if d.get("coco_name") != "Gun" and g.get("coco_name") == "Gun":
                out[best_i] = g
            elif d.get("coco_name") == "Gun" and g.get("coco_name") == "Gun":
                if g["confidence"] > d["confidence"]:
                    out[best_i] = g
        else:
            out.append(g)
    return out


def _weapon_model_path() -> str:
    return os.path.join(os.path.dirname(__file__), "weapon_model.pt")


def ensure_weapon_weights() -> str:
    """
    Return absolute path to weapon_model.pt, downloading from Hugging Face if needed.
    """
    dest = _weapon_model_path()
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        from huggingface_hub import hf_hub_download

        print("[WeaponDetector] weapon_model.pt missing — downloading from Hugging Face...")
        path = hf_hub_download(repo_id=HF_WEAPON_REPO, filename=HF_WEAPON_FILE)
        shutil.copy(path, dest)
        print(f"[WeaponDetector] Saved {dest}")
    except Exception as e:
        raise RuntimeError(
            "weapon_model.pt not found and Hugging Face download failed: "
            f"{e}\nInstall: pip install huggingface_hub\n"
            "Or run: python download_model.py"
        ) from e
    return dest


class WeaponDetector:
    """
    Single-model weapon detector using Hugging Face Threat-Detection-YOLOv8n weights only.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.38,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        if model_path and os.path.isfile(model_path):
            self.model_path = os.path.abspath(model_path)
        else:
            self.model_path = os.path.abspath(ensure_weapon_weights())

        self.model = YOLO(self.model_path)
        self.class_names = self.model.names
        self.is_custom = True
        self._gun_class_id = _resolve_gun_class_id(self.class_names)

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        print(f"[WeaponDetector] Loaded (HF weapon weights only): {self.model_path}")
        print(f"[WeaponDetector] Class names: {dict(self.class_names)} | gun_cls_id={self._gun_class_id}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        if np.mean(l) / 255.0 < (50 / 255):
            lab[:, :, 0] = self._clahe.apply(l)
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            table = np.array(
                [((i / 255.0) ** (1.0 / 0.7)) * 255 for i in range(256)],
                dtype=np.uint8,
            )
            frame = cv2.LUT(frame, table)
        return frame

    @staticmethod
    def _valid_weapon_box(bbox, frame_shape, cls_name: str) -> bool:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return False
        h, w = frame_shape[:2]
        area_pct = (bw * bh) / (w * h) * 100.0

        # Allow small distant firearms (high-res frames); still reject speck noise
        if area_pct < 0.02 or area_pct > 90.0:
            return False

        aspect = bw / bh

        if cls_name == "Knife":
            if area_pct > 12.0 and 0.75 < aspect < 1.35:
                return False
            if aspect > 10.0 or aspect < 0.1:
                return False
            if area_pct > 90.0:
                return False

        handgun_classes = {"Handgun", "Gun", "Pistol", "Revolver", "Firearm", "Glock"}
        if cls_name in handgun_classes:
            if aspect > 3.0 or aspect < 0.25:
                return False
            if area_pct > 90.0:
                return False

        return True

    def _run_model(self, frame, conf: float, imgsz: int, class_ids: list[int] | None = None):
        kw = dict(
            imgsz=imgsz,
            conf=conf,
            iou=self.iou_threshold,
            verbose=False,
        )
        if class_ids is not None:
            kw["classes"] = class_ids
        return self.model(frame, **kw)[0]

    def _boxes_to_detections(self, r, frame_shape: tuple) -> list:
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            cls_id = int(cls_id)
            raw = self.class_names.get(cls_id, "unknown")
            raw = raw if isinstance(raw, str) else str(raw)
            weapon_cls = _name_to_weapon(raw)
            if weapon_cls is None:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            bbox = [x1, y1, x2, y2]
            if not self._valid_weapon_box(bbox, frame_shape, weapon_cls):
                continue
            out.append(
                {
                    "class_name": weapon_cls,
                    "confidence": round(float(conf), 4),
                    "bbox": bbox,
                    "coco_name": raw,
                    "source": "weapon_model",
                }
            )
        return out

    def detect(self, frame: np.ndarray, imgsz: int | None = None) -> tuple[list, float]:
        """
        Run inference. If ``imgsz`` is set, it overrides ``self.input_size`` for this call
        (use 640 for uploads so live-feed edge mode cannot leave the singleton at 416).
        """
        t0 = time.perf_counter()
        proc = self._preprocess(frame)
        sz = int(imgsz) if imgsz is not None else int(self.input_size)

        r = self._run_model(proc, conf=self.conf_threshold, imgsz=sz)
        detections = self._boxes_to_detections(r, proc.shape)

        # Recover real pistols the main threshold misses (class Gun only, very low floor)
        r_gun = self._run_model(
            proc,
            conf=GUN_CONF_FLOOR,
            imgsz=sz,
            class_ids=[self._gun_class_id],
        )
        gun_dets = self._boxes_to_detections(r_gun, proc.shape)
        detections = _merge_prefer_true_gun(detections, gun_dets)
        detections = _drop_orphan_grenade_if_gun_present(detections)

        latency = (time.perf_counter() - t0) * 1000.0
        return detections, latency

    def draw_detections(self, frame: np.ndarray, dets: list) -> np.ndarray:
        out = frame.copy()
        for det in dets:
            cls = det.get("class_name", "Unknown")
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", [0, 0, 100, 100])
            risk_level = det.get("risk_level", "Low")
            risk_score = det.get("risk_score", 0.0)

            x1, y1, x2, y2 = map(int, bbox)
            color = RISK_COLORS.get(risk_level, (0, 220, 90))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {conf:.2f} | {risk_level} R:{risk_score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52
            thick = 1
            (tw, th), base = cv2.getTextSize(label, font, scale, thick)
            label_h = th + base + 6

            if y1 - label_h >= 0:
                lt = y1 - label_h
                ty = y1 - base - 3
            else:
                lt = y1
                ty = y1 + th + 3

            cv2.rectangle(out, (x1, lt), (x1 + tw + 8, lt + label_h), color, cv2.FILLED)
            cv2.putText(
                out, label, (x1 + 4, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA
            )

        return out

    def switch_model(self, model_path: str | None, input_size: int):
        """Edge mode: only input resolution changes; optional explicit weight path."""
        if model_path is not None and os.path.isfile(str(model_path)):
            self.model = YOLO(model_path)
            self.model_path = os.path.abspath(model_path)
            self.class_names = self.model.names
            self._gun_class_id = _resolve_gun_class_id(self.class_names)
        self.input_size = input_size
        print(f"[WeaponDetector] Resolution -> {input_size}")
