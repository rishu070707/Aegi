# -*- coding: utf-8 -*-
"""
detector.py  --  Dual-Model Weapon Detector

Primary  : weapon_model.pt (Subh775/Threat-Detection-YOLOv8n)
           Classes: Gun -> Handgun, Knife, Grenade -> Handgun
           Confidence threshold: 0.55 (high, to kill hand false-positives)

Secondary: yolov8l.pt  (COCO, user-requested large model)
           Only the single class: knife (COCO id=43)
           Confidence threshold: 0.30
           Runs ONLY when primary is loaded to catch knives the small model misses.

Results are merged and de-duplicated by IoU overlap.
"""

import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# ---- COCO classes relevant to weapons ----------------------------------------
COCO_WEAPON_MAP = {
    43: "Knife",    # knife
    76: "Knife",    # scissors
    38: "Rifle",    # baseball bat
}

# ---- Custom model name->display class (name-based, model-agnostic) -----------
def _name_to_weapon(name: str):
    """Return our display class for a raw class name, or None if irrelevant."""
    n = name.lower().strip()
    # Explicit mapping to avoid substring match issues
    if any(k in n for k in ("handgun","pistol","glock","revolver","gun","firearm","explosive")):
        return "Handgun"
    if n == "hand": # explicit skip for 'hand' class often found in custom models
        return None
    if any(k in n for k in ("knife","blade","dagger","machete","scissors")):
        return "Knife"
    if any(k in n for k in ("rifle","longgun","assault","carbine","sniper")):
        return "Rifle"
    if any(k in n for k in ("shotgun","heavyweapon")):
        return "Shotgun"
    return None

WEAPON_CLASSES = ["Handgun", "Knife", "Rifle", "Shotgun"]

RISK_COLORS = {
    "High":   (0, 0, 255),
    "Medium": (0, 165, 255),
    "Low":    (0, 220, 90),
}


def _find_custom_model():
    base = os.path.dirname(__file__)
    for name in ("weapon_model.pt",):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def _find_coco_large():
    base = os.path.dirname(__file__)
    for name in ("yolov8l.pt", "yolov8s.pt", "yolov8n.pt"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p, name
    return None, "yolov8l.pt"   # ultralytics will auto-download


def _iou(b1, b2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class WeaponDetector:
    """
    Dual-model weapon detector.

    - weapon_model.pt  (custom): high-threshold (0.55) to kill false handgun positives
    - yolov8l.pt       (COCO):   knife-only, lower threshold (0.30)

    Results are merged; overlapping boxes (IoU > 0.4) keep the higher-confidence one.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.65,      # Increased from 0.55 for precision
        iou_threshold:  float = 0.45,
        input_size:     int   = 640,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.input_size     = input_size

        # ---- Primary model (weapon_model.pt if available) --------------------
        custom = _find_custom_model()
        if model_path and os.path.exists(model_path):
            custom = model_path

        if custom:
            self.model_path = custom
            self.is_custom  = True
        else:
            coco_p, coco_name = _find_coco_large()
            self.model_path = coco_p or coco_name
            self.is_custom  = False

        self.model       = YOLO(self.model_path)
        self.class_names = self.model.names

        # ---- Secondary: yolov8l for COCO knife class -------------------------
        self._coco_model      = None
        self._coco_class_names = {}
        if self.is_custom:
            coco_p, coco_name = _find_coco_large()
            try:
                self._coco_model       = YOLO(coco_p or coco_name)
                self._coco_class_names = self._coco_model.names
                print(f"[WeaponDetector] Secondary COCO model: {coco_p or coco_name}")
            except Exception as e:
                print(f"[WeaponDetector] Warning: secondary model failed: {e}")

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        mode = "CUSTOM (weapon_model.pt)" if self.is_custom else "COCO pretrained (yolov8l)"
        print(f"[WeaponDetector] Primary: {self.model_path} | {mode}")

    # ---- PREPROCESSING --------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l   = lab[:, :, 0]
        if np.mean(l) / 255.0 < (50 / 255):
            lab[:, :, 0] = self._clahe.apply(l)
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            table = np.array(
                [((i / 255.0) ** (1.0 / 0.7)) * 255 for i in range(256)],
                dtype=np.uint8,
            )
            frame = cv2.LUT(frame, table)
        return frame

    # ---- DETECTION ------------------------------------------------------------

    @staticmethod
    def _valid_weapon_box(bbox, frame_shape, cls_name: str) -> bool:
        """
        Geometric sanity check to reject blatant false positives.
        Returns False if the bbox is implausibly shaped for the claimed weapon.
        """
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return False
        h, w = frame_shape[:2]
        area_pct = (bw * bh) / (w * h) * 100.0

        # Too tiny (noise) or too huge (whole frame = usually false positive)
        if area_pct < 0.3 or area_pct > 70.0:
            return False

        aspect = bw / bh  # >1 = wider, <1 = taller

        if cls_name == "Knife":
            # Knife should be elongated
            # Reject square blobs (>15% area and aspect near 1.0)
            if area_pct > 12.0 and 0.75 < aspect < 1.35:
                return False
            # Reject extremely flat/vertical slivers that are likely noise
            if aspect > 10.0 or aspect < 0.1:
                return False

        if cls_name == "Handgun":
            # Handguns are usually compact; reject extremely wide aspect ratios (like car bumpers)
            if aspect > 2.5 or aspect < 0.3:
                return False
            # Also reject very large handguns (usually background misclassified)
            if area_pct > 35.0:
                return False

        if cls_name in ("Shotgun", "Rifle"):
            # Rifles are usually thin and wide/tall
            if 0.6 < aspect < 1.6 and area_pct > 20.0:
                return False

        return True

    def _run_model(self, model, frame, conf, classes=None):
        kwargs = dict(imgsz=self.input_size, conf=conf,
                      iou=self.iou_threshold, verbose=False)
        if classes is not None:
            kwargs["classes"] = classes
        return model(frame, **kwargs)[0]

    def detect(self, frame: np.ndarray) -> tuple[list, float]:
        t0    = time.perf_counter()
        frame = self._preprocess(frame)

        detections = []

        # 1. Primary weapon model
        r = self._run_model(self.model, frame, conf=self.conf_threshold)
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            cls_id = int(cls_id)
            raw    = self.class_names.get(cls_id, "unknown")

            weapon_cls = _name_to_weapon(raw)
            if weapon_cls is None:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            bbox = [x1, y1, x2, y2]
            if not self._valid_weapon_box(bbox, frame.shape, weapon_cls):
                continue

            detections.append({
                "class_name": weapon_cls,
                "confidence": round(float(conf), 4),
                "bbox":       bbox,
                "coco_name":  raw,
                "source":     "weapon_model",
            })

        # 2. Secondary: yolov8l COCO knife-only (conf 0.40 -- higher than before)
        if self._coco_model is not None:
            r2 = self._run_model(self._coco_model, frame,
                                  conf=0.40, classes=[43])  # 43 = knife
            for box, cls_id, conf in zip(r2.boxes.xyxy, r2.boxes.cls, r2.boxes.conf):
                raw = self._coco_class_names.get(int(cls_id), "knife")
                x1, y1, x2, y2 = map(int, box.tolist())
                bbox = [x1, y1, x2, y2]
                # Geometric sanity check
                if not self._valid_weapon_box(bbox, frame.shape, "Knife"):
                    continue
                new_det = {
                    "class_name": "Knife",
                    "confidence": round(float(conf), 4),
                    "bbox":       bbox,
                    "coco_name":  raw,
                    "source":     "yolov8l",
                }
                # De-duplicate: skip if overlaps >0.4 with existing detection
                overlap = any(
                    _iou(new_det["bbox"], d["bbox"]) > 0.4
                    for d in detections
                )
                if not overlap:
                    detections.append(new_det)

        latency = (time.perf_counter() - t0) * 1000.0
        return detections, latency

    # ---- DRAWING --------------------------------------------------------------

    def draw_detections(self, frame: np.ndarray, dets: list) -> np.ndarray:
        out = frame.copy()
        for det in dets:
            cls        = det.get("class_name", "Unknown")
            conf       = det.get("confidence", 0.0)
            bbox       = det.get("bbox", [0, 0, 100, 100])
            risk_level = det.get("risk_level", "Low")
            risk_score = det.get("risk_score", 0.0)
            source     = det.get("source", "")

            x1, y1, x2, y2 = map(int, bbox)
            color = RISK_COLORS.get(risk_level, (0, 220, 90))

            # Box thickness 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {conf:.2f} | {risk_level} R:{risk_score:.2f}"
            if source == "yolov8l":
                label += " [L]"   # marker: came from yolov8l secondary

            font  = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52
            thick = 1
            (tw, th), base = cv2.getTextSize(label, font, scale, thick)
            label_h = th + base + 6

            if y1 - label_h >= 0:
                lt = y1 - label_h;  ty = y1 - base - 3
            else:
                lt = y1;            ty = y1 + th + 3

            cv2.rectangle(out, (x1, lt), (x1 + tw + 8, lt + label_h), color, cv2.FILLED)
            cv2.putText(out, label, (x1 + 4, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

        return out

    # ---- HOT SWAP (edge mode, resolution only) --------------------------------

    def switch_model(self, model_path: str | None, input_size: int):
        """Change input resolution only. Never swap model file."""
        if model_path is not None and os.path.exists(str(model_path)):
            self.model       = YOLO(model_path)
            self.model_path  = model_path
            self.class_names = self.model.names
        self.input_size = input_size
        print(f"[WeaponDetector] Resolution -> {input_size}")
