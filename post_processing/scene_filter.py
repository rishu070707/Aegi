"""
post_processing/scene_filter.py — Scene-Aware False Alarm Suppression

Runs YOLOv8n person detector on each frame.
Checks spatial proximity of weapon bbox centroid to person bbox centroid.

Context multiplier psi rules:
  Weapon + Human co-located (norm. distance < 0.3): psi = 1.0   (confirmed threat)
  Weapon + Human present but not proximate:         psi = 0.75  (probable threat)
  Weapon alone in frame (no person detected):       psi = 0.50  (still plausible)

effective_confidence = Cs * ψ
Suppress if effective_confidence < inference threshold (0.25)
"""

import numpy as np
from typing import List, Dict, Tuple

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

PERSON_CLASS_ID = 0   # COCO person class


class SceneAwareFilter:
    """
    Filters weapon detections based on human co-occurrence context.

    Parameters
    ----------
    person_model_path : str
        Path to YOLOv8n model for person detection.
    conf_threshold : float
        Minimum effective confidence to keep a detection.
    proximity_threshold : float
        Normalized distance threshold for 'co-located' classification.
    """

    def __init__(
        self,
        person_model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        proximity_threshold: float = 0.3,
    ):
        self.conf_threshold = conf_threshold
        self.proximity_threshold = proximity_threshold
        self._person_model = None

        if _YOLO_AVAILABLE:
            try:
                self._person_model = _YOLO(person_model_path)
                print(f"[SceneAwareFilter] Loaded person model: {person_model_path}")
            except Exception as e:
                print(f"[SceneAwareFilter] Warning: Could not load person model: {e}")

    def _detect_persons(self, frame: np.ndarray) -> List[List[int]]:
        """Run person detection and return list of bboxes [x1,y1,x2,y2]."""
        if self._person_model is None:
            return []
        try:
            results = self._person_model(frame, classes=[PERSON_CLASS_ID], conf=0.30, verbose=False)[0]
            persons = []
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if int(cls) == PERSON_CLASS_ID:
                    persons.append(list(map(int, box.tolist())))
            return persons
        except Exception:
            return []

    @staticmethod
    def _centroid(bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _normalized_distance(
        c1: Tuple[float, float],
        c2: Tuple[float, float],
        frame_shape: Tuple[int, int],
    ) -> float:
        h, w = frame_shape[:2]
        dx = (c1[0] - c2[0]) / w
        dy = (c1[1] - c2[1]) / h
        return float(np.sqrt(dx ** 2 + dy ** 2))

    def filter(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Apply scene-context multiplier ψ and suppress low-confidence detections.

        Parameters
        ----------
        detections : list of dict
            Raw weapon detections with at least {class_name, confidence, bbox}.
        frame : np.ndarray
            Current video frame (BGR).

        Returns
        -------
        list of dict
            Filtered detections with updated effective_confidence field.
        """
        person_bboxes = self._detect_persons(frame)
        frame_shape = frame.shape
        filtered = []

        for det in detections:
            weapon_centroid = self._centroid(det["bbox"])
            cs = det.get("confidence", 0.0)

            if cs >= 0.95:
                # Extreme high certainty detections bypassing context decay
                psi = 1.0
            elif not person_bboxes:
                psi = 0.50  # weapon alone
            else:
                # Find closest person
                distances = [
                    self._normalized_distance(
                        weapon_centroid, self._centroid(pb), frame_shape
                    )
                    for pb in person_bboxes
                ]
                min_dist = min(distances)
                if min_dist < self.proximity_threshold:
                    psi = 1.0
                else:
                    psi = 0.75

            effective_conf = round(cs * psi, 4)
            det = dict(det)
            det["effective_confidence"] = effective_conf
            det["psi"] = psi

            if effective_conf >= self.conf_threshold:
                filtered.append(det)

        return filtered
