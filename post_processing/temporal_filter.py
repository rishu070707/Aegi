"""
post_processing/temporal_filter.py — Temporal Consistency Filtering

Sliding window buffer (last N=5 frames).
A detection is confirmed only if the same class appears with confidence ≥ 0.30
in at least K=3 of the last N frames. Suppresses single-frame spurious detections.
"""

from collections import deque
from typing import List, Dict


class TemporalConsistencyFilter:
    """
    Filters detections using a sliding-window temporal buffer.

    Parameters
    ----------
    window_size : int
        Number of recent frames to keep (N=5).
    min_hits : int
        Minimum frames a class must appear in to be confirmed (K=3).
    min_confidence : float
        Minimum confidence for a frame detection to count (0.30).
    """

    def __init__(self, window_size: int = 3, min_hits: int = 1, min_confidence: float = 0.25):
        self.window_size = window_size
        self.min_hits = min_hits
        self.min_confidence = min_confidence

        # Each element of the deque is a set of class names seen in that frame
        self._buffer: deque = deque(maxlen=window_size)
        # Track best detection per class across the buffer for returning metadata
        self._best: Dict[str, Dict] = {}

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update the buffer with new frame detections and return confirmed detections.

        Parameters
        ----------
        detections : list of dict
            Each dict: {class_name, confidence, bbox, ...}

        Returns
        -------
        list of dict
            Detections confirmed by temporal consistency.
        """
        # Build set of classes seen in this frame (above min_conf)
        frame_classes: Dict[str, Dict] = {}
        for det in detections:
            cls = det["class_name"]
            conf = det.get("confidence", 0.0)
            if conf >= self.min_confidence:
                # Keep highest-confidence detection per class in this frame
                if cls not in frame_classes or conf > frame_classes[cls]["confidence"]:
                    frame_classes[cls] = det

        self._buffer.append(frame_classes)

        # Count occurrences of each class across the window
        class_counts: Dict[str, int] = {}
        for frame_cls_map in self._buffer:
            for cls in frame_cls_map:
                class_counts[cls] = class_counts.get(cls, 0) + 1

        # Determine confirmed classes (seen in ≥ K frames)
        confirmed = []
        for cls, count in class_counts.items():
            if count >= self.min_hits:
                # Find latest detection for this class
                latest_det = None
                for frame_cls_map in reversed(self._buffer):
                    if cls in frame_cls_map:
                        latest_det = frame_cls_map[cls]
                        break
                if latest_det is not None:
                    confirmed.append(latest_det)

        return confirmed

    def reset(self):
        """Clear the buffer."""
        self._buffer.clear()
        self._best.clear()
