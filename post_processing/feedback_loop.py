"""
post_processing/feedback_loop.py — User Feedback Learning Loop

Stores operator feedback (Correct / Incorrect) for each detection.
Saves to feedback_data/feedback_log.csv.
"""

import os
import csv
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional


FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "feedback_data")
FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "feedback_log.csv")

CSV_COLUMNS = [
    "timestamp", "detection_id", "frame_id", "class",
    "confidence", "bbox", "risk_score", "label"
]


class FeedbackLoop:
    """
    Records operator feedback on detection accuracy for continuous improvement.

    Feedback is stored as CSV with columns:
      timestamp, detection_id, frame_id, class, confidence, bbox, risk_score, label
    """

    def __init__(self, feedback_dir: str = FEEDBACK_DIR):
        self.feedback_dir = feedback_dir
        self.feedback_csv = os.path.join(feedback_dir, "feedback_log.csv")
        os.makedirs(self.feedback_dir, exist_ok=True)

        # Create CSV with header if it doesn't exist
        if not os.path.exists(self.feedback_csv):
            with open(self.feedback_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

        # In-memory store for active session detections
        self._active_detections: Dict[str, Dict] = {}

    def register_detection(self, detection_id: str, detection: Dict, frame_id: str = ""):
        """
        Register a detection for potential feedback.

        Parameters
        ----------
        detection_id : str
            Unique ID for this detection instance.
        detection : dict
            Detection metadata {class_name, confidence, bbox, risk_score}.
        frame_id : str
            Optional frame identifier.
        """
        self._active_detections[detection_id] = {
            "detection_id": detection_id,
            "frame_id": frame_id,
            "class": detection.get("class_name", "Unknown"),
            "confidence": detection.get("confidence", 0.0),
            "bbox": json.dumps(detection.get("bbox", [])),
            "risk_score": detection.get("risk_score", 0.0),
        }

    def record_feedback(self, detection_id: str, label: str) -> bool:
        """
        Record operator feedback for a detection.

        Parameters
        ----------
        detection_id : str
            Detection to provide feedback on.
        label : str
            "correct" or "incorrect".

        Returns
        -------
        bool
            True if recorded successfully.
        """
        label = label.lower().strip()
        if label not in ("correct", "incorrect"):
            print(f"[FeedbackLoop] Invalid label '{label}'. Must be 'correct' or 'incorrect'.")
            return False

        det = self._active_detections.get(detection_id)
        if det is None:
            # Create minimal entry if detection not registered
            det = {"detection_id": detection_id, "frame_id": "", "class": "Unknown",
                   "confidence": 0.0, "bbox": "[]", "risk_score": 0.0}

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detection_id": detection_id,
            "frame_id": det.get("frame_id", ""),
            "class": det.get("class", "Unknown"),
            "confidence": det.get("confidence", 0.0),
            "bbox": det.get("bbox", "[]"),
            "risk_score": det.get("risk_score", 0.0),
            "label": label,
        }

        try:
            with open(self.feedback_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(row)
            return True
        except Exception as e:
            print(f"[FeedbackLoop] Error writing feedback: {e}")
            return False

    def get_feedback_stats(self) -> Dict:
        """
        Compute aggregate feedback statistics.

        Returns
        -------
        dict
            {total, correct, incorrect, accuracy_pct, by_class: {class: {correct, incorrect}}}
        """
        stats: Dict = {
            "total": 0, "correct": 0, "incorrect": 0,
            "accuracy_pct": 0.0, "by_class": {}
        }

        try:
            if not os.path.exists(self.feedback_csv):
                return stats

            with open(self.feedback_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats["total"] += 1
                    label = row.get("label", "").lower()
                    cls = row.get("class", "Unknown")

                    if cls not in stats["by_class"]:
                        stats["by_class"][cls] = {"correct": 0, "incorrect": 0}

                    if label == "correct":
                        stats["correct"] += 1
                        stats["by_class"][cls]["correct"] += 1
                    elif label == "incorrect":
                        stats["incorrect"] += 1
                        stats["by_class"][cls]["incorrect"] += 1

            if stats["total"] > 0:
                stats["accuracy_pct"] = round(stats["correct"] / stats["total"] * 100, 1)

        except Exception as e:
            print(f"[FeedbackLoop] Error reading feedback: {e}")

        return stats

    def get_recent_feedback(self, n: int = 20) -> List[Dict]:
        """Return the N most recent feedback entries."""
        entries = []
        try:
            with open(self.feedback_csv, "r") as f:
                reader = csv.DictReader(f)
                entries = list(reader)
        except Exception:
            pass
        return entries[-n:][::-1]  # newest first
