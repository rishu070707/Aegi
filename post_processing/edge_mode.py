"""
post_processing/edge_mode.py — Adaptive Edge Deployment Mode

Monitors inference latency. If latency is consistently high, reduces
input resolution to speed up inference.

IMPORTANT: Never swaps the model file — only changes input_size.
This preserves custom weapon weights across all performance tiers.
"""

from collections import deque
from typing import Dict


class EdgeModeManager:
    """
    Adapts inference speed by changing input resolution only.
    Does NOT switch model files — the weapon model stays loaded always.

    Tiers
    -----
    full : imgsz=640  (best accuracy, used when latency < 120ms)
    edge : imgsz=416  (faster, used when latency > 200ms sustained)
    """

    def __init__(
        self,
        high_latency_threshold_ms: float = 200.0,   # raise from 40 -> 200
        low_latency_threshold_ms:  float = 150.0,
        recovery_window: int = 15,
    ):
        self.high_threshold   = high_latency_threshold_ms
        self.low_threshold    = low_latency_threshold_ms
        self.recovery_window  = recovery_window

        self._current_mode    = "full"
        self._recovery_counter = 0
        self._latency_history: deque = deque(maxlen=50)

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def check_and_adapt(self, current_latency: float) -> Dict:
        """
        Evaluate latency. Returns a dict with:
          model_variant : None  (never swap model file)
          input_size    : int   (640 or 416)
          mode_changed  : bool
        """
        self._latency_history.append(current_latency)
        mode_changed = False

        if self._current_mode == "full":
            if current_latency > self.high_threshold:
                self._current_mode = "edge"
                self._recovery_counter = 0
                mode_changed = True
                print(
                    f"[EdgeMode] -> edge (latency={current_latency:.0f}ms, "
                    f"imgsz now 416)"
                )

        elif self._current_mode == "edge":
            if current_latency < self.low_threshold:
                self._recovery_counter += 1
                if self._recovery_counter >= self.recovery_window:
                    self._current_mode = "full"
                    self._recovery_counter = 0
                    mode_changed = True
                    print(
                        f"[EdgeMode] -> full (latency={current_latency:.0f}ms, "
                        f"imgsz now 640)"
                    )
            else:
                self._recovery_counter = 0

        input_size = 640 if self._current_mode == "full" else 416
        return {
            "model_variant": None,    # Never swap model — keep weapon_model.pt
            "input_size":    input_size,
            "mode_changed":  mode_changed,
            "current_mode":  self._current_mode,
        }

    def get_stats(self) -> Dict:
        history = list(self._latency_history)
        avg = round(sum(history) / len(history), 2) if history else 0.0
        return {
            "current_mode":    self._current_mode,
            "avg_latency_ms":  avg,
            "last_latency_ms": round(history[-1], 2) if history else 0.0,
            "recovery_counter": self._recovery_counter,
        }
