"""
post_processing/confidence_stabilizer.py — Anti-Flicker via EMA

EMA formula: S_hat(t) = α * C(t) + (1 - α) * S_hat(t-1)
where α = 0.4

Per-class tracking of smoothed confidence values.
"""

from typing import Dict


class ConfidenceStabilizer:
    """
    Exponential Moving Average (EMA) smoother for per-class detection confidence.
    Reduces temporal flickering of confidence scores.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (0 < α ≤ 1). Default 0.4.
        Higher α = more reactive to new values; lower α = smoother.
    """

    def __init__(self, alpha: float = 0.4):
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._ema: Dict[str, float] = {}  # class_name → smoothed confidence

    def smooth(self, class_name: str, raw_confidence: float) -> float:
        """
        Apply EMA smoothing to a raw confidence value for a given class.

        Parameters
        ----------
        class_name : str
            Weapon class name (e.g., "Handgun").
        raw_confidence : float
            Raw model confidence in [0, 1].

        Returns
        -------
        float
            EMA-smoothed confidence value.
        """
        raw_confidence = float(raw_confidence)
        if class_name not in self._ema:
            # Initialize with first observed value
            self._ema[class_name] = raw_confidence
        else:
            self._ema[class_name] = (
                self.alpha * raw_confidence + (1.0 - self.alpha) * self._ema[class_name]
            )
        return round(self._ema[class_name], 4)

    def get_ema(self, class_name: str) -> float:
        """Return current EMA for a class (0.0 if never seen)."""
        return self._ema.get(class_name, 0.0)

    def reset(self, class_name: str | None = None):
        """Reset EMA state for one or all classes."""
        if class_name:
            self._ema.pop(class_name, None)
        else:
            self._ema.clear()
