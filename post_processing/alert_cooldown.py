"""
post_processing/alert_cooldown.py — Alert Cooldown Mechanism

Per-class, per-spatial-region cooldown window: Δt = 5 seconds.
Tracks last alert time per (class, region_key) pair.
If within cooldown: annotate visually but suppress new alert + evidence log entry.
"""

import time
from typing import Dict, Tuple


class AlertCooldown:
    """
    Prevents alert fatigue by enforcing a cooldown period between alerts
    for the same (class, spatial region) combination.

    Parameters
    ----------
    cooldown_seconds : float
        Minimum time between alerts for the same (class, region). Default 5.0 s.
    """

    def __init__(self, cooldown_seconds: float = 5.0):
        self.cooldown_seconds = cooldown_seconds
        # (class_name, region_key) → last_alert_timestamp
        self._last_alert: Dict[Tuple[str, str], float] = {}

    def _make_key(self, class_name: str, region_key: str) -> Tuple[str, str]:
        return (class_name.lower(), str(region_key))

    def should_alert(self, class_name: str, region_key: str = "global") -> bool:
        """
        Check whether a new alert should be fired for this (class, region) pair.

        Parameters
        ----------
        class_name : str
            Weapon class name.
        region_key : str
            Spatial region identifier (e.g., "center", "roi_0", "global").

        Returns
        -------
        bool
            True if cooldown has elapsed and alert is allowed; False otherwise.
        """
        key = self._make_key(class_name, region_key)
        now = time.time()
        last = self._last_alert.get(key, 0.0)

        if now - last >= self.cooldown_seconds:
            self._last_alert[key] = now
            return True
        return False

    def time_remaining(self, class_name: str, region_key: str = "global") -> float:
        """Return seconds remaining in cooldown for a (class, region) pair (0 if ready)."""
        key = self._make_key(class_name, region_key)
        now = time.time()
        last = self._last_alert.get(key, 0.0)
        remaining = self.cooldown_seconds - (now - last)
        return max(0.0, remaining)

    def reset(self, class_name: str | None = None, region_key: str = "global"):
        """Manually reset cooldown for a class/region pair or all entries."""
        if class_name is None:
            self._last_alert.clear()
        else:
            key = self._make_key(class_name, region_key)
            self._last_alert.pop(key, None)
