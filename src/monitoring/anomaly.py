from __future__ import annotations

from typing import Iterable

import numpy as np


def detect_anomalies(values: Iterable[float]) -> dict:
    """Flag a sharp cost spike in a sequence of observed values."""
    array = np.asarray(list(values), dtype="float64")

    if array.size == 0:
        return {
            "flagged": False,
            "severity": "none",
            "score": 0.0,
            "max_value": None,
            "mean": None,
            "std": None,
        }

    mean_value = float(np.mean(array))
    std_value = float(np.std(array, ddof=0))
    max_value = float(np.max(array))

    if std_value == 0:
        z_score = 0.0
    else:
        z_score = abs(max_value - mean_value) / std_value

    spike_ratio = max_value / max(mean_value, 1e-9)
    flagged = bool(z_score >= 3.0 or spike_ratio >= 2.5)

    if not flagged:
        severity = "normal"
    elif z_score >= 6.0 or spike_ratio >= 5.0:
        severity = "critical"
    elif z_score >= 4.0 or spike_ratio >= 3.0:
        severity = "high"
    else:
        severity = "moderate"

    return {
        "flagged": flagged,
        "severity": severity,
        "score": float(z_score),
        "max_value": max_value,
        "mean": mean_value,
        "std": std_value,
    }
