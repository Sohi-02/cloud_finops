from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COST_HISTORY_COLUMNS = [
    "time_bucket",
    "provider",
    "service",
    "cost_usd",
]


def prepare_cost_history(raw_history: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate an incoming billing history table."""
    if not isinstance(raw_history, pd.DataFrame):
        raise TypeError("raw_history must be a pandas DataFrame.")

    missing_columns = set(REQUIRED_COST_HISTORY_COLUMNS) - set(raw_history.columns)
    if missing_columns:
        raise ValueError(
            "Billing history is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = raw_history[REQUIRED_COST_HISTORY_COLUMNS].copy()
    prepared["time_bucket"] = pd.to_datetime(
        prepared["time_bucket"],
        utc=True,
        errors="coerce",
    )
    prepared["cost_usd"] = pd.to_numeric(
        prepared["cost_usd"],
        errors="coerce",
    )

    prepared = prepared.dropna(subset=["time_bucket", "cost_usd"]).copy()
    if prepared.empty:
        raise ValueError("Billing history does not contain valid rows.")

    prepared = prepared.sort_values("time_bucket", ascending=True).reset_index(drop=True)
    return prepared


def load_cost_history(csv_path: str | Path) -> pd.DataFrame:
    """Load a billing-history CSV and prepare it for downstream use."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Billing history file not found: {path}")

    raw_history = pd.read_csv(path)
    return prepare_cost_history(raw_history)
