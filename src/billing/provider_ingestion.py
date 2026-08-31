from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.billing.ingestion import prepare_cost_history


def ingest_provider_costs(
    raw_history: pd.DataFrame,
    provider_name: str,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize provider billing data and summarize it for downstream use."""
    if csv_path is not None:
        data = pd.read_csv(csv_path)
    else:
        data = raw_history

    prepared = prepare_cost_history(data)
    filtered = prepared[prepared["provider"].astype(str).str.lower() == provider_name.lower()].copy()
    if filtered.empty:
        return {
            "provider": provider_name,
            "row_count": 0,
            "total_cost_usd": 0.0,
            "service_totals": {},
            "time_range_start": None,
            "time_range_end": None,
        }

    service_totals = (
        filtered.groupby("service", dropna=False)["cost_usd"]
        .sum()
        .astype(float)
        .to_dict()
    )

    summary = {
        "provider": provider_name,
        "row_count": int(len(filtered)),
        "total_cost_usd": float(filtered["cost_usd"].sum()),
        "service_totals": {str(key): float(value) for key, value in service_totals.items()},
        "time_range_start": filtered["time_bucket"].min().isoformat(),
        "time_range_end": filtered["time_bucket"].max().isoformat(),
    }
    return summary
