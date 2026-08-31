import pandas as pd

from src.billing.ingestion import prepare_cost_history


def test_prepare_cost_history_normalizes_and_orders_rows():
    raw = pd.DataFrame(
        {
            "time_bucket": ["2026-08-03T02:00:00Z", "2026-08-03T01:00:00Z"],
            "provider": ["AWS", "AWS"],
            "service": ["EC2", "EC2"],
            "cost_usd": [12.5, 10.0],
        }
    )

    prepared = prepare_cost_history(raw)

    assert list(prepared.columns) == [
        "time_bucket",
        "provider",
        "service",
        "cost_usd",
    ]
    assert prepared["time_bucket"].is_monotonic_increasing
    assert prepared["cost_usd"].tolist() == [10.0, 12.5]
