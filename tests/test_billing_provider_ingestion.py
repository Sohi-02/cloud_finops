import pandas as pd

from src.billing.provider_ingestion import ingest_provider_costs


def test_ingest_provider_costs_normalizes_and_summarizes():
    raw = pd.DataFrame(
        {
            "time_bucket": ["2026-08-03T02:00:00Z", "2026-08-03T01:00:00Z"],
            "provider": ["AWS", "AWS"],
            "service": ["EC2", "S3"],
            "cost_usd": [12.5, 7.5],
        }
    )

    summary = ingest_provider_costs(raw, provider_name="AWS")

    assert summary["provider"] == "AWS"
    assert summary["row_count"] == 2
    assert summary["total_cost_usd"] == 20.0
    assert summary["service_totals"]["S3"] == 7.5
