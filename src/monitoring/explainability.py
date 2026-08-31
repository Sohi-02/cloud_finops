from __future__ import annotations

from typing import Any, Iterable

import numpy as np

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None


def explain_model(model_summary: Any, feature_matrix: Iterable[Iterable[float]]) -> dict:
    """Return a lightweight explanation payload for a learned model."""
    if isinstance(model_summary, dict) and model_summary.get("model_type") == "persistence_baseline":
        return {
            "supported": False,
            "reason": (
                "Persistence baseline has no learned feature importance and "
                "SHAP is not meaningful for a one-point persistence model."
            ),
            "top_features": [],
        }

    features = np.asarray(list(feature_matrix), dtype="float64")
    if features.size == 0:
        return {
            "supported": False,
            "reason": "No feature matrix was provided for explanation.",
            "top_features": [],
        }

    model_features = []
    if isinstance(model_summary, dict):
        model_features = model_summary.get("feature_names", []) or []

    if hasattr(model_summary, "feature_names_in_"):
        model_features = list(model_summary.feature_names_in_)

    if shap is not None and hasattr(model_summary, "predict"):
        try:
            explainer = shap.Explainer(model_summary)
            values = explainer(features)
            if hasattr(values, "values"):
                values_array = np.asarray(values.values)
                if values_array.ndim == 3:
                    values_array = values_array.reshape(values_array.shape[0], -1)
                if values_array.shape[1] == features.shape[1]:
                    mean_abs = np.abs(values_array).mean(axis=0)
                    ranking = np.argsort(mean_abs)[::-1]
                    top_features = []
                    for idx in ranking[: min(5, len(ranking))]:
                        label = model_features[idx] if idx < len(model_features) else f"feature_{idx}"
                        top_features.append({"feature": label, "importance": float(mean_abs[idx])})
                    return {
                        "supported": True,
                        "reason": "SHAP values were computed for the deployed model.",
                        "top_features": top_features,
                    }
        except Exception:
            pass

    if hasattr(model_summary, "feature_importances_"):
        importances = np.asarray(model_summary.feature_importances_, dtype="float64")
        if features.shape[1] == importances.shape[0]:
            ranking = np.argsort(importances)[::-1]
            top_features = []
            for idx in ranking[: min(5, len(ranking))]:
                label = model_features[idx] if idx < len(model_features) else f"feature_{idx}"
                top_features.append({"feature": label, "importance": float(importances[idx])})
            return {
                "supported": True,
                "reason": "Tree-based or learned feature importance was used for explanation.",
                "top_features": top_features,
            }

    if hasattr(model_summary, "coef_"):
        coefficients = np.asarray(model_summary.coef_, dtype="float64")
        if coefficients.ndim == 1 and features.shape[1] == coefficients.shape[0]:
            ranking = np.argsort(np.abs(coefficients))[::-1]
            top_features = []
            for idx in ranking[: min(5, len(ranking))]:
                label = model_features[idx] if idx < len(model_features) else f"feature_{idx}"
                top_features.append({"feature": label, "importance": float(coefficients[idx])})
            return {
                "supported": True,
                "reason": "Linear coefficients were used for explanation.",
                "top_features": top_features,
            }

    if shap is None:
        return {
            "supported": False,
            "reason": "SHAP is not installed in the environment; explainability is unavailable.",
            "top_features": [],
        }

    return {
        "supported": False,
        "reason": "Model explanation is unavailable for this model type.",
        "top_features": [],
    }
