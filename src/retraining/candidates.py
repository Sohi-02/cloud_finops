# ============================================================
# FINOPS CHALLENGER TRAINING AND SELECTION
# ============================================================

from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from src.retraining.evaluation import (
    evaluate_candidate_for_promotion
)


CHAMPION_INPUT_COLUMNS = [
    "estimated_cost_index"
]


def calculate_candidate_metrics(
    actual_values,
    predicted_values
) -> dict[str, float]:
    """
    Calculate MAE, RMSE, and bias.

    Positive bias means the model underpredicts.
    Negative bias means the model overpredicts.
    """

    actual = np.asarray(
        actual_values,
        dtype="float64"
    ).reshape(-1)

    predicted = np.asarray(
        predicted_values,
        dtype="float64"
    ).reshape(-1)

    if actual.size == 0:

        raise ValueError(
            "Evaluation data cannot be empty."
        )

    if actual.size != predicted.size:

        raise ValueError(
            "Actual and predicted values must have "
            "the same length."
        )

    if (
        not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
    ):

        raise ValueError(
            "Evaluation values must be finite."
        )

    errors = actual - predicted

    return {
        "mae": float(
            mean_absolute_error(
                actual,
                predicted
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            )
        ),
        "bias": float(
            np.mean(errors)
        )
    }


def _generate_predictions(
    model: Any,
    features: pd.DataFrame
) -> np.ndarray:
    """
    Generate finite, non-negative cost predictions.
    """

    predictions = np.asarray(
        model.predict(features),
        dtype="float64"
    ).reshape(-1)

    if len(predictions) != len(features):

        raise ValueError(
            "Model returned an unexpected number "
            "of predictions."
        )

    if not np.isfinite(
        predictions
    ).all():

        raise ValueError(
            "Model returned non-finite predictions."
        )

    if (
        predictions < 0
    ).any():

        raise ValueError(
            "Model returned negative cost predictions."
        )

    return predictions


def build_notebook_candidate_models(
    random_state: int = 42
) -> dict[str, Any]:
    """
    Build the exact three candidate configurations from
    notebook 04.

    Imports remain local so FastAPI does not need the heavy
    training libraries during application startup.
    """

    try:

        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

    except ImportError as error:

        raise ImportError(
            "The retraining environment requires "
            "xgboost, lightgbm and catboost."
        ) from error

    return {
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.80,
            objective="regression",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        ),
        "catboost": CatBoostRegressor(
            iterations=300,
            depth=4,
            learning_rate=0.05,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=random_state,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False
        )
    }


def train_and_select_challenger(
    split_data: dict[str, Any],
    champion_model: Any,
    candidate_models: Optional[
        dict[str, Any]
    ] = None,
    minimum_improvement_percent: float = 2.0,
    minimum_evaluation_samples: int = 30
) -> dict[str, Any]:
    """
    Train, select, refit, and evaluate a challenger.

    Model selection uses validation MAE. Promotion uses the
    untouched chronological test split.
    """

    required_split_keys = {
        "X_train",
        "y_train",
        "X_validation",
        "y_validation",
        "X_test",
        "y_test"
    }

    missing_split_keys = (
        required_split_keys
        - set(split_data)
    )

    if missing_split_keys:

        raise ValueError(
            "Split data is missing keys: "
            f"{sorted(missing_split_keys)}"
        )

    if champion_model is None:

        raise ValueError(
            "champion_model is required."
        )

    X_train = split_data[
        "X_train"
    ]

    y_train = split_data[
        "y_train"
    ]

    X_validation = split_data[
        "X_validation"
    ]

    y_validation = split_data[
        "y_validation"
    ]

    X_test = split_data[
        "X_test"
    ]

    y_test = split_data[
        "y_test"
    ]

    if candidate_models is None:

        candidate_models = (
            build_notebook_candidate_models()
        )

    if (
        not isinstance(
            candidate_models,
            dict
        )
        or not candidate_models
    ):

        raise ValueError(
            "candidate_models must be a non-empty "
            "dictionary."
        )

    validation_metrics = {}

    candidate_failures = {}

    # --------------------------------------------------------
    # 1. Train each candidate on training data
    # --------------------------------------------------------

    for model_name, model_template in (
        candidate_models.items()
    ):

        try:

            candidate_model = clone(
                model_template
            )

            candidate_model.fit(
                X_train,
                y_train
            )

            validation_predictions = (
                _generate_predictions(
                    model=candidate_model,
                    features=X_validation
                )
            )

            validation_metrics[
                model_name
            ] = calculate_candidate_metrics(
                actual_values=y_validation,
                predicted_values=(
                    validation_predictions
                )
            )

        except Exception as error:

            candidate_failures[
                model_name
            ] = {
                "error_type": (
                    type(error).__name__
                ),
                "message": str(error)
            }

    if not validation_metrics:

        raise RuntimeError(
            "Every candidate failed during training "
            "or validation."
        )

    # --------------------------------------------------------
    # 2. Select candidate using validation MAE
    # --------------------------------------------------------

    selected_candidate_name = min(
        validation_metrics,
        key=lambda model_name: (
            validation_metrics[
                model_name
            ][
                "mae"
            ],
            model_name
        )
    )

    # --------------------------------------------------------
    # 3. Refit selected candidate on train + validation
    # --------------------------------------------------------

    selected_model = clone(
        candidate_models[
            selected_candidate_name
        ]
    )

    X_retraining = pd.concat(
        [
            X_train,
            X_validation
        ],
        axis=0,
        ignore_index=True
    )

    y_retraining = pd.concat(
        [
            y_train,
            y_validation
        ],
        axis=0,
        ignore_index=True
    )

    selected_model.fit(
        X_retraining,
        y_retraining
    )

    # --------------------------------------------------------
    # 4. Evaluate selected candidate on untouched test data
    # --------------------------------------------------------

    candidate_test_predictions = (
        _generate_predictions(
            model=selected_model,
            features=X_test
        )
    )

    candidate_test_metrics = (
        calculate_candidate_metrics(
            actual_values=y_test,
            predicted_values=(
                candidate_test_predictions
            )
        )
    )

    # --------------------------------------------------------
    # 5. Evaluate deployed persistence champion
    # --------------------------------------------------------

    champion_test_input = (
        X_test[
            CHAMPION_INPUT_COLUMNS
        ]
        .astype("float64")
        .copy()
    )

    champion_test_predictions = (
        _generate_predictions(
            model=champion_model,
            features=champion_test_input
        )
    )

    champion_test_metrics = (
        calculate_candidate_metrics(
            actual_values=y_test,
            predicted_values=(
                champion_test_predictions
            )
        )
    )

    # --------------------------------------------------------
    # 6. Apply promotion gate
    # --------------------------------------------------------

    promotion_evaluation = (
        evaluate_candidate_for_promotion(
            champion_mae=(
                champion_test_metrics[
                    "mae"
                ]
            ),
            candidate_mae=(
                candidate_test_metrics[
                    "mae"
                ]
            ),
            evaluation_sample_count=len(
                y_test
            ),
            minimum_evaluation_samples=(
                minimum_evaluation_samples
            ),
            minimum_improvement_percent=(
                minimum_improvement_percent
            )
        )
    )

    return {
        "selected_candidate_name": (
            selected_candidate_name
        ),
        "selected_candidate_model": (
            selected_model
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "candidate_failures": (
            candidate_failures
        ),
        "candidate_test_metrics": (
            candidate_test_metrics
        ),
        "champion_test_metrics": (
            champion_test_metrics
        ),
        "promotion_evaluation": (
            promotion_evaluation
        ),
        "training_report": {
            "training_samples": int(
                len(X_train)
            ),
            "validation_samples": int(
                len(X_validation)
            ),
            "retraining_samples": int(
                len(X_retraining)
            ),
            "test_samples": int(
                len(X_test)
            ),
            "feature_count": int(
                X_train.shape[1]
            )
        }
    }