# ============================================================
# MONGODB PREDICTION STORAGE
# ============================================================

from datetime import datetime, timezone

from pymongo import (
    ASCENDING,
    DESCENDING,
    MongoClient
)
from pymongo.errors import PyMongoError


class MongoPredictionStore:
    """
    Store predictions, actual results, and production input
    features in MongoDB.
    """

    def __init__(
        self,
        mongodb_uri,
        database_name="finops_monitoring",
        collection_name="prediction_logs"
    ):

        if not mongodb_uri:
            raise ValueError(
                "MONGODB_URI must not be empty."
            )

        self.client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
            tz_aware=True
        )

        try:

            # MongoClient normally connects lazily.
            # ping forces it to verify the connection now.
            self.client.admin.command("ping")

        except PyMongoError as error:

            raise ConnectionError(
                "Could not connect to MongoDB."
            ) from error

        self.database = self.client[
            database_name
        ]

        self.collection = self.database[
            collection_name
        ]

        self._create_indexes()

    # --------------------------------------------------------
    # MongoDB indexes
    # --------------------------------------------------------

    def _create_indexes(self):

        # Every prediction must have a unique identifier.
        self.collection.create_index(
            [
                (
                    "prediction_id",
                    ASCENDING
                )
            ],
            unique=True
        )

        # Speeds up recent-prediction queries.
        self.collection.create_index(
            [
                (
                    "prediction_timestamp_utc",
                    DESCENDING
                )
            ]
        )

        # Helps retrieve records belonging to one model version.
        self.collection.create_index(
            [
                (
                    "model.version",
                    ASCENDING
                )
            ]
        )

        # Helps separate completed predictions from predictions
        # that are still waiting for their actual values.
        self.collection.create_index(
            [
                (
                    "status",
                    ASCENDING
                )
            ]
        )

    # --------------------------------------------------------
    # Store a new prediction
    # --------------------------------------------------------

    def save_prediction(
        self,
        prediction_id,
        input_features,
        predicted_next_hour_cost,
        registered_model,
        model_alias,
        model_version,
        prediction_timestamp_utc=None,
        data_quality_report=None
    ):

        if prediction_timestamp_utc is None:

            prediction_timestamp_utc = (
                datetime.now(timezone.utc)
            )

        prediction_document = {
            "prediction_id": prediction_id,
            "prediction_timestamp_utc": (
                prediction_timestamp_utc
            ),
            "input_features": input_features,
            "data_quality": (
                data_quality_report
                if data_quality_report is not None
                else {
                    "passed": True,
                    "errors": [],
                    "warnings": []
                }
            ),
            "prediction": {
                "predicted_next_hour_cost": float(
                    predicted_next_hour_cost
                )
            },
            "actual": {
                "actual_next_hour_cost": None,
                "absolute_error": None,
                "received_timestamp_utc": None
            },
            "model": {
                "registered_model": registered_model,
                "alias": model_alias,
                "version": str(model_version)
            },
            "status": "awaiting_actual"
        }

        result = self.collection.insert_one(
            prediction_document
        )

        return str(result.inserted_id)

    # --------------------------------------------------------
    # Attach the real next-hour cost
    # --------------------------------------------------------

    def record_actual_cost(
        self,
        prediction_id,
        actual_next_hour_cost
    ):

        prediction_document = (
            self.collection.find_one(
                {
                    "prediction_id": prediction_id
                }
            )
        )

        if prediction_document is None:

            raise KeyError(
                "Prediction ID was not found."
            )

        if (
            prediction_document["status"]
            == "completed"
        ):

            raise ValueError(
                "Actual cost has already been recorded."
            )

        predicted_cost = float(
            prediction_document[
                "prediction"
            ][
                "predicted_next_hour_cost"
            ]
        )

        actual_cost = float(
            actual_next_hour_cost
        )

        absolute_error = abs(
            actual_cost - predicted_cost
        )

        update_result = self.collection.update_one(
            {
                "prediction_id": prediction_id,
                "status": "awaiting_actual"
            },
            {
                "$set": {
                    "actual.actual_next_hour_cost": (
                        actual_cost
                    ),
                    "actual.absolute_error": (
                        absolute_error
                    ),
                    "actual.received_timestamp_utc": (
                        datetime.now(timezone.utc)
                    ),
                    "status": "completed"
                }
            }
        )

        if update_result.modified_count != 1:

            raise RuntimeError(
                "Actual cost could not be recorded."
            )

        return {
            "prediction_id": prediction_id,
            "predicted_next_hour_cost": (
                predicted_cost
            ),
            "actual_next_hour_cost": actual_cost,
            "absolute_error": absolute_error
        }

    # --------------------------------------------------------
    # Read completed records for performance monitoring
    # --------------------------------------------------------

    def get_completed_predictions(
        self,
        limit=1000
    ):

        cursor = (
            self.collection
            .find(
                {
                    "status": "completed",
                    (
                        "actual."
                        "actual_next_hour_cost"
                    ): {
                        "$ne": None
                    }
                },
                {
                    "_id": 0
                }
            )
            .sort(
                "prediction_timestamp_utc",
                DESCENDING
            )
            .limit(int(limit))
        )

        return list(cursor)

    # --------------------------------------------------------
    # Read complete recent prediction records
    # --------------------------------------------------------

    def get_recent_predictions(
        self,
        limit=100
    ):

        cursor = (
            self.collection
            .find(
                {},
                {
                    "_id": 0
                }
            )
            .sort(
                "prediction_timestamp_utc",
                DESCENDING
            )
            .limit(int(limit))
        )

        return list(cursor)

    # --------------------------------------------------------
    # Read recent feature values for drift monitoring
    # --------------------------------------------------------

    def get_recent_feature_values(
        self,
        feature_name,
        limit=1000,
        model_version=None
    ):
        """
        Return recent valid production values for one feature.

        Prediction inputs are stored under:

        input_features.<feature_name>
        """

        if not isinstance(feature_name, str):

            raise TypeError(
                "Feature name must be a string."
            )

        if (
            not feature_name
            or "." in feature_name
            or feature_name.startswith("$")
        ):

            raise ValueError(
                "Feature name is invalid."
            )

        try:
            numeric_limit = int(limit)

        except (TypeError, ValueError) as error:

            raise ValueError(
                "Limit must be an integer."
            ) from error

        if numeric_limit < 1:

            raise ValueError(
                "Limit must be at least 1."
            )

        feature_path = (
            f"input_features.{feature_name}"
        )

        query = {
            feature_path: {
                "$exists": True,
                "$ne": None
            }
        }

        # Compare only records generated by the model version
        # associated with the reference profile.
        if model_version is not None:

            query["model.version"] = str(
                model_version
            )

        cursor = (
            self.collection
            .find(
                query,
                {
                    "_id": 0,
                    feature_path: 1,
                    "prediction_timestamp_utc": 1
                }
            )
            .sort(
                "prediction_timestamp_utc",
                DESCENDING
            )
            .limit(numeric_limit)
        )

        feature_values = []

        for document in cursor:

            value = (
                document
                .get("input_features", {})
                .get(feature_name)
            )

            if (
                value is None
                or isinstance(value, bool)
            ):
                continue

            try:
                numeric_value = float(value)

            except (TypeError, ValueError):
                continue

            feature_values.append(
                numeric_value
            )

        return feature_values

    # --------------------------------------------------------
    # Close MongoDB connection
    # --------------------------------------------------------

    def close(self):

        self.client.close()