"""
crop_service.py - single source of truth for validation and prediction.

The Streamlit web app, the FastAPI backend (used by the Expo web/mobile app) and the
notebook integration tests all call CropService, so every interface applies the same
rules and returns the same result for the same input.
"""
import json
import math
from pathlib import Path

import joblib
import pandas as pd

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
NON_NEGATIVE = ["N", "P", "K", "humidity", "rainfall"]

DISCLAIMER = (
    "Prototype trained on a public crop-recommendation dataset, not on measured local farm "
    "data. Scores are model class probabilities, not calibrated suitability scores. Validate "
    "with local agronomic experts before any farming decision."
)


class ValidationError(ValueError):
    """Raised when user input is missing, non-numeric, non-finite or outside valid limits."""


class CropService:
    def __init__(self, model_path=None, metadata_path=None):
        base = Path(__file__).resolve().parent.parent / "model_artifacts"

        self.model = joblib.load(model_path or base / "model.joblib")

        meta_file = Path(metadata_path or base / "model_metadata.json")
        self.metadata = json.loads(meta_file.read_text(encoding="utf-8"))

        if self.metadata["features"] != FEATURES:
            raise ValueError(f"Unexpected feature order in metadata: {self.metadata['features']}")

        self.training_min = self.metadata["training_min"]
        self.training_max = self.metadata["training_max"]

    def info(self):
        """Model details the front ends can display (feature list, classes, training ranges)."""
        return {
            "features": FEATURES,
            "classes": [str(c) for c in self.model.classes_],
            "training_min": self.training_min,
            "training_max": self.training_max,
            "disclaimer": DISCLAIMER,
        }

    def validate(self, user_input):
        if not isinstance(user_input, dict):
            raise ValidationError("Input must be a dictionary of feature names and values.")
        missing = [
            f for f in FEATURES
            if f not in user_input or user_input[f] is None or str(user_input[f]).strip() == ""
        ]
        if missing:
            raise ValidationError("Missing value(s): " + ", ".join(missing))

        values = {}
        for f in FEATURES:
            raw = user_input[f]
            if isinstance(raw, bool):
                raise ValidationError(f"'{f}' must be a numeric value.")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValidationError(f"'{f}' must be a numeric value.")
            if not math.isfinite(value):
                raise ValidationError(f"'{f}' must be a finite number.")
            values[f] = value

        for f in NON_NEGATIVE:
            if values[f] < 0:
                raise ValidationError(f"'{f}' cannot be negative.")
        if values["humidity"] > 100:
            raise ValidationError("'humidity' cannot be above 100%.")
        if not 0 <= values["ph"] <= 14:
            raise ValidationError("'ph' must be between 0 and 14.")
        return values

    def predict(self, user_input):
        values = self.validate(user_input)
        row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

        outside = [
            f for f in FEATURES
            if values[f] < self.training_min[f] or values[f] > self.training_max[f]
        ]
        warnings = []
        if outside:
            warnings.append(
                "Input is outside the training data's observed range for: " + ", ".join(outside)
                + ". The prediction may be less reliable."
            )

        scores = self.model.predict_proba(row)[0]
        return {
            "predicted_crop": str(self.model.predict(row)[0]),
            "model_scores": {str(c): round(float(p), 4) for c, p in zip(self.model.classes_, scores)},
            "warnings": warnings,
            "outside_range": outside,
            "disclaimer": DISCLAIMER,
        }
