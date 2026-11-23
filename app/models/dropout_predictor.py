"""Load the trained pipeline and apply the Dropout decision rule.

Serving steps follow notebook Section 8.2: (1) rebuild engineered features,
(2) apply ``log1p`` to the metadata columns, (3) select the feature columns,
(4) call ``predict_proba`` and (5) apply the tuned Dropout threshold.
Calling ``pipeline.predict()`` directly would ignore that threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.models.feature_engineering import add_engineered_features, apply_log_transform
from app.models.schemas import required_input_columns

PREDICTED_STATUS = "Predicted status"
DROPOUT_PROBABILITY = "Dropout probability"


class ArtifactsMissingError(FileNotFoundError):
    """Raised when the exported model files are not in ``artifacts/trained_models``."""


@dataclass(frozen=True)
class DecisionRule:
    """Flag Dropout when P(Dropout) >= ``threshold``, else the likelier class."""

    dropout_class: str
    threshold: float


@dataclass(frozen=True)
class ModelMetadata:
    """Typed view of ``model_metadata.joblib`` (notebook Section 8.1)."""

    feature_columns: list[str]
    log_transformed_cols: list[str]
    class_labels: list[str]
    decision_rule: DecisionRule
    selected_candidate: str

    @classmethod
    def from_dict(cls, raw: dict) -> ModelMetadata:
        rule = raw["decision_rule"]
        return cls(
            feature_columns=list(raw["feature_columns"]),
            log_transformed_cols=list(raw["log_transformed_cols"]),
            class_labels=list(raw["class_labels"]),
            decision_rule=DecisionRule(
                dropout_class=rule["dropout_class"],
                threshold=float(rule["dropout_threshold"]),
            ),
            selected_candidate=raw.get("selected_candidate", ""),
        )


def apply_decision_rule(
    probabilities: np.ndarray, class_labels: list[str], rule: DecisionRule
) -> np.ndarray:
    """Turn class probabilities into status labels with the tuned threshold.

    Args:
        probabilities: Array ``(n_students, n_classes)`` in ``class_labels`` order.
        class_labels: Status names matching the probability columns.
        rule: Dropout threshold rule from the metadata.

    Returns:
        Array of predicted status names.
    """
    dropout_index = class_labels.index(rule.dropout_class)
    others = probabilities.copy()
    others[:, dropout_index] = -np.inf
    predicted = np.where(
        probabilities[:, dropout_index] >= rule.threshold,
        dropout_index,
        others.argmax(axis=1),
    )
    return np.asarray(class_labels)[predicted]


@dataclass(frozen=True)
class DropoutPredictor:
    """Trained pipeline plus the metadata needed to serve it."""

    pipeline: Pipeline
    metadata: ModelMetadata

    @classmethod
    def load(
        cls, pipeline_path: Path, encoder_path: Path, metadata_path: Path
    ) -> DropoutPredictor:
        """Load the three exported artifacts.

        Raises:
            ArtifactsMissingError: If any artifact file is missing.
            ValueError: If the encoder and metadata disagree on class order.
        """
        missing = [
            p.name
            for p in (pipeline_path, encoder_path, metadata_path)
            if not p.exists()
        ]
        if missing:
            raise ArtifactsMissingError(
                f"Missing model files in {pipeline_path.parent}: {', '.join(missing)}. "
                "Run the notebook export (Section 8.1)."
            )
        metadata = ModelMetadata.from_dict(joblib.load(metadata_path))
        encoder_classes = list(joblib.load(encoder_path).classes_)
        if encoder_classes != metadata.class_labels:
            raise ValueError(
                f"Class order mismatch: encoder {encoder_classes} vs "
                f"metadata {metadata.class_labels}."
            )
        return cls(pipeline=joblib.load(pipeline_path), metadata=metadata)

    @property
    def input_columns(self) -> list[str]:
        """Raw columns a user must provide."""
        return required_input_columns(self.metadata.feature_columns)

    @property
    def threshold(self) -> float:
        return self.metadata.decision_rule.threshold

    @property
    def category_options(self) -> dict[str, list[int]]:
        """Codes seen in training for each categorical input (from the encoder)."""
        preprocess = self.pipeline.named_steps["preprocess"]
        encoder = preprocess.named_transformers_["cat"]
        columns = next(
            cols for name, _, cols in preprocess.transformers_ if name == "cat"
        )
        return {
            column: [int(code) for code in codes]
            for column, codes in zip(columns, encoder.categories_, strict=True)
        }

    def prepare_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Steps 1-3: engineered features, log transform, column selection."""
        numeric = raw[self.input_columns].apply(pd.to_numeric)
        # Categorical codes were integers at training time; keep that dtype.
        categorical = [c for c in self.category_options if c in numeric.columns]
        numeric[categorical] = numeric[categorical].astype("int64")
        engineered = add_engineered_features(numeric)
        transformed = apply_log_transform(
            engineered, self.metadata.log_transformed_cols
        )
        return transformed[self.metadata.feature_columns]

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Score students and return probabilities plus the predicted status.

        Args:
            raw: Validated raw records with at least ``input_columns``.

        Returns:
            DataFrame aligned with ``raw`` with one ``P(<status>)`` column per
            class, ``Dropout probability`` and ``Predicted status``.
        """
        probabilities = self.pipeline.predict_proba(self.prepare_features(raw))
        labels = self.metadata.class_labels
        result = pd.DataFrame(
            probabilities, columns=[f"P({label})" for label in labels], index=raw.index
        )
        result[DROPOUT_PROBABILITY] = result[
            f"P({self.metadata.decision_rule.dropout_class})"
        ]
        result[PREDICTED_STATUS] = apply_decision_rule(
            probabilities, labels, self.metadata.decision_rule
        )
        return result
