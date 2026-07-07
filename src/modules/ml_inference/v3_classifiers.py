"""
V3 Classifiers - Embedding-based classifiers for base_color, design, tonality

Uses pre-trained model pickling. Falls back to KNN prototype matching.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.modules.ml_inference import embeddings as v3_embeddings

MODELS_DIR = Path("models")
CONFIDENCE_CAP_SMALL_DATASET = 0.85
SMALL_DATASET_THRESHOLD = 100


@dataclass
class ClassifierPrediction:
    label: str
    confidence: float
    method: str

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "method": self.method,
        }


class V3Classifier:
    """Classifier for a single field (base_color, design, or tonality)."""

    def __init__(self, field_name: str, valid_labels: List[str]):
        self.field_name = field_name
        self.valid_labels = valid_labels
        self.model = None
        self.label_encoder = None
        self.prototypes = None
        self.training_count = 0
        self.is_trained = False

    def predict(self, embedding: np.ndarray) -> Optional[ClassifierPrediction]:
        """Predict label for an embedding."""
        if not self.is_trained:
            return None

        embedding = embedding / np.linalg.norm(embedding)

        if self.model is not None and self.label_encoder is not None:
            try:
                probs = self.model.predict_proba(embedding.reshape(1, -1))[0]
                pred_idx = probs.argmax()
                confidence = float(probs[pred_idx])
                label = self.label_encoder.inverse_transform([pred_idx])[0]

                if self.training_count < SMALL_DATASET_THRESHOLD:
                    confidence = min(confidence, CONFIDENCE_CAP_SMALL_DATASET)

                return ClassifierPrediction(
                    label=label, confidence=confidence, method="logistic_regression"
                )
            except Exception as e:
                print(
                    f"[V3 Classifier] {self.field_name}: Predict failed, using KNN: {e}"
                )

        if self.prototypes:
            return self._predict_knn(embedding)

        return None

    def _predict_knn(self, embedding: np.ndarray) -> Optional[ClassifierPrediction]:
        """Predict using KNN prototype matching."""
        if not self.prototypes:
            return None

        similarities = {}
        for label, prototype in self.prototypes.items():
            sim = float(np.dot(embedding, prototype))
            similarities[label] = sim

        best_label = max(similarities, key=similarities.get)
        best_sim = similarities[best_label]

        sorted_sims = sorted(similarities.values(), reverse=True)
        if len(sorted_sims) > 1:
            margin = sorted_sims[0] - sorted_sims[1]
        else:
            margin = 0.5

        confidence = min(0.5 + margin, 0.95)

        if self.training_count < SMALL_DATASET_THRESHOLD:
            confidence = min(confidence, CONFIDENCE_CAP_SMALL_DATASET)

        return ClassifierPrediction(
            label=best_label, confidence=confidence, method="knn_prototype"
        )

    @classmethod
    def load(
        cls, field_name: str, valid_labels: List[str], path: Optional[Path] = None
    ) -> "V3Classifier":
        """Load classifier from disk."""
        if path is None:
            path = MODELS_DIR / f"v3_{field_name}.pkl"

        classifier = cls(field_name, valid_labels)

        if not path.exists():
            return classifier

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            classifier.model = state.get("model")
            classifier.label_encoder = state.get("label_encoder")
            classifier.prototypes = state.get("prototypes")
            classifier.training_count = state.get("training_count", 0)
            classifier.is_trained = state.get("is_trained", False)

            print(f"[V3 Classifier] Loaded {field_name} from {path}")
        except Exception as e:
            print(f"[V3 Classifier] Failed to load {field_name}: {e}")

        return classifier


VALID_BASE_COLORS = [
    "Grey",
    "White",
    "Blue",
    "Black",
    "Pink",
    "Red",
    "Rose",
    "Brown",
    "Yellow",
    "Orange",
    "Green",
    "Beige",
    "Exotic",
    "Charcoal",
    "Purple",
    "Gold",
    "Multi",
]

VALID_DESIGNS = ["Plain", "Veiny", "Spotty", "Cloudy"]

VALID_TONALITIES = ["Light", "Medium", "Dark"]


class V3ClassifierManager:
    """Manages all three classifiers."""

    def __init__(self):
        self.base_color_clf = V3Classifier("base_color", VALID_BASE_COLORS)
        self.design_clf = V3Classifier("design", VALID_DESIGNS)
        self.tonality_clf = V3Classifier("tonality", VALID_TONALITIES)

    def predict_all(
        self, embedding: np.ndarray
    ) -> Dict[str, Optional[ClassifierPrediction]]:
        """Predict all fields for an embedding."""
        return {
            "base_color": self.base_color_clf.predict(embedding),
            "design": self.design_clf.predict(embedding),
            "tonality": self.tonality_clf.predict(embedding),
        }

    def load_all(self):
        """Load all classifiers."""
        self.base_color_clf = V3Classifier.load("base_color", VALID_BASE_COLORS)
        self.design_clf = V3Classifier.load("design", VALID_DESIGNS)
        self.tonality_clf = V3Classifier.load("tonality", VALID_TONALITIES)


_classifier_manager = None


def get_classifier_manager() -> V3ClassifierManager:
    """Get singleton classifier manager."""
    global _classifier_manager

    if _classifier_manager is None:
        _classifier_manager = V3ClassifierManager()
        _classifier_manager.load_all()

    return _classifier_manager


def predict_stock(company_stone_id: str) -> Dict[str, Optional[Dict]]:
    """Predict base_color, design, tonality for a stock.

    Returns dict with predictions for each field, or None if no embedding.
    """
    embedding = v3_embeddings.load_stock_embedding(company_stone_id)

    if embedding is None:
        return {"base_color": None, "design": None, "tonality": None}

    manager = get_classifier_manager()
    predictions = manager.predict_all(embedding)

    return {
        field: pred.to_dict() if pred else None for field, pred in predictions.items()
    }
