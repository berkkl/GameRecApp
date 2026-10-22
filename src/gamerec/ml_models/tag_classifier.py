"""
Multi-label tag classifier using sklearn.
Trained on Steam game descriptions to predict tags.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from ..config import settings
from ..text.embedder import Embedder


class TagClassifier:
    """
    Multi-label classifier for predicting game tags from text.
    Uses pre-trained embeddings + OneVsRestClassifier with LogisticRegression.
    """

    DEFAULT_MODEL_PATH = settings.DATA_DIR / "models/tag_classifier.pkl"

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.embedder = Embedder()
        self.classifier: Optional[OneVsRestClassifier] = None
        self.mlb: Optional[MultiLabelBinarizer] = None
        self._is_loaded = False

    def train(
        self,
        texts: List[str],
        labels: List[List[str]],
        test_size: float = 0.2,
        groups: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """
        Train the classifier on (text, [tags]) pairs.

        Args:
            texts: List of game descriptions
            labels: List of tag lists for each game
            test_size: Fraction for test split

        Returns:
            Dict with training metrics (f1_score, accuracy)
        """
        from sklearn.metrics import f1_score
        from sklearn.model_selection import GroupShuffleSplit

        if len(texts) != len(labels) or not texts:
            raise ValueError(
                "Training texts and labels must have matching, nonzero lengths."
            )
        if groups is None:
            raise ValueError(
                "Supply game groups to prevent training/evaluation leakage."
            )
        if len(groups) != len(texts) or len(set(groups)) < 2:
            raise ValueError("Training requires at least two distinct game groups.")

        print(f"Training on {len(texts)} samples...")

        # 1. Fit multi-label binarizer on all labels
        self.mlb = MultiLabelBinarizer()
        y_all = self.mlb.fit_transform(labels)
        print(f"Found {len(self.mlb.classes_)} unique tags")

        # 2. Generate embeddings
        print("Generating embeddings...")
        X_all = self.embedder.embed(texts)

        # 3. Train/test split
        train_indices, test_indices = next(
            GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42).split(
                X_all, y_all, groups
            )
        )
        X_train, X_test = X_all[train_indices], X_all[test_indices]
        y_train, y_test = y_all[train_indices], y_all[test_indices]

        # 4. Train classifier
        print("Training classifier...")
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            ),
            n_jobs=1,
        )
        self.classifier.fit(X_train, y_train)

        # 5. Evaluate
        y_pred = self.classifier.predict(X_test)

        metrics = {
            "f1_micro": float(f1_score(y_test, y_pred, average="micro")),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "f1_samples": float(f1_score(y_test, y_pred, average="samples")),
            "n_samples": len(texts),
            "n_tags": len(self.mlb.classes_),
            "train_games": len(set(groups[i] for i in train_indices)),
            "test_games": len(set(groups[i] for i in test_indices)),
        }

        print(f"F1 (micro): {metrics['f1_micro']:.3f}")
        print(f"F1 (macro): {metrics['f1_macro']:.3f}")

        self._is_loaded = True
        return metrics

    def save(self, path: Optional[Path] = None) -> None:
        """Save trained model to disk."""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "classifier": self.classifier,
            "mlb": self.mlb,
            "version": "1.0",
            "embedding_model": self.embedder.model_name,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Model saved to {save_path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """Load trained model from disk."""
        load_path = path or self.model_path
        self._is_loaded = False
        self.classifier = None
        self.mlb = None

        if not load_path.exists():
            print(f"Model not found at {load_path}")
            return False

        with open(load_path, "rb") as f:
            data = pickle.load(f)
        model_name = data.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        if model_name != self.embedder.model_name:
            return False

        self.classifier = data["classifier"]
        self.mlb = data["mlb"]
        self._is_loaded = True

        print(f"Model loaded from {load_path} ({len(self.mlb.classes_)} tags)")
        return True

    def predict(
        self, text: str, top_k: int = 10, threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Predict tags for a given text.

        Args:
            text: Game description or user prompt
            top_k: Maximum number of tags to return
            threshold: Minimum probability threshold

        Returns:
            List of dicts with 'name' and 'score' keys
        """
        if not self._is_loaded:
            if not self.load():
                return []

        # Get embedding
        embedding = self.embedder.embed([text])[0].reshape(1, -1)

        # Get probabilities for each tag
        # OneVsRest with LogisticRegression supports predict_proba
        probs = self.classifier.predict_proba(embedding)[0]

        # Get top-k tags above threshold
        results = []
        sorted_indices = np.argsort(probs)[::-1]

        for idx in sorted_indices[:top_k]:
            prob = probs[idx]
            if prob >= threshold:
                results.append({"name": self.mlb.classes_[idx], "score": float(prob)})

        return results

    def predict_batch(
        self, texts: List[str], top_k: int = 10, threshold: float = 0.3
    ) -> List[List[Dict[str, Any]]]:
        """Predict tags for multiple texts efficiently."""
        if not self._is_loaded:
            if not self.load():
                return [[] for _ in texts]

        embeddings = self.embedder.embed(texts)
        all_probs = self.classifier.predict_proba(embeddings)

        results = []
        for probs in all_probs:
            tags = []
            sorted_indices = np.argsort(probs)[::-1]
            for idx in sorted_indices[:top_k]:
                prob = probs[idx]
                if prob >= threshold:
                    tags.append({"name": self.mlb.classes_[idx], "score": float(prob)})
            results.append(tags)

        return results

    @property
    def is_available(self) -> bool:
        """Check if model is loaded or can be loaded."""
        if self._is_loaded:
            return True
        return self.load()

    @property
    def tag_classes(self) -> List[str]:
        """Return list of known tag classes."""
        if self.mlb is None:
            return []
        return list(self.mlb.classes_)


# Module-level singleton for easy access
_classifier_instance: Optional[TagClassifier] = None


def get_classifier() -> TagClassifier:
    """Get or create the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TagClassifier()
    return _classifier_instance
