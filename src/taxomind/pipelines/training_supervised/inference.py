"""
Inference utilities for trained hierarchical classification models using SetFit.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn.functional as F
from pathlib import Path
import pickle
from setfit import SetFitModel

logger = logging.getLogger(__name__)


class LevelModelPredictor:
    """Predictor for a single level SetFit model."""

    def __init__(
        self,
        model: SetFitModel,
        label2idx: Dict[str, int],
        idx2label: Dict[int, str],
        device: torch.device,
    ):
        self.model = model
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.device = device

        self.model.to(self.device)

    def predict(
        self, text: str, return_top_k: int = 5
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict label for a single text using SetFit's native methods.

        Args:
            text: Input text
            return_top_k: Number of top predictions to return

        Returns:
            Tuple of (predicted_label, confidence, top_k_predictions)
            where top_k_predictions is a list of (label, score) tuples
        """
        # Get probabilities using SetFit's predict_proba
        proba = self.model.predict_proba([text])[0]  # Get first (and only) result

        # Get top-k predictions
        top_indices = proba.argsort()[::-1][:return_top_k]
        top_predictions = [
            (self.idx2label[idx], float(proba[idx]))
            for idx in top_indices
        ]

        # Best prediction
        best_label = top_predictions[0][0]
        best_confidence = top_predictions[0][1]

        return best_label, best_confidence, top_predictions

    def predict_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """
        Predict labels for a batch of texts using SetFit's native batch prediction.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing (SetFit handles batching internally)

        Returns:
            List of (predicted_label, confidence) tuples
        """
        # SetFit's predict method returns class indices
        predictions = self.model.predict(texts)

        # Get probabilities for confidence scores
        probabilities = self.model.predict_proba(texts)

        results = []
        for pred_idx, proba in zip(predictions, probabilities):
            label = self.idx2label[int(pred_idx)]
            confidence = float(proba[pred_idx])
            results.append((label, confidence))

        return results


class HierarchicalModelManager:
    """Manager for all level models in hierarchical classification."""

    def __init__(self, models_dir: str, device: Optional[torch.device] = None):
        """
        Initialize the model manager.

        Args:
            models_dir: Directory containing trained models (e.g., data/06_models)
            device: Torch device to use (auto-detected if None)
        """
        self.models_dir = Path(models_dir)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.predictors: Dict[int, LevelModelPredictor] = {}

        logger.info(f"Initializing HierarchicalModelManager with device: {self.device}")

    def load_level_model(self, level: int) -> bool:
        """
        Load a SetFit model for a specific level.

        Args:
            level: Level ID (1, 2, 3, etc.)

        Returns:
            True if model was loaded successfully, False otherwise
        """
        model_path = self.models_dir / f"model_level_{level}.pkl"

        if not model_path.exists():
            logger.warning(f"Model for level {level} not found at {model_path}")
            return False

        try:
            logger.info(f"Loading SetFit model for level {level}")

            # Load model data
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            # Extract components
            model = model_data["model"]
            label2idx = model_data["label2idx"]
            idx2label = model_data["idx2label"]

            # Move model to device
            model.to(self.device)

            # Create predictor
            predictor = LevelModelPredictor(
                model=model,
                label2idx=label2idx,
                idx2label=idx2label,
                device=self.device,
            )

            self.predictors[level] = predictor
            logger.info(
                f"Successfully loaded level {level} model "
                f"with {len(label2idx)} classes"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load model for level {level}: {e}")
            return False

    def load_all_models(self) -> List[int]:
        """
        Load all available models.

        Returns:
            List of successfully loaded level IDs
        """
        loaded_levels = []

        for level in range(1, 6):  # Try levels 1-5
            if self.load_level_model(level):
                loaded_levels.append(level)

        logger.info(f"Loaded models for levels: {loaded_levels}")
        return loaded_levels

    def predict_level(
        self, text: str, level: int, return_top_k: int = 5
    ) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
        """
        Predict label for a specific level.

        Args:
            text: Input text
            level: Level ID to predict
            return_top_k: Number of top predictions to return

        Returns:
            Tuple of (predicted_label, confidence, top_k_predictions)
            or None if level model not available
        """
        if level not in self.predictors:
            logger.warning(f"Model for level {level} not loaded")
            return None

        return self.predictors[level].predict(text, return_top_k)

    def predict_hierarchical(
        self, text: str, return_top_k: int = 5
    ) -> Dict[int, Tuple[str, float, List[Tuple[str, float]]]]:
        """
        Predict labels for all loaded levels.

        Args:
            text: Input text
            return_top_k: Number of top predictions per level

        Returns:
            Dictionary mapping level ID to (predicted_label, confidence, top_k_predictions)
        """
        results = {}

        for level in sorted(self.predictors.keys()):
            prediction = self.predict_level(text, level, return_top_k)
            if prediction:
                results[level] = prediction

        return results

    def predict_batch_level(
        self, texts: List[str], level: int, batch_size: int = 32
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Predict labels for a batch of texts at a specific level.

        Args:
            texts: List of input texts
            level: Level ID to predict
            batch_size: Batch size for processing

        Returns:
            List of (predicted_label, confidence) tuples or None if level not available
        """
        if level not in self.predictors:
            logger.warning(f"Model for level {level} not loaded")
            return None

        return self.predictors[level].predict_batch(texts, batch_size)

    def get_available_levels(self) -> List[int]:
        """Get list of available (loaded) levels."""
        return sorted(self.predictors.keys())

    def get_level_classes(self, level: int) -> Optional[List[str]]:
        """Get list of classes for a specific level."""
        if level not in self.predictors:
            return None

        return list(self.predictors[level].label2idx.keys())


# Convenience functions for direct usage

def load_trained_level_models(
    models_dir: str = "data/06_models",
) -> HierarchicalModelManager:
    """
    Load all trained hierarchical SetFit models.

    Args:
        models_dir: Directory containing trained models

    Returns:
        HierarchicalModelManager with all available models loaded
    """
    manager = HierarchicalModelManager(models_dir)
    manager.load_all_models()
    return manager


def predict_level(
    text: str,
    level: int,
    models_dir: str = "data/06_models",
    manager: Optional[HierarchicalModelManager] = None,
) -> Optional[Tuple[str, float]]:
    """
    Quick prediction for a single text at a specific level.

    Args:
        text: Input text
        level: Level ID to predict
        models_dir: Directory containing trained models
        manager: Optional pre-loaded manager (for efficiency)

    Returns:
        Tuple of (predicted_label, confidence) or None if level not available
    """
    if manager is None:
        manager = HierarchicalModelManager(models_dir)
        manager.load_level_model(level)

    result = manager.predict_level(text, level, return_top_k=1)

    if result is None:
        return None

    predicted_label, confidence, _ = result
    return predicted_label, confidence
