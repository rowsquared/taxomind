"""
Example script demonstrating supervised hierarchical classification pipeline.

This script shows:
1. How to run the training pipeline
2. How to load and use trained models for inference
3. How to combine supervised and zero-shot predictions
"""

import logging
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_run_training():
    """Example 1: Run the training pipeline using Kedro."""
    logger.info("=" * 60)
    logger.info("Example 1: Running Training Pipeline")
    logger.info("=" * 60)

    # This would normally be run via CLI:
    # kedro run --pipeline=training_supervised

    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    # Bootstrap project
    project_path = Path(__file__).parent.parent
    bootstrap_project(project_path)

    # Run pipeline
    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name="training_supervised")

    logger.info("Training completed! Models saved to data/06_models/")


def example_2_single_prediction():
    """Example 2: Make a single prediction with a trained model."""
    logger.info("=" * 60)
    logger.info("Example 2: Single Prediction")
    logger.info("=" * 60)

    from taxomind.pipelines.training_supervised.inference import predict_level

    text = "Software engineer developing web applications using Python and Django"

    # Predict at level 3
    result = predict_level(
        text=text,
        level=3,
        models_dir="data/06_models"
    )

    if result:
        label, confidence = result
        logger.info(f"Input: {text}")
        logger.info(f"Predicted Level 3 Code: {label}")
        logger.info(f"Confidence: {confidence:.2%}")
    else:
        logger.warning("Model for level 3 not found")


def example_3_hierarchical_prediction():
    """Example 3: Predict all levels for a text."""
    logger.info("=" * 60)
    logger.info("Example 3: Hierarchical Prediction (All Levels)")
    logger.info("=" * 60)

    from taxomind.pipelines.training_supervised.inference import (
        load_trained_level_models
    )

    # Load all available models
    manager = load_trained_level_models("data/06_models")

    text = "Medical doctor specializing in cardiology at a hospital"

    logger.info(f"Input: {text}\n")

    # Predict all levels
    results = manager.predict_hierarchical(text, return_top_k=3)

    for level, (label, confidence, top_k) in results.items():
        logger.info(f"Level {level}: {label} ({confidence:.2%})")
        logger.info("  Top 3 predictions:")
        for pred_label, pred_conf in top_k[:3]:
            logger.info(f"    - {pred_label}: {pred_conf:.2%}")
        logger.info("")


def example_4_batch_prediction():
    """Example 4: Batch prediction for multiple texts."""
    logger.info("=" * 60)
    logger.info("Example 4: Batch Prediction")
    logger.info("=" * 60)

    from taxomind.pipelines.training_supervised.inference import (
        load_trained_level_models
    )

    # Sample texts
    texts = [
        "Software engineer developing mobile applications",
        "Elementary school teacher",
        "Registered nurse in intensive care unit",
        "Data scientist analyzing business metrics",
        "Construction worker building houses",
    ]

    # Load models
    manager = load_trained_level_models("data/06_models")

    # Batch predict at level 2
    if 2 in manager.get_available_levels():
        predictions = manager.predict_batch_level(
            texts=texts,
            level=2,
            batch_size=32
        )

        logger.info("Predictions for Level 2:\n")
        for text, (label, conf) in zip(texts, predictions):
            logger.info(f"{text[:50]:50s} → {label} ({conf:.2%})")
    else:
        logger.warning("Level 2 model not available")


def example_5_hybrid_prediction():
    """Example 5: Combine supervised and zero-shot predictions."""
    logger.info("=" * 60)
    logger.info("Example 5: Hybrid Prediction (Supervised + Zero-Shot)")
    logger.info("=" * 60)

    from taxomind.pipelines.training_supervised.inference import (
        HierarchicalModelManager
    )

    # Load supervised models
    supervised_manager = HierarchicalModelManager("data/06_models")
    supervised_manager.load_all_models()

    def hybrid_predict(text: str, level: int, confidence_threshold: float = 0.7):
        """
        Hybrid prediction using supervised model with zero-shot fallback.

        Args:
            text: Input text
            level: Hierarchy level to predict
            confidence_threshold: Minimum confidence to use supervised prediction

        Returns:
            Tuple of (label, confidence, source)
        """
        # Try supervised first
        if level in supervised_manager.get_available_levels():
            result = supervised_manager.predict_level(text, level)

            if result:
                label, confidence, _ = result

                if confidence >= confidence_threshold:
                    return label, confidence, "supervised"

                logger.info(
                    f"Supervised confidence ({confidence:.2%}) below threshold "
                    f"({confidence_threshold:.2%}), falling back to zero-shot"
                )
        else:
            logger.info(f"No supervised model for level {level}, using zero-shot")

        # Fallback to zero-shot
        # In a real implementation, you would call your zero-shot pipeline here
        # For this example, we'll just return a placeholder
        return "zero-shot-prediction", 0.6, "zero_shot"

    # Test hybrid prediction
    test_text = "Software engineer specializing in machine learning"

    for level in [1, 2, 3]:
        label, conf, source = hybrid_predict(test_text, level)
        logger.info(
            f"Level {level}: {label} ({conf:.2%}) [source: {source}]"
        )


def example_6_model_inspection():
    """Example 6: Inspect trained model details."""
    logger.info("=" * 60)
    logger.info("Example 6: Model Inspection")
    logger.info("=" * 60)

    import pickle

    model_path = Path("data/06_models/model_level_2.pkl")

    if model_path.exists():
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        logger.info("Model Configuration:")
        logger.info(f"  Model Name: {model_data['model_config']['model_name']}")
        logger.info(f"  Number of Classes: {model_data['model_config']['num_classes']}")
        logger.info(f"  Dropout: {model_data['model_config']['dropout']}")

        logger.info("\nValidation Metrics:")
        metrics = model_data['metrics']
        logger.info(f"  Accuracy:  {metrics['val_accuracy']:.2%}")
        logger.info(f"  F1 Score:  {metrics['val_f1']:.2%}")
        logger.info(f"  Precision: {metrics['val_precision']:.2%}")
        logger.info(f"  Recall:    {metrics['val_recall']:.2%}")

        logger.info(f"\nBest F1 Score: {model_data['best_f1']:.2%}")

        logger.info("\nClasses:")
        for idx, label in sorted(model_data['idx2label'].items())[:10]:
            logger.info(f"  {idx}: {label}")
        if len(model_data['idx2label']) > 10:
            logger.info(f"  ... and {len(model_data['idx2label']) - 10} more classes")

        logger.info("\nTraining History:")
        for epoch_data in model_data['training_history']:
            logger.info(
                f"  Epoch {epoch_data['epoch']}: "
                f"Loss={epoch_data['val_loss']:.4f}, "
                f"F1={epoch_data['val_f1']:.4f}"
            )
    else:
        logger.warning(f"Model not found at {model_path}")


def example_7_check_available_models():
    """Example 7: Check which models are available."""
    logger.info("=" * 60)
    logger.info("Example 7: Check Available Models")
    logger.info("=" * 60)

    from taxomind.pipelines.training_supervised.inference import (
        load_trained_level_models
    )

    manager = load_trained_level_models("data/06_models")

    available_levels = manager.get_available_levels()

    logger.info(f"Available levels: {available_levels}\n")

    for level in available_levels:
        classes = manager.get_level_classes(level)
        logger.info(f"Level {level}: {len(classes)} classes")
        logger.info(f"  Sample classes: {classes[:5]}")
        logger.info("")


if __name__ == "__main__":
    # Run examples
    # Note: Comment out examples that require trained models if you haven't trained yet

    # Training example (requires labeled data)
    # example_1_run_training()

    # Inference examples (require trained models)
    # Uncomment after training:

    # example_2_single_prediction()
    # example_3_hierarchical_prediction()
    # example_4_batch_prediction()
    # example_5_hybrid_prediction()
    # example_6_model_inspection()
    example_7_check_available_models()

    logger.info("\n" + "=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)
