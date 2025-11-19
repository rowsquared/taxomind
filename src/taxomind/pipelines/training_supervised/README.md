# Supervised Hierarchical Classification Pipeline

## Overview

This pipeline implements **supervised hierarchical classification training** using **ModernBERT** as the encoder. It trains separate classification models for each level of your taxonomy hierarchy when labeled examples are available.

### Key Features

- ✅ **ModernBERT-based** encoder for state-of-the-art text understanding
- ✅ **Level-specific models** - independent classifier for each hierarchy level
- ✅ **Automatic level selection** - only trains levels with sufficient data
- ✅ **GPU/CPU support** - automatic device detection with fallback
- ✅ **Comprehensive metrics** - accuracy, F1, precision, and recall tracking
- ✅ **Production-ready inference** - easy-to-use prediction interface
- ✅ **Graceful degradation** - skips training if no labeled data available

---

## Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: taxonomy_enriched + labeled_dataset                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  prepare_training_data                                       │
│  • Derives parent labels for each level from taxonomy       │
│  • Filters levels with < min_samples_per_level              │
│  • Outputs: training_level_1, ..., training_level_5         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Level-Specific Training (parallel)                         │
│                                                               │
│  train_level_1 → model_level_1                              │
│  train_level_2 → model_level_2                              │
│  train_level_3 → model_level_3                              │
│  train_level_4 → model_level_4                              │
│  train_level_5 → model_level_5 (optional)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Trained models in data/06_models/                  │
│  • model_level_*.pkl (model + tokenizer + label mappings)  │
│  • Training metrics and history                             │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture

```
Input Text
    ↓
┌──────────────────────┐
│  ModernBERT Encoder  │  ← Pre-trained, fine-tuned
│  (answerdotai/       │
│   ModernBERT-base)   │
└──────────────────────┘
    ↓
[CLS] Token Representation
    ↓
┌──────────────────────┐
│  Dropout (0.1)       │
└──────────────────────┘
    ↓
┌──────────────────────┐
│  Linear Classifier   │  ← Level-specific
│  (hidden_size →      │     classification head
│   num_classes)       │
└──────────────────────┘
    ↓
Logits → Softmax → Predictions
```

---

## Installation & Setup

### 1. Install Dependencies

Add to your `requirements.txt`:

```txt
transformers>=4.36.0
torch>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.66.0
```

Install:

```bash
pip install -r requirements.txt
```

### 2. Register the Pipeline

In your `src/taxomind/pipeline_registry.py`:

```python
from taxomind.pipelines import training_supervised

def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "taxonomy": taxonomy.create_pipeline(),
        "training_supervised": training_supervised.create_pipeline(),
        # ... other pipelines
    }
```

### 3. Configure Parameters

Copy the provided configuration:

```bash
# Already created at:
# conf/base/parameters/supervised.yml
```

Key parameters to adjust:

- `model_name`: ModernBERT model variant
- `min_samples_per_level`: Minimum samples to train a level (default: 10)
- `batch_size`: Training batch size (default: 16)
- `epochs`: Number of training epochs (default: 3)
- `learning_rate`: Learning rate (default: 2e-5)

### 4. Add Catalog Entries

Merge `conf/base/catalog_supervised.yml` into your `conf/base/catalog.yml`:

```bash
cat conf/base/catalog_supervised.yml >> conf/base/catalog.yml
```

Or manually add the entries for:
- `labeled_dataset`
- `training_level_1` through `training_level_5`
- `model_level_1` through `model_level_5`

---

## Data Preparation

### Input Data Format

Your `labeled_dataset.csv` should have:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Input text to classify |
| `leaf_code` | string | Final hierarchical code (e.g., "2512.1.3") |

**Example:**

```csv
text,leaf_code
"Software engineer developing web applications","2512.1"
"Medical doctor specializing in cardiology","2212.3"
"Elementary school teacher","2341.1"
```

### Hierarchical Code Format

The pipeline expects codes in hierarchical format where levels are separated by dots:

- Level 1: `"2"`
- Level 2: `"2.5"`
- Level 3: `"2.5.1"`
- Level 4: `"2.5.1.2"`
- Level 5: `"2.5.1.2.1"`

The pipeline automatically derives parent codes from the leaf code using your taxonomy.

---

## Running the Pipeline

### Full Training Pipeline

```bash
kedro run --pipeline=training_supervised
```

This will:
1. Prepare training data for each level
2. Train models for levels with sufficient data
3. Save trained models to `data/06_models/`
4. Generate training metrics

### Training Specific Levels

```bash
# Train only level 1
kedro run --pipeline=training_supervised --to-nodes=train_level_1

# Train levels 1 and 2
kedro run --pipeline=training_supervised --to-nodes=train_level_2
```

### Monitoring Training

Training progress is logged with:
- Epoch-by-epoch metrics (loss, accuracy, F1)
- Training and validation performance
- Best model selection based on validation F1

---

## Using Trained Models

### Basic Inference

```python
from taxomind.pipelines.training_supervised.inference import (
    load_trained_level_models,
    predict_level,
)

# Option 1: Quick single prediction
label, confidence = predict_level(
    text="Software engineer developing AI applications",
    level=3,
    models_dir="data/06_models"
)
print(f"Predicted: {label} (confidence: {confidence:.2%})")

# Option 2: Load all models once (more efficient)
manager = load_trained_level_models("data/06_models")

# Predict for specific level
label, conf, top_5 = manager.predict_level(
    text="Medical doctor in hospital",
    level=2,
    return_top_k=5
)

print(f"Top prediction: {label} ({conf:.2%})")
print("Top 5 predictions:")
for pred_label, pred_conf in top_5:
    print(f"  {pred_label}: {pred_conf:.2%}")
```

### Hierarchical Prediction (All Levels)

```python
# Predict all levels at once
results = manager.predict_hierarchical(
    text="Elementary school teacher",
    return_top_k=3
)

for level, (label, conf, top_k) in results.items():
    print(f"Level {level}: {label} ({conf:.2%})")
```

### Batch Prediction

```python
texts = [
    "Software engineer",
    "Medical doctor",
    "School teacher",
    # ... more texts
]

# Batch prediction for level 3
predictions = manager.predict_batch_level(
    texts=texts,
    level=3,
    batch_size=32
)

for text, (label, conf) in zip(texts, predictions):
    print(f"{text[:30]:30s} → {label} ({conf:.2%})")
```

### Integration with Zero-Shot Pipeline

```python
from taxomind.pipelines.training_supervised.inference import (
    HierarchicalModelManager,
)

# Load supervised models
supervised_manager = HierarchicalModelManager("data/06_models")
supervised_manager.load_all_models()

# Check which levels are available
available_levels = supervised_manager.get_available_levels()
print(f"Supervised models available for levels: {available_levels}")

def hybrid_predict(text: str, level: int):
    """Combine supervised and zero-shot predictions."""

    # Try supervised first
    supervised_result = supervised_manager.predict_level(text, level)

    if supervised_result:
        label, confidence, _ = supervised_result

        # Use supervised prediction if confidence is high
        if confidence > 0.7:
            return label, confidence, "supervised"

    # Fallback to zero-shot
    # ... your zero-shot prediction logic here ...

    return label, confidence, "zero_shot"
```

---

## Model Storage

### File Structure

```
data/06_models/
├── model_level_1.pkl    # Level 1 model + metadata
├── model_level_2.pkl    # Level 2 model + metadata
├── model_level_3.pkl    # Level 3 model + metadata
├── model_level_4.pkl    # Level 4 model + metadata
└── model_level_5.pkl    # Level 5 model + metadata (if exists)
```

### Model Contents

Each `.pkl` file contains:

```python
{
    "model_state_dict": {...},           # PyTorch model weights
    "model_config": {
        "model_name": "answerdotai/ModernBERT-base",
        "num_classes": 42,
        "dropout": 0.1,
    },
    "tokenizer_name": "answerdotai/ModernBERT-base",
    "label2idx": {"2.5.1": 0, "2.5.2": 1, ...},
    "idx2label": {0: "2.5.1", 1: "2.5.2", ...},
    "metrics": {
        "val_accuracy": 0.89,
        "val_f1": 0.87,
        "val_precision": 0.88,
        "val_recall": 0.86,
    },
    "training_history": [...],           # Full training history
    "best_f1": 0.87,                     # Best F1 score achieved
}
```

---

## Configuration Reference

### Complete `supervised.yml`

```yaml
supervised:
  # Model configuration
  model_name: "answerdotai/ModernBERT-base"

  # Data preparation
  min_samples_per_level: 10

  # Training hyperparameters
  training:
    batch_size: 16          # Adjust based on GPU memory
    epochs: 3               # More epochs for larger datasets
    learning_rate: 2.0e-5   # Standard for BERT fine-tuning
    max_length: 512         # Maximum sequence length
    test_size: 0.2          # Validation split
    dropout: 0.1            # Dropout in classifier head
    warmup_ratio: 0.1       # Warmup steps ratio

  # Inference settings
  inference:
    batch_size: 32
    top_k_predictions: 5
    confidence_threshold: 0.5
```

### Tuning Guidelines

**For small datasets (< 1000 samples):**
- Increase `min_samples_per_level` to 20-30
- Use smaller `batch_size` (8-12)
- Add more regularization (`dropout: 0.2`)

**For large datasets (> 10,000 samples):**
- Decrease `min_samples_per_level` to 5
- Increase `batch_size` (32-64 if GPU allows)
- Train for more `epochs` (5-10)

**For GPU memory issues:**
- Reduce `batch_size`
- Reduce `max_length` to 256 or 128
- Use `answerdotai/ModernBERT-base` instead of large variants

---

## Performance Metrics

The pipeline tracks:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 (accounts for class imbalance)
- **Precision**: Weighted precision
- **Recall**: Weighted recall

Metrics are computed for both training and validation sets at each epoch.

### Accessing Metrics

```python
import pickle

# Load a trained model
with open("data/06_models/model_level_3.pkl", "rb") as f:
    model_data = pickle.load(f)

# Print final metrics
print("Validation Metrics:")
print(f"  Accuracy:  {model_data['metrics']['val_accuracy']:.2%}")
print(f"  F1 Score:  {model_data['metrics']['val_f1']:.2%}")
print(f"  Precision: {model_data['metrics']['val_precision']:.2%}")
print(f"  Recall:    {model_data['metrics']['val_recall']:.2%}")

# Print training history
for epoch_metrics in model_data['training_history']:
    print(f"Epoch {epoch_metrics['epoch']}: "
          f"Val F1 = {epoch_metrics['val_f1']:.4f}")
```

---

## Graceful Handling of Missing Data

### Pipeline Behavior

The pipeline automatically handles missing or insufficient data:

1. **No labeled data**: Pipeline completes without training
2. **Insufficient level data**: Skips that level, trains others
3. **Missing taxonomy levels**: Only trains available levels

### Checking Available Models

```python
from pathlib import Path

models_dir = Path("data/06_models")

for level in range(1, 6):
    model_path = models_dir / f"model_level_{level}.pkl"
    if model_path.exists():
        print(f"✓ Level {level} model available")
    else:
        print(f"✗ Level {level} model not found")
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce `batch_size` in `supervised.yml`
2. Reduce `max_length` to 256 or 128
3. Use CPU: `device: "cpu"` in config
4. Use smaller ModernBERT variant

### Issue: "No models trained"

**Check:**
1. `labeled_dataset.csv` exists and has correct format
2. Dataset has at least `min_samples_per_level` samples
3. `leaf_code` values match taxonomy codes
4. Run with `--log-level=DEBUG` to see detailed filtering

### Issue: "Poor model performance"

**Improvements:**
1. Increase training data (aim for 100+ samples per class)
2. Train for more epochs
3. Adjust `learning_rate` (try 1e-5 or 3e-5)
4. Increase `max_length` if texts are long
5. Balance class distribution in training data

### Issue: "Training is very slow"

**Optimizations:**
1. Use GPU: ensure PyTorch CUDA is installed
2. Increase `batch_size` (if GPU memory allows)
3. Reduce `max_length`
4. Use DataLoader with multiple workers (requires code change)

---

## Comparison with Zero-Shot Pipeline

| Feature | Zero-Shot (BGE) | Supervised (ModernBERT) |
|---------|-----------------|-------------------------|
| **Training Data** | Not required | Required |
| **Setup Time** | Immediate | Hours (training) |
| **Accuracy** | Good for general cases | Better for domain-specific |
| **Adaptability** | Works on any taxonomy | Specific to trained taxonomy |
| **New Categories** | Handles instantly | Requires retraining |
| **Use Case** | Exploration, prototyping | Production, optimized accuracy |

### When to Use Each

**Use Zero-Shot when:**
- You have no labeled data
- Taxonomy changes frequently
- Need immediate results
- Exploring new domains

**Use Supervised when:**
- You have 100+ labeled examples per level
- Taxonomy is stable
- Accuracy is critical
- Domain-specific language is complex

**Hybrid Approach (Recommended):**
```python
def predict_with_fallback(text, level):
    # Try supervised first
    if supervised_available(level):
        pred = supervised_predict(text, level)
        if pred.confidence > 0.7:
            return pred

    # Fallback to zero-shot
    return zero_shot_predict(text, level)
```

---

## Advanced Usage

### Custom Training Loop

For advanced users who want to customize training:

```python
from taxomind.pipelines.training_supervised.nodes import (
    train_level_model,
    prepare_training_data,
)
import pandas as pd

# Prepare your data
taxonomy = pd.read_csv("data/01_raw/taxonomy.csv")
labeled_data = pd.read_csv("data/01_raw/labeled_dataset.csv")

training_data = prepare_training_data(
    taxonomy=taxonomy,
    labeled_dataset=labeled_data,
    min_samples_per_level=10,
)

# Train with custom parameters
custom_params = {
    "batch_size": 8,
    "epochs": 5,
    "learning_rate": 1e-5,
    "max_length": 256,
    "test_size": 0.15,
    "dropout": 0.2,
    "warmup_ratio": 0.05,
}

model_data = train_level_model(
    level_data=training_data["training_level_3"],
    level_id=3,
    model_name="answerdotai/ModernBERT-base",
    parameters=custom_params,
)

# Save manually
import pickle
with open("custom_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
```

### Multi-GPU Training

For distributed training, modify `nodes.py`:

```python
# In train_level_model function
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    logger.info(f"Using {torch.cuda.device_count()} GPUs")
```

---

## Contributing

To extend this pipeline:

1. **Add new model architectures**: Modify `ModernBERTClassifier` in `nodes.py`
2. **Add evaluation metrics**: Extend metric tracking in `train_level_model`
3. **Add data augmentation**: Modify `HierarchicalClassificationDataset`
4. **Add early stopping**: Extend training loop with patience parameter

---

## License & Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{taxomind_supervised,
  title = {TaxoMind Supervised Hierarchical Classification Pipeline},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/taxomind}
}
```

---

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check Kedro documentation: https://kedro.org
- Check ModernBERT: https://huggingface.co/answerdotai/ModernBERT-base

---

**Happy Training! 🚀**
