# Module 2 - Inference Pipeline

## Overview

This pipeline implements the runtime inference system for hierarchical classification over deep taxonomies. It follows the approved specification for Module 2 with label-based retrieval, multi-view scoring, and asymmetric stopping.

## Architecture

### Key Design Principles

1. **Label-based retrieval**: Uses stable label embeddings (NOT multi-view) for initial candidate recall
2. **On-demand computation**: `embedding_effective` computed during scoring (not materialized)
3. **Asymmetric stopping**: Parent veto prevents over-specification
4. **Sibling expansion**: Ensures complete local context for ambiguity assessment
5. **Explainability**: Returns stopping reason, alternatives, and routing trace

### Terminology

- `embedding_effective = (1-β) * embedding_label + β * evidence_centroid`
- NOT `E_emp` (deprecated terminology)

## Pipeline Nodes

### Setup Phase (Once per Session)

1. **load_taxonomy_index**: Load pre-computed taxonomy index from Module 1
2. **load_taxonomy_graph**: Build parent-child adjacency dictionary
3. **build_retrieval_index**: Create label-based index for fast cosine similarity
4. **prepare_scoring_views**: Prepare handles for on-demand `embedding_effective` computation

### Query Phase (Per Query)

5. **embed_query**: Embed input text using same model as taxonomy preparation
6. **retrieve_candidates**: Label-based retrieval + sibling expansion
7. **route_query**: Top-down routing with asymmetric stopping
8. **format_predictions**: Format output with explainability

## Configuration

### Parameters (`conf/base/parameters.yml`)

```yaml
inference:
  # Retrieval
  retrieval_k: 10  # Number of initial candidates

  # Routing (asymmetric stopping)
  min_descent_gap: 0.05  # Sibling separation threshold
  parent_veto_margin: 0.05  # Parent competitiveness margin

  # Multi-view scoring
  beta: 0.0  # Evidence weight (0.0 = pure label, 1.0 = pure evidence)

  # Optional constraints
  max_depth: null  # Maximum depth to descend (null = no limit)
```

### Datasets (`conf/base/catalog.yml`)

**Inputs:**
- `taxonomy_index`: Partitioned dataset with taxonomy embeddings (from Module 1)
- `embedding_model`: Pre-loaded SentenceTransformer model
- `inference_query_text`: Input text to classify (MemoryDataset)

**Outputs:**
- `inference_taxonomy_df`: Loaded taxonomy DataFrame
- `inference_taxonomy_graph`: Parent-child adjacency
- `inference_retrieval_index`: Label-based index
- `inference_scoring_views`: Multi-view handles
- `inference_query_embedding`: Query embedding vector
- `inference_candidates`: Retrieved candidates
- `inference_routing_result`: Routing decisions
- `inference_prediction`: Final formatted prediction

## Usage

### Single Query Inference

```python
from kedro.framework.session import KedroSession

with KedroSession.create() as session:
    # Set query text
    context = session.load_context()
    context.catalog.save("inference_query_text", "I work as a software developer")

    # Run inference pipeline
    session.run(pipeline_name="inference")

    # Get prediction
    prediction = context.catalog.load("inference_prediction")
    print(prediction)
```

### Expected Output Format

```python
{
    "query": "I work as a software developer",
    "prediction": {
        "code": "2512",
        "label": "Software developers",
        "level": 4,
        "score": 0.856
    },
    "ambiguous": False,
    "alternatives": [],
    "stopping_reason": "leaf_node_reached",
    "path": ["2", "25", "251", "2512"],
    "routing_trace": [
        {
            "level": 1,
            "candidates": ["1", "2", "3", ...],
            "scores": {"1": 0.412, "2": 0.823, ...},
            "selected": "2",
            "stopped": False,
            "reason": "descending"
        },
        ...
    ]
}
```

### Ambiguous Prediction Example

```python
{
    "query": "I teach students",
    "prediction": {
        "code": "23",
        "label": "Teaching professionals",
        "level": 2,
        "score": 0.752
    },
    "ambiguous": True,
    "alternatives": [
        {"code": "231", "label": "University and higher education teachers", "score": 0.748},
        {"code": "232", "label": "Vocational education teachers", "score": 0.745},
        {"code": "233", "label": "Secondary education teachers", "score": 0.743}
    ],
    "stopping_reason": "sibling_near_tie (gap=0.003 < 0.05)",
    "path": ["2", "23"],
    "routing_trace": [...]
}
```

## Stopping Criteria (Asymmetric)

The routing logic uses asymmetric stopping criteria to prevent both over-specification and under-specification:

### 1. Sibling Separation (Drives Descent)

```
If max(sibling_scores) - second_max(sibling_scores) < min_descent_gap:
    STOP at parent (siblings are near-tied)
```

**Purpose**: Prevents forcing arbitrary choices among near-tied siblings (Education problem)

### 2. Parent Veto (Prevents Over-Specification)

```
If parent_score + parent_veto_margin >= best_child_score:
    STOP at parent (parent is competitive)
```

**Purpose**: Prevents descending when parent is semantically competitive

### 3. Leaf Node Reached

```
If node has no children:
    STOP (natural terminus)
```

### 4. Max Depth (Optional)

```
If current_level >= max_depth:
    STOP (configured constraint)
```

## Failure Mode Prevention

### Over-Specification (Education Problem)

**Scenario**: Query "I work in education" forced to choose between "University teachers", "Secondary teachers", "Primary teachers"

**Prevention**:
- Parent veto: If "Education" is competitive, stop there
- Sibling near-tie: If siblings have similar scores, stop at parent

**Parameters**:
- `min_descent_gap`: Higher values = more conservative (stop earlier)
- `parent_veto_margin`: Higher values = more parent preference

### Under-Specification

**Scenario**: Query "I work as a PHP developer" stopping too early at "Professionals"

**Prevention**:
- Sibling separation: Only descend if clear winner exists
- Retrieval expansion: Include all siblings for complete local context

**Parameters**:
- `min_descent_gap`: Lower values = more aggressive descent

### Top-Down Error Propagation

**Scenario**: Wrong branch at level 1 locks out correct nodes at deeper levels

**Prevention**:
- Sibling expansion: Retrieves across branches
- Candidate set: Not limited to single descent path

**Implementation**: `retrieve_candidates` expands to include siblings

### Semantic Dilution

**Scenario**: Broad-topic evidence dominates narrow coherent branches

**Prevention**:
- Beta default: `beta=0.0` uses pure label embeddings
- Evidence weight: Gradually increase beta as corrections accumulate

**Parameters**:
- `beta`: Start at 0.0, increase to 0.3-0.5 with >10 corrections per node

## Multi-View Scoring

### On-Demand Computation

`embedding_effective` is computed on-demand during scoring (NOT materialized):

```python
def compute_effective_embedding(code, taxonomy_df, code_to_idx, beta):
    embedding_label = taxonomy_df.loc[code, "embedding_label"]
    evidence_centroid = taxonomy_df.loc[code, "evidence_centroid"]

    if evidence_centroid is None or beta == 0.0:
        return embedding_label
    else:
        embedding_eff = (1 - beta) * embedding_label + beta * evidence_centroid
        return embedding_eff / np.linalg.norm(embedding_eff)
```

### Why On-Demand?

1. **Real-time updates**: Evidence centroids updated by Module 3
2. **Memory efficiency**: No redundant storage
3. **Flexible beta**: Can adjust per inference call

## Evidence State

Evidence columns in taxonomy_index (initialized by Module 1):

- `evidence_centroid`: None (will store np.ndarray after corrections)
- `evidence_count`: 0 (incremented by Module 3)
- `evidence_last_updated`: None (timestamp, optional)

These are updated by Module 3 (Incremental Learning), NOT by Module 2.

## Integration with Other Modules

### Module 1 (Build Taxonomy)

**Inputs**:
- `taxonomy_index`: Partitioned dataset with:
  - Taxonomy structure (code, level, parentCode, label)
  - Embedding views (embedding_label, embedding_definition, embedding_examples)
  - Evidence state (evidence_centroid, evidence_count)

**Dependency**: Must run `build_taxonomy` pipeline before `inference`

### Module 3 (Incremental Learning)

**Outputs used by Module 3**:
- `inference_prediction`: Used for correction comparison
- Evidence state columns: Updated by Module 3 based on corrections

**Integration**: Module 3 updates `evidence_centroid` and `evidence_count` in taxonomy_index

## Testing

### Smoke Test

```bash
# Run inference pipeline
kedro run --pipeline=inference
```

### Integration Test with Module 1

```bash
# Build taxonomy first
kedro run --pipeline=build_taxonomy

# Then run inference
kedro run --pipeline=inference
```

### Evaluation (Future)

Batch inference pipeline (not yet implemented):

```python
# Load validation queries
queries_df = pd.DataFrame({
    "text": ["software developer", "teacher", "shop manager"],
    "ground_truth": ["2512", "23", "5221"]
})

# Run batch inference
context.catalog.save("inference_queries_df", queries_df)
session.run(pipeline_name="inference_batch")

# Get predictions
predictions_df = context.catalog.load("inference_predictions_df")
```

## Spec Compliance

This implementation follows the approved specification:

- **Terminology**: Uses `embedding_effective` (not `E_emp`)
- **Evidence columns**: `evidence_centroid`, `evidence_count`, `evidence_last_updated`
- **No evidence_source column**: Not included (per spec)
- **Label-based retrieval**: Stable, fast, consistent recall
- **On-demand computation**: No materialization of `embedding_effective`
- **Asymmetric stopping**: Parent veto prevents over-specification
- **Sibling expansion**: Complete local context
- **Explainability**: Stopping reason, alternatives, routing trace

## Known Limitations

1. **Batch inference**: Not yet implemented (use single-query pipeline)
2. **Multi-view retrieval**: Not implemented (uses label-only per spec)
3. **Hierarchical evidence**: Ancestor evidence used only as secondary signal
4. **Beta tuning**: Manual parameter adjustment (no auto-tuning)

## Next Steps

1. Implement batch inference pipeline for evaluation
2. Add comprehensive unit tests
3. Implement Module 3 (Incremental Learning) for evidence updates
4. Add beta auto-tuning based on evidence_count
5. Add retrieval metrics (recall@k, MRR)
