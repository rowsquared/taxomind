# Module 1 — Implementation notes (Codex constraints)

This reflects the current implementation direction (DataFrame-centric multi-view embeddings).

## Output schema (taxonomy_index)
Must include at least:

- Structural: `code, level, parentCode, label, definition, examples, taxonomyKey, isLeaf`
- Embeddings:
  - `embedding_label` (np.ndarray, always present)
  - `embedding_definition` (np.ndarray, always present unless hard failure; definition is mandatory)
  - `embedding_examples` (np.ndarray or None)
- Metadata:
  - `embedding_model_name` (str)
  - `embedding_dim` (int)

### Evidence columns (recommended to exist from day 0)
To support Module 3 without schema migrations, initialize:

- `evidence_centroid` = None
- `evidence_count` = 0
- `evidence_last_updated` = None (optional)

## Root normalization
Normalize any missing/empty parentCode to `__root__` during preparation.

## Missing embeddings handling
- Examples: if examples text is missing/empty → set `embedding_examples=None`.
- Do **not** use zero vectors and do **not** embed empty strings (both introduce misleading similarity).
