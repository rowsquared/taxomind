# Module 1 — Implementation notes (Codex constraints)

This reflects the current implementation direction (DataFrame-centric multi-view embeddings).

## Output schema (taxonomy_index)
Must include at least:

- Structural: `code, level, parentCode, label, definition, examples, taxonomyKey, isLeaf`
- Embeddings:
  - `embedding_label` (np.ndarray, always present)
- `embedding_definition` (np.ndarray or None; empty definitions are treated as missing)
  - `embedding_examples` (np.ndarray or None)
- Metadata:
  - `embedding_model_name` (str)
  - `embedding_dim` (int)

### Evidence columns (recommended to exist from day 0)
To support Module 3 without schema migrations, initialize:

- `evidence_centroid` = None
- `evidence_count` = 0
- `evidence_last_updated` = None (optional)

Optional snapshot fields (if you want rollback/inspect previous state):
- `last_evidence_centroid` = None
- `last_evidence_count` = 0
- `last_evidence_last_updated` = None

## Root normalization
Normalize any missing/empty parentCode to `__root__` during preparation.

## Missing embeddings handling
- Definitions: if definition text is missing/empty → set `embedding_definition=None`.
- Examples: if examples text is missing/empty → set `embedding_examples=None`.
- Do **not** use zero vectors and do **not** embed empty strings (both introduce misleading similarity).
