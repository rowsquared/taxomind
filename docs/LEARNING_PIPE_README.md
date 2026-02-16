# Learning Pipeline (`learning_pipe`)

## Overview

`learning_pipe` applies incremental evidence updates from `/learn` payloads.
It validates payloads, converts sentences to updates, embeds update text, and updates
per-node evidence centroids in `taxonomy_index`.

## Pipeline Nodes

1. `learning_validate_payload`
2. `learning_convert_payload`
3. `learning_load_taxonomy_index`
4. `learning_embed_updates`
5. `learning_apply_updates`

## Inputs

- `api_training_payload` (PartitionedDataset, JSON)
- `taxonomy_index_input` (PartitionedDataset, Parquet)
- `params:model_name`
- `params:embedding.cache_dir`
- `params:embedding.local_files_only`
- `params:embedding_prefix.query`
- `params:embedding.batch_size`

Expected payload shape (simplified):

```json
{
  "taxonomyKey": "ISCO",
  "sentences": [
    {
      "sentenceId": "abc",
      "fields": {"...": "..."},
      "annotations": [{"level": 4, "nodeCode": "2512"}]
    }
  ]
}
```

## Outputs

- `taxonomy_index` (updated partitioned index)
- `learning_update_summary` (counts and stats)

Intermediate outputs:

- `learning_validated_payload`
- `learning_updates_df`
- `learning_taxonomy_key`
- `learning_taxonomy_df`
- `learning_embedded_updates_df`
- `learning_embed_stats`

## Behavior

- Uses deepest valid annotation per sentence.
- Skips sentences with missing/invalid annotations or empty text.
- Updates only annotated nodes (no ancestor propagation).
- Stores previous evidence snapshot columns before first update per touched node.

## Run

```bash
kedro run --pipeline=learning_pipe
```
