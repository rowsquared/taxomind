# Build Taxonomy Pipeline (`build_taxonomy`)

## Overview

`build_taxonomy` creates the taxonomy index used by inference and learning pipelines.
It loads one taxonomy partition, normalizes text, computes embedding views, adds metadata,
and saves a partitioned Parquet index.

## Pipeline Nodes

1. `load_taxonomy_from_partition`
2. `normalize_prototype_views`
3. `load_embedding_model`
4. `build_label_embeddings`
5. `build_definition_embeddings`
6. `build_examples_embeddings`
7. `build_negative_examples_embeddings`
8. `add_embedding_metadata`
9. `save_taxonomy_index`

## Inputs

- `taxonomy_definition` (PartitionedDataset)
- `params:taxonomy_key`
- `params:model_name`
- `params:embedding.cache_dir`
- `params:embedding.local_files_only`
- `params:embedding.batch_size`
- `params:embedding_prefix.document`
- `params:embedding_label`
- `params:embedding_definition`
- `params:embedding_examples`
- `params:embedding_negative_examples`

## Output

- `taxonomy_index` (PartitionedDataset, Parquet)

Each saved partition includes taxonomy structure, embedding views, and evidence state columns (initialized for incremental learning):

- `embedding_label`, `embedding_definition`, `embedding_examples`
- `embedding_model_name`, `embedding_dim`
- `evidence_centroid`, `evidence_count`, `evidence_last_updated`
- `last_evidence_centroid`, `last_evidence_count`, `last_evidence_last_updated`

## Run

```bash
kedro run --pipeline=build_taxonomy --params="taxonomy_key:ISCO"
```

Use `taxonomy_key:ISIC` (or another existing partition key) as needed.
