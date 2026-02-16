# Inference Pipelines (`inference` and `inference_batch`)

## Overview

This project currently exposes two inference pipeline names:

- `inference`
- `inference_batch`

Both run the same logical stages:

1. Load embedding model and taxonomy index
2. Build graph/retrieval/scoring views
3. Load query input (string, list, or DataFrame)
4. Embed queries in batch
5. Retrieve candidates
6. Route top-down with stopping logic
7. Run scoped validation/override checks
8. Format final predictions as a DataFrame

`inference_batch` uses batch-oriented node names, but behavior and outputs are equivalent.

## Upstream Dependency

Run `build_taxonomy` for the target `taxonomy_key` first, so `taxonomy_index` exists.

## Pipeline Nodes

1. `load_inference_embedding_model` / `load_inference_embedding_model_batch`
2. `load_taxonomy_index_node` / `load_taxonomy_index_batch`
3. `load_taxonomy_graph_node` / `load_taxonomy_graph_batch`
4. `build_retrieval_index_node` / `build_retrieval_index_batch`
5. `prepare_scoring_views_node` / `prepare_scoring_views_batch`
6. `load_queries_node` / `load_queries_batch`
7. `embed_queries_node` / `embed_queries_batch`
8. `batch_retrieve_candidates_node`
9. `batch_route_topdown_node`
10. `batch_validate_scoped_node`
11. `format_predictions_batch_node`

## Inputs

From catalog/params:

- `taxonomy_index` (partitioned dataset)
- `params:taxonomy_key`
- `params:model_name`
- `params:embedding.cache_dir`
- `params:embedding.local_files_only`
- `params:embedding.batch_size`
- `params:embedding_prefix.query`
- `params:inference_query_input`
- `params:inference.*`

`inference_query_input` supported formats:

- `str`
- `List[str]`
- `pd.DataFrame` with `text` column

## Outputs

Primary output:

- `inference_predictions_df`

Intermediate datasets:

- `inference_embedding_model`
- `inference_taxonomy_df`
- `inference_taxonomy_graph`
- `inference_retrieval_index`
- `inference_scoring_views`
- `inference_queries_df`
- `inference_queries_embedded_df`
- `inference_candidates_df`
- `inference_routing_df`
- `inference_validated_df`

## Final Output Schema (`inference_predictions_df`)

- `query_id`
- `query`
- `predicted_code`
- `predicted_label`
- `predicted_level`
- `score`
- `ambiguous`
- `alternatives`
- `stopping_reason`
- `path`
- `validation_status`
- `validation_override_code`
- `validation_margin`
- `validation_stability_gap`

## Main Parameters (`conf/base/parameters.yml`)

- `inference.retrieval_k`
- `inference.beam_count`
- `inference.enable_beam_selection`
- `inference.min_descent_gap`
- `inference.parent_veto_margin`
- `inference.enable_parent_veto`
- `inference.evidence_tau`
- `inference.evidence_max_beta`
- `inference.short_query_tokens`
- `inference.validation_threshold`
- `inference.validation_stability_margin`
- `inference.use_updated_evidence`
- `inference.max_depth`

## Run

Build index first:

```bash
kedro run --pipeline=build_taxonomy --params="taxonomy_key:ISCO"
```

Run inference (single string/list/DataFrame input via `inference_query_input`):

```bash
kedro run --pipeline=inference
```

Run batch variant explicitly:

```bash
kedro run --pipeline=inference_batch
```

Override input at runtime:

```bash
kedro run --pipeline=inference_batch --params="inference_query_input:'I work as a software developer'"
```

## Notes

- Retrieval is label-embedding based; routing uses scoring views and evidence blending.
- Non-leaf nodes are valid outputs when stopping criteria are triggered.
- `inference_query_text` and single-result datasets (`inference_prediction`, `inference_candidates`, etc.) are in catalog for compatibility but are not produced by the current DAG.
