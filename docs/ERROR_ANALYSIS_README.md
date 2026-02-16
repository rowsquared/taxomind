# Error Analysis Pipeline (`error_analysis`)

## Overview

`error_analysis` standardizes multiple labeled sources into a common target schema
for downstream evaluation and debugging.

## Pipeline Nodes

1. `load_classifai_validation_targets`
2. `load_taxonomy_training_targets`
3. `load_training_sentences_targets`

## Inputs

- `classifai_validation_data` (CSV)
- `taxonomy_training` (PartitionedDataset)
- `training_sentences` (PartitionedDataset)
- `taxonomy_index` (PartitionedDataset)

## Outputs

- `error_analysis_classifai_targets`
- `error_analysis_taxonomy_training_targets`
- `error_analysis_training_sentences_targets`

All outputs are normalized to a comparable schema including:

- `dataset`
- `taxonomy_key`
- `query_id`
- `source_id`
- `query_text`
- `target_code`
- `target_level`
- `target_level_1`
- `target_level_2`
- `target_level_3`
- `target_level_4`

## Behavior Notes

- Codes are normalized to digit strings when possible.
- Missing higher-level targets are reconstructed from taxonomy parent maps.
- `classifai_validation_data` emits separate rows for `ISCO` and `ISIC` per source row.

## Run

```bash
kedro run --pipeline=error_analysis
```
