# Inference Batch Pipeline (`inference_batch`)

## Overview

`inference_batch` is the batch-oriented registration of the inference DAG.
It is used heavily by API classification/labeling jobs and produces `inference_predictions_df`.

For full details of stages, inputs, outputs, and parameters, see:

- `docs/INFERENCE_README.md`

## Differences From `inference`

- Same logical processing and output schema.
- Uses a separate `create_batch_pipeline()` registration and batch-specific node names.

## Run

```bash
kedro run --pipeline=inference_batch
```

With runtime override:

```bash
kedro run --pipeline=inference_batch --params="inference_query_input:['query 1','query 2']"
```
