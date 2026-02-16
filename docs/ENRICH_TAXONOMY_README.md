# Enrich Taxonomy Pipeline (`enrich_taxonomy`)

## Overview

`enrich_taxonomy` enriches taxonomy text with LLM-generated cleanup and examples.
It adds parent-path context, finds similar labels by embedding similarity, calls an LLM,
and saves enriched taxonomy rows to `taxonomy_definition_llm`.

## Pipeline Nodes

1. `load_and_prepare_taxonomy`
2. `build_parent_paths`
3. `load_embedding_model`
4. `build_similar_labels`
5. `enrich_with_llm`
6. `finalize_enriched_taxonomy`

## Inputs

- `taxonomy_definition` (PartitionedDataset)
- `params:taxonomy_key`
- `params:model_name`
- `params:embedding.cache_dir`
- `params:embedding.local_files_only`
- `params:embedding_prefix`
- `params:embedding.batch_size`
- `params:enrich_taxonomy.k_similar_labels`
- `params:enrich_taxonomy.llm_config`
- `params:enrich_taxonomy.max_rows`
- `params:enrich_taxonomy.apply_cleaned`

## Output

- `taxonomy_definition_llm` (PartitionedDataset)

Output rows include added enrichment fields:

- `definition_clean`
- `examples_clean`
- `positive_examples`
- `negative_examples`

If `apply_cleaned: true`, cleaned text overwrites `definition` and `examples`.

## LLM Providers

`llm_config.provider` supports:

- `openai` (requires `OPENAI_API_KEY`)
- `ollama`

## Run

```bash
kedro run --pipeline=enrich_taxonomy --params="taxonomy_key:ISCO"
```
