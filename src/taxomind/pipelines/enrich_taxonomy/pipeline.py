"""
Pipeline for enriching taxonomy definitions using LLM.

This pipeline:
1. Loads taxonomy from partitioned dataset
2. Builds parent path labels for context
3. Loads embedding model
4. Finds similar labels at each level using embeddings
5. Calls LLM to clean definitions and generate pos/neg examples
6. Outputs enriched taxonomy to partitioned dataset
"""

from kedro.pipeline import Pipeline, node

from taxomind.utils import embedding_utils
from .nodes import (
    build_parent_paths,
    build_similar_labels,
    enrich_with_llm,
    finalize_enriched_taxonomy,
    load_and_prepare_taxonomy,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the enrich_taxonomy pipeline."""
    return Pipeline(
        [
            node(
                func=load_and_prepare_taxonomy,
                inputs=[
                    "taxonomy_definition",
                    "params:taxonomy_key",
                ],
                outputs="taxonomy_prepared",
                name="load_and_prepare_taxonomy",
            ),
            node(
                func=build_parent_paths,
                inputs="taxonomy_prepared",
                outputs="taxonomy_with_paths",
                name="build_parent_paths",
            ),
            node(
                func=embedding_utils.load_embedding_model,
                inputs=[
                    "params:model_name",
                    "params:embedding.cache_dir",
                    "params:embedding.local_files_only",
                ],
                outputs="enrich_embedding_model",
                name="load_embedding_model",
            ),
            node(
                func=build_similar_labels,
                inputs=[
                    "taxonomy_with_paths",
                    "enrich_embedding_model",
                    "params:enrich_taxonomy.k_similar_labels",
                    "params:embedding_prefix",
                    "params:embedding.batch_size",
                ],
                outputs="taxonomy_with_similar",
                name="build_similar_labels",
            ),
            node(
                func=enrich_with_llm,
                inputs=[
                    "taxonomy_with_similar",
                    "params:enrich_taxonomy.llm_config",
                    "params:enrich_taxonomy.max_rows",
                ],
                outputs="taxonomy_llm_enriched",
                name="enrich_with_llm",
            ),
            node(
                func=finalize_enriched_taxonomy,
                inputs=[
                    "taxonomy_llm_enriched",
                    "params:enrich_taxonomy.apply_cleaned",
                    "params:taxonomy_key",
                ],
                outputs="taxonomy_definition_llm",
                name="finalize_enriched_taxonomy",
            ),
        ]
    )
