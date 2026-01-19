"""
Module 3 - Incremental Learning Pipeline Definition.

This pipeline updates per-node evidence centroids from /learn corrections.
"""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create the incremental learning pipeline."""

    return pipeline(
        [
            node(
                func=nodes.validate_learning_payload,
                inputs="api_training_payload",
                outputs="learning_validated_payload",
                name="learning_validate_payload",
            ),
            node(
                func=nodes.convert_payload_to_updates_df,
                inputs="learning_validated_payload",
                outputs=["learning_updates_df", "learning_taxonomy_key"],
                name="learning_convert_payload",
            ),
            node(
                func=nodes.load_taxonomy_index,
                inputs=["taxonomy_index_input", "learning_taxonomy_key"],
                outputs="learning_taxonomy_df",
                name="learning_load_taxonomy_index",
            ),
            node(
                func=nodes.embed_learning_updates,
                inputs=[
                    "learning_updates_df",
                    "learning_taxonomy_df",
                    "params:model_name",
                    "params:embedding.cache_dir",
                    "params:embedding.local_files_only",
                    "params:embedding_prefix.query",
                    "params:embedding.batch_size",
                ],
                outputs=["learning_embedded_updates_df", "learning_embed_stats"],
                name="learning_embed_updates",
            ),
            node(
                func=nodes.apply_evidence_updates,
                inputs=[
                    "learning_embedded_updates_df",
                    "learning_taxonomy_df",
                    "learning_embed_stats",
                ],
                outputs=["taxonomy_index", "learning_update_summary"],
                name="learning_apply_updates",
            ),
        ]
    )
