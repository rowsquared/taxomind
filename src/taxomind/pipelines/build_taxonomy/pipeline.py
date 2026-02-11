"""
Build Taxonomy Pipeline - Numpy-Based Embedding Approach.

This pipeline processes taxonomy definitions and creates numpy-based embeddings
for fast exact similarity search suitable for taxonomies with <1k nodes.
"""

from kedro.pipeline import Pipeline, node, pipeline

from taxomind.utils import embedding_utils
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the build_taxonomy pipeline.

    This pipeline implements Module 1 - Taxonomy Preparation:
    1. Loads taxonomy from partitioned CSV dataset by key
    2. Normalizes text fields (code, labels, definitions, examples)
    3. Loads embedding model once for reuse
    4. Creates label embeddings (primary anchor)
    5. Creates definition embeddings (secondary semantic view)
    6. Creates examples embeddings (tertiary semantic view, optional)
    7. Creates negative examples embeddings (optional, stored only)
    8. Adds embedding metadata (model name, dimension)
    9. Saves taxonomy index with multi-view embeddings

    For API requests (POST /taxonomies), the service layer converts
    the JSON payload to CSV before running this pipeline.

    Returns:
        Pipeline with 9 nodes for multi-view taxonomy index building
    """
    return pipeline(
        [
            node(
                func=nodes.load_taxonomy_from_partition,
                inputs=["taxonomy_definition", "params:taxonomy_key"],
                outputs="taxonomy_raw",
                name="load_taxonomy_from_partition",
            ),
            node(
                func=nodes.normalize_prototype_views,
                inputs="taxonomy_raw",
                outputs="taxonomy_normalized",
                name="normalize_prototype_views",
            ),
            node(
                func=embedding_utils.load_embedding_model,
                inputs=[
                    "params:model_name",
                    "params:embedding.cache_dir",
                    "params:embedding.local_files_only",
                ],
                outputs="embedding_model",
                name="load_embedding_model",
            ),
            node(
                func=nodes.build_text_embeddings,
                inputs=[
                    "taxonomy_normalized",
                    "embedding_model",
                    "params:embedding_label",
                    "params:embedding.batch_size",
                    "params:embedding_prefix.document",
                ],
                outputs="taxonomy_with_label_embeddings",
                name="build_label_embeddings",
            ),
            node(
                func=nodes.build_text_embeddings,
                inputs=[
                    "taxonomy_with_label_embeddings",
                    "embedding_model",
                    "params:embedding_definition",
                    "params:embedding.batch_size",
                    "params:embedding_prefix.document",
                ],
                outputs="taxonomy_with_definition_embeddings",
                name="build_definition_embeddings",
            ),
            node(
                func=nodes.build_text_embeddings,
                inputs=[
                    "taxonomy_with_definition_embeddings",
                    "embedding_model",
                    "params:embedding_examples",
                    "params:embedding.batch_size",
                    "params:embedding_prefix.document",
                ],
                outputs="taxonomy_with_examples_embeddings",
                name="build_examples_embeddings",
            ),
            node(
                func=nodes.build_text_embeddings,
                inputs=[
                    "taxonomy_with_examples_embeddings",
                    "embedding_model",
                    "params:embedding_negative_examples",
                    "params:embedding.batch_size",
                    "params:embedding_prefix.document",
                ],
                outputs="taxonomy_with_negative_examples_embeddings",
                name="build_negative_examples_embeddings",
            ),
            node(
                func=nodes.add_embedding_metadata,
                inputs=[
                    "taxonomy_with_negative_examples_embeddings",
                    "params:model_name",
                ],
                outputs="taxonomy_embeddings",
                name="add_embedding_metadata",
            ),
            node(
                func=nodes.save_taxonomy_index,
                inputs="taxonomy_embeddings",
                outputs="taxonomy_index",
                name="save_taxonomy_index",
            ),
        ]
    )
