"""
Build Taxonomy Pipeline - Numpy-Based Embedding Approach.

This pipeline processes taxonomy definitions and creates numpy-based embeddings
for fast exact similarity search suitable for taxonomies with <1k nodes.
"""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the build_taxonomy pipeline (from CSV files).

    This pipeline:
    1. Loads taxonomy from partitioned CSV dataset by key
    2. Normalizes text fields (code, labels, definitions, examples)
    3. Builds taxonomy adjacency graph (parent-child relationships)
    4. Creates numpy embedding matrix for label-based similarity search
    5. Saves taxonomy index with embeddings and metadata

    Returns:
        Pipeline with 5 nodes for taxonomy index building
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
                func=nodes.build_taxonomy_adjacency,
                inputs="taxonomy_normalized",
                outputs="taxonomy_adjacency",
                name="build_taxonomy_adjacency",
            ),
            node(
                func=nodes.build_numpy_embeddings,
                inputs=["taxonomy_adjacency", "params:model_name"],
                outputs="taxonomy_embeddings",
                name="build_numpy_embeddings",
            ),
            node(
                func=nodes.save_taxonomy_index,
                inputs="taxonomy_embeddings",
                outputs="taxonomy_index",
                name="save_taxonomy_index",
            ),
        ]
    )


def create_pipeline_from_request(**kwargs) -> Pipeline:
    """
    Create the build_taxonomy_from_request pipeline (from JSON files).

    This pipeline:
    1. Loads taxonomy from JSON request files by key
    2. Parses JSON structure and converts to DataFrame
    3. Normalizes text fields (code, labels, definitions, examples)
    4. Builds taxonomy adjacency graph (parent-child relationships)
    5. Creates numpy embedding matrix for label-based similarity search
    6. Saves taxonomy index with embeddings and metadata

    Returns:
        Pipeline with 5 nodes for taxonomy index building from JSON
    """
    return pipeline(
        [
            node(
                func=nodes.load_taxonomy_from_request,
                inputs=["taxonomy_request_files", "params:taxonomy_key"],
                outputs="taxonomy_raw",
                name="load_taxonomy_from_request",
            ),
            node(
                func=nodes.normalize_prototype_views,
                inputs="taxonomy_raw",
                outputs="taxonomy_normalized",
                name="normalize_prototype_views",
            ),
            node(
                func=nodes.build_taxonomy_adjacency,
                inputs="taxonomy_normalized",
                outputs="taxonomy_adjacency",
                name="build_taxonomy_adjacency",
            ),
            node(
                func=nodes.build_numpy_embeddings,
                inputs=["taxonomy_adjacency", "params:model_name"],
                outputs="taxonomy_embeddings",
                name="build_numpy_embeddings",
            ),
            node(
                func=nodes.save_taxonomy_index,
                inputs="taxonomy_embeddings",
                outputs="taxonomy_index",
                name="save_taxonomy_index",
            ),
        ]
    )
