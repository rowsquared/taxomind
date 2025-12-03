"""Taxonomy pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    taxonomy_pipeline = pipeline(
        [
            node(
                func=nodes.load_taxonomy,
                inputs="taxonomy_request",
                outputs="taxonomy_validated",
                name="load_taxonomy",
            ),
            node(
                func=nodes.add_unknowns,
                inputs="taxonomy_validated",
                outputs="taxonomy_table",
                name="add_unknowns",
            ),
            node(
                func=nodes.enrich_labels,
                inputs="taxonomy_table",
                outputs="taxonomy_enriched",
                name="enrich_labels",
            ),
            node(
                func=nodes.embed_taxonomy,
                inputs={
                    "taxonomy": "taxonomy_enriched",
                    "model_name": "params:embedding.model_name",
                },
                outputs="taxonomy_embedded",
                name="embed_taxonomy",
            ),
            node(
                func=nodes.build_full_paths,
                inputs="taxonomy_enriched",
                outputs="taxonomy_full_paths",
                name="build_full_paths",
            ),
            node(
                func=nodes.embed_full_paths,
                inputs={
                    "taxonomy": "taxonomy_full_paths",
                    "model_name": "params:zero_shot.model_name",
                },
                outputs="taxonomy_full_path_embedded",
                name="embed_full_paths",
            ),
            node(
                func=nodes.build_flat_hierarchical_labels,
                inputs="taxonomy_enriched",
                outputs="taxonomy_flat_hierarchical",
                name="build_flat_hierarchical_labels",
            ),
            node(
                func=nodes.embed_flat_hierarchical_taxonomy,
                inputs={
                    "taxonomy": "taxonomy_flat_hierarchical",
                    "model_name": "params:embedding.model_name",
                },
                outputs="taxonomy_flat_hierarchical_embedded",
                name="embed_flat_hierarchical_taxonomy",
            ),

        ]
    )
    return taxonomy_pipeline
