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
                outputs="taxonomy_embedded_raw",
                name="embed_taxonomy",
            ),
            node(
                func=nodes.prepare_partitioned_taxonomy,
                inputs="taxonomy_embedded_raw",
                outputs="taxonomy_embedded",
                name="prepare_partitioned_taxonomy",
            ),
        ]
    )
    return taxonomy_pipeline
