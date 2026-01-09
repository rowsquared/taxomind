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
                func=nodes.embed_taxonomy,
                inputs={
                    "taxonomy": "taxonomy_validated",
                    "model_name": "params:embedding.model_name",
                },
                outputs="taxonomy_embedded",
                name="embed_taxonomy",
            ),


        ]
    )
    return taxonomy_pipeline
