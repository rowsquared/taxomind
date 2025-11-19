"""Zero-shot pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.load_test_sentences,
                inputs="isco_test_sentences",
                outputs="zero_shot_inputs",
                name="load_test_sentences",
            ),
            node(
                func=nodes.build_full_paths,
                inputs="taxonomy_embedded",
                outputs="taxonomy_full_paths",
                name="build_full_paths",
            ),
            node(
                func=nodes.embed_full_paths,
                inputs={
                    "paths": "taxonomy_full_paths",
                    "model_name": "params:zero_shot.model_name",
                },
                outputs="taxonomy_full_path_embeddings",
                name="embed_full_paths",
            ),
            node(
                func=nodes.compute_sentence_embeddings,
                inputs={
                    "test_sentences": "zero_shot_inputs",
                    "model_name": "params:zero_shot.model_name",
                },
                outputs="zero_shot_sentence_embeddings",
                name="compute_sentence_embeddings",
            ),
            node(
                func=nodes.compute_top_down_routes,
                inputs={
                    "sentence_embeddings": "zero_shot_sentence_embeddings",
                    "taxonomy_embedded": "taxonomy_embedded",
                    "top_k": "params:zero_shot.topdown_k",
                },
                outputs="zero_shot_topdown",
                name="compute_top_down",
            ),
            node(
                func=nodes.compute_bottom_up_routes,
                inputs={
                    "sentence_embeddings": "zero_shot_sentence_embeddings",
                    "taxonomy_embedded": "taxonomy_embedded",
                    "top_k": "params:zero_shot.bottomup_k",
                },
                outputs="zero_shot_bottomup",
                name="compute_bottom_up",
            ),
            node(
                func=nodes.compute_flat_routes,
                inputs={
                    "sentence_embeddings": "zero_shot_sentence_embeddings",
                    "taxonomy_full_paths": "taxonomy_full_path_embeddings",
                    "top_k": "params:zero_shot.flat_k",
                },
                outputs="zero_shot_flat",
                name="compute_flat_routes",
            ),
            node(
                func=nodes.compare_routes,
                inputs=[
                    "zero_shot_topdown",
                    "zero_shot_bottomup",
                    "zero_shot_flat",
                ],
                outputs="zero_shot_comparison",
                name="compare_routes",
            ),
            node(
                func=nodes.finalize_predictions,
                inputs={
                    "compared_results": "zero_shot_comparison",
                    "taxonomy_embedded": "taxonomy_embedded",
                    "judge_model_name": "params:judge.model_name",
                    "encoder_model_name": "params:zero_shot.model_name",
                    "debug_level": "params:zero_shot.debug_level",
                },
                outputs="zero_shot_judgement",
                name="finalize_predictions",
            ),
        ]
    )
