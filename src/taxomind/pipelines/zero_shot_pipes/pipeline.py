"""Zero-shot pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.load_sentences,
                inputs="isco_test_sentences",
                outputs=["zero_shot_inputs", "taxonomyKey"],
                name="load_sentences",
            ),

            node(
                func=nodes.compute_sentence_embeddings,
                inputs={
                    "test_sentences": "zero_shot_inputs",
                    "model_name": "params:zero_shot.model_name",
                    "cache_dir": "params:embedding.cache_dir",
                    "local_files_only": "params:embedding.local_files_only",
                },
                outputs="zero_shot_sentence_embeddings",
                name="compute_sentence_embeddings",
            ),
            node(
                func=nodes.compute_top_down_routes,
                inputs={
                    "sentence_embeddings": "zero_shot_sentence_embeddings",
                    "taxonomy_embedded": "taxonomy_embedded",
                    "taxonomy_key": "taxonomyKey",
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
                    "taxonomy_key": "taxonomyKey",
                    "top_k": "params:zero_shot.bottomup_k",
                },
                outputs="zero_shot_bottomup",
                name="compute_bottom_up",
            ),
            node(
                func=nodes.compute_flat_routes,
                inputs={
                    "sentence_embeddings": "zero_shot_sentence_embeddings",
                    "taxonomy_full_paths": "taxonomy_full_path_embedded",
                    "taxonomy_key": "taxonomyKey",
                    "top_k": "params:zero_shot.flat_k",
                },
                outputs="zero_shot_flat",
                name="compute_flat_routes",
            ),
            node(
                func=nodes.compute_hybrid_routes,
                inputs={
                    "sentence_embeddings": "zero_shot_sentence_embeddings",
                    "taxonomy_embedded": "taxonomy_embedded",
                    "taxonomy_full_paths": "taxonomy_full_path_embedded",
                    "taxonomy_key": "taxonomyKey",
                    "top_k": "params:zero_shot.hybrid_k",
                },
                outputs="zero_shot_hybrid",
                name="compute_hybrid_routes",
            ),
            node(
                func=nodes.compare_routes,
                inputs=[
                    "zero_shot_topdown",
                    "zero_shot_bottomup",
                    "zero_shot_flat",
                    "zero_shot_hybrid",
                ],
                outputs="zero_shot_comparison",
                name="compare_routes",
            ),
            node(
                func=nodes.finalize_predictions,
                inputs={
                    "compared_results": "zero_shot_comparison",
                    "taxonomy_embedded": "taxonomy_embedded",
                    "taxonomy_key": "taxonomyKey",
                    "judge_model_name": "params:judge.model_name",
                    "encoder_model_name": "params:zero_shot.model_name",
                    "debug_level": "params:zero_shot.debug_level",
                    "min_confidence": "params:zero_shot.min_confidence",
                    "judge_enabled": "params:judge.enabled",
                },
                outputs="zero_shot_judgement",
                name="finalize_predictions",
            ),
        ]
    )
