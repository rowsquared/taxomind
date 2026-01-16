"""Error analysis pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.load_classifai_validation_targets,
                inputs=["classifai_validation_data", "taxonomy_index"],
                outputs="error_analysis_classifai_targets",
                name="load_classifai_validation_targets",
            ),
            node(
                func=nodes.load_taxonomy_training_targets,
                inputs=["taxonomy_training", "taxonomy_index"],
                outputs="error_analysis_taxonomy_training_targets",
                name="load_taxonomy_training_targets",
            ),
            node(
                func=nodes.load_training_sentences_targets,
                inputs=["training_sentences", "taxonomy_index"],
                outputs="error_analysis_training_sentences_targets",
                name="load_training_sentences_targets",
            ),
        ]
    )
