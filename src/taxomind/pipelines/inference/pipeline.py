"""
Module 2 - Inference Pipeline Definition.

This module defines the Kedro pipeline for runtime inference with:
1. Taxonomy loading and graph construction
2. Label-based retrieval index
3. Multi-view scoring preparation
4. Query embedding
5. Top-down routing with asymmetric stopping
6. Explainable prediction formatting

Spec Reference: Module 2 — Inference
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_taxonomy_index,
    load_taxonomy_graph,
    build_retrieval_index,
    prepare_scoring_views,
    load_queries,
    embed_queries,
    batch_inference,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the inference pipeline for runtime classification.

    This pipeline supports both single and batch queries through
    the inference_query_input parameter, which can be:
    - Single string: "I work as a teacher"
    - List of strings: ["query 1", "query 2", ...]
    - DataFrame: with 'text' column

    The pipeline loads the model and taxonomy once, then processes
    all queries in batch for efficiency.

    Pipeline Flow:
    1. Load embedding model (setup, once)
    2. Load taxonomy index and build graph (setup, once)
    3. Build retrieval index and scoring views (setup, once)
    4. Load and normalize queries (handles string/list/DataFrame)
    5. Embed all queries in batch
    6. Process each query through retrieval + routing + formatting
    7. Return DataFrame with all predictions

    Inputs (from catalog/params):
        - taxonomy_index: Partitioned dataset with taxonomy embeddings
        - params:inference_query_input: One of:
            * str: Single query "text here"
            * List[str]: Multiple queries ["query1", "query2"]
            * pd.DataFrame: DataFrame with 'text' column
        - params:taxonomy_key: Which taxonomy to use
        - params:inference.*: All inference parameters

    Outputs (to catalog):
        - inference_predictions_df: DataFrame with columns:
            * query_id, query, predicted_code, predicted_label,
            * predicted_level, score, ambiguous, alternatives,
            * stopping_reason, path,
            * validation_status, validation_override_code, validation_margin

    Parameters:
        inference.retrieval_k: Number of candidates to retrieve (default 20)
        inference.beam_count: Number of root beams to select (default 2)
        inference.min_descent_gap: Sibling separation threshold (default 0.05)
        inference.parent_veto_margin: Parent competitiveness margin (default 0.05)
        inference.evidence_tau: Evidence confidence threshold (default 10.0)
        inference.evidence_max_beta: Evidence weight cap (default 0.8)
        inference.short_query_tokens: Short-query token threshold (default 2)
        inference.validation_threshold: Override threshold (default 0.05)
        inference.max_depth: Optional maximum depth to descend (default None)
        inference.embedding_batch_size: Batch size for embedding (default 32)

    Returns:
        Kedro Pipeline object

    Usage:
        # Single query
        with KedroSession.create(
            runtime_params={"inference_query_input": "I work as a software developer"}
        ) as session:
            session.run(pipeline_name="inference")

        # Multiple queries
        with KedroSession.create(
            runtime_params={"inference_query_input": ["query 1", "query 2", "query 3"]}
        ) as session:
            session.run(pipeline_name="inference")
    """
    return pipeline(
        [
            # ================================================================
            # Setup Phase: Load taxonomy and build indices (once per session)
            # ================================================================
            node(
                func=lambda model_name: __import__('sentence_transformers', fromlist=['SentenceTransformer']).SentenceTransformer(model_name, trust_remote_code=True),
                inputs="params:model_name",
                outputs="inference_embedding_model",
                name="load_inference_embedding_model",
            ),
            node(
                func=load_taxonomy_index,
                inputs=["taxonomy_index", "params:taxonomy_key"],
                outputs="inference_taxonomy_df",
                name="load_taxonomy_index_node",
            ),
            node(
                func=load_taxonomy_graph,
                inputs="inference_taxonomy_df",
                outputs="inference_taxonomy_graph",
                name="load_taxonomy_graph_node",
            ),
            node(
                func=build_retrieval_index,
                inputs="inference_taxonomy_df",
                outputs="inference_retrieval_index",
                name="build_retrieval_index_node",
            ),
            node(
                func=prepare_scoring_views,
                inputs="inference_taxonomy_df",
                outputs="inference_scoring_views",
                name="prepare_scoring_views_node",
            ),
            # ================================================================
            # Query Phase: Process all queries in batch
            # ================================================================
            node(
                func=load_queries,
                inputs="params:inference_query_input",
                outputs="inference_queries_df",
                name="load_queries_node",
            ),
            node(
                func=embed_queries,
                inputs=[
                    "inference_queries_df",
                    "inference_embedding_model",
                    "params:inference.embedding_batch_size",
                ],
                outputs="inference_queries_embedded_df",
                name="embed_queries_node",
            ),
            node(
                func=batch_inference,
                inputs=[
                    "inference_queries_embedded_df",
                    "inference_retrieval_index",
                    "inference_scoring_views",
                    "inference_taxonomy_graph",
                    "inference_taxonomy_df",
                    "params:inference.retrieval_k",
                    "params:inference.beam_count",
                    "params:inference.min_descent_gap",
                    "params:inference.parent_veto_margin",
                    "params:inference.evidence_tau",
                    "params:inference.evidence_max_beta",
                    "params:inference.short_query_tokens",
                    "params:inference.validation_threshold",
                    "params:inference.max_depth",
                ],
                outputs="inference_predictions_df",
                name="batch_inference_node",
            ),
        ]
    )


def create_batch_pipeline(**kwargs) -> Pipeline:
    """
    Create batch inference pipeline for processing multiple queries.

    This pipeline processes a DataFrame of queries (or string/list inputs)
    through the full inference system. It's optimized for:
    - Batch embedding (more efficient than one-by-one)
    - Single model/taxonomy load
    - Progress tracking
    - Error handling per query

    Pipeline Flow:
    1. Load taxonomy and build indices (setup, once)
    2. Load queries (from string, list, or DataFrame)
    3. Embed all queries in batch
    4. Process each query through retrieval + routing + formatting
    5. Return DataFrame with all predictions

    Inputs (from catalog/params):
        - taxonomy_index: Partitioned dataset with taxonomy embeddings
        - params:inference_query_input: One of:
            * str: Single query "text here"
            * List[str]: Multiple queries ["query1", "query2"]
            * pd.DataFrame: DataFrame with 'text' column
        - params:taxonomy_key: Which taxonomy to use
        - params:inference.*: All inference parameters

    Outputs (to catalog):
        - inference_predictions_df: DataFrame with columns:
            * query_id, query, predicted_code, predicted_label,
            * predicted_level, score, ambiguous, alternatives,
            * stopping_reason, path,
            * validation_status, validation_override_code, validation_margin

    Parameters:
        inference.retrieval_k: Number of candidates to retrieve (default 20)
        inference.beam_count: Number of root beams to select (default 2)
        inference.min_descent_gap: Sibling separation threshold (default 0.05)
        inference.parent_veto_margin: Parent competitiveness margin (default 0.05)
        inference.evidence_tau: Evidence confidence threshold (default 10.0)
        inference.evidence_max_beta: Evidence weight cap (default 0.8)
        inference.short_query_tokens: Short-query token threshold (default 2)
        inference.validation_threshold: Override threshold (default 0.05)
        inference.max_depth: Optional maximum depth (default None)
        inference.embedding_batch_size: Batch size for embedding (default 32)

    Returns:
        Kedro Pipeline object

    Usage:
        # Single query
        kedro run --pipeline=inference_batch --params="inference_query_input:'I work as a software engineer'"

        # Multiple queries (set in parameters.yml)
        inference_query_input:
          - "query 1"
          - "query 2"

        # DataFrame (via catalog)
        inference_query_input:
          type: pandas.CSVDataset
          filepath: data/queries.csv

    Note:
        This pipeline is optimized for batch evaluation and testing.
        For production single-query API inference, use create_pipeline().
    """
    return pipeline(
        [
            # ================================================================
            # Setup Phase: Load taxonomy and build indices (once per batch)
            # ================================================================
            node(
                func=lambda model_name: __import__('sentence_transformers', fromlist=['SentenceTransformer']).SentenceTransformer(model_name, trust_remote_code=True),
                inputs="params:model_name",
                outputs="inference_embedding_model",
                name="load_inference_embedding_model_batch",
            ),
            node(
                func=load_taxonomy_index,
                inputs=["taxonomy_index", "params:taxonomy_key"],
                outputs="inference_taxonomy_df",
                name="load_taxonomy_index_batch",
            ),
            node(
                func=load_taxonomy_graph,
                inputs="inference_taxonomy_df",
                outputs="inference_taxonomy_graph",
                name="load_taxonomy_graph_batch",
            ),
            node(
                func=build_retrieval_index,
                inputs="inference_taxonomy_df",
                outputs="inference_retrieval_index",
                name="build_retrieval_index_batch",
            ),
            node(
                func=prepare_scoring_views,
                inputs="inference_taxonomy_df",
                outputs="inference_scoring_views",
                name="prepare_scoring_views_batch",
            ),
            # ================================================================
            # Query Phase: Process all queries in batch
            # ================================================================
            node(
                func=load_queries,
                inputs="params:inference_query_input",
                outputs="inference_queries_df",
                name="load_queries_batch",
            ),
            node(
                func=embed_queries,
                inputs=[
                    "inference_queries_df",
                    "inference_embedding_model",
                    "params:inference.embedding_batch_size",
                ],
                outputs="inference_queries_embedded_df",
                name="embed_queries_batch",
            ),
            node(
                func=batch_inference,
                inputs=[
                    "inference_queries_embedded_df",
                    "inference_retrieval_index",
                    "inference_scoring_views",
                    "inference_taxonomy_graph",
                    "inference_taxonomy_df",
                    "params:inference.retrieval_k",
                    "params:inference.beam_count",
                    "params:inference.min_descent_gap",
                    "params:inference.parent_veto_margin",
                    "params:inference.evidence_tau",
                    "params:inference.evidence_max_beta",
                    "params:inference.short_query_tokens",
                    "params:inference.validation_threshold",
                    "params:inference.max_depth",
                ],
                outputs="inference_predictions_df",
                name="batch_inference_node",
            ),
        ]
    )
