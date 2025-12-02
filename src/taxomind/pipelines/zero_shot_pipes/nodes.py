"""Zero-shot routing nodes with multilingual guarantees."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from taxomind.utils import (
    embedding_utils,
    judge_utils,
    scoring_utils,
    taxonomy_utils,
)


def compose_inference_text(fields: Dict[str, Any]) -> str:
    """Concatenate key-value pairs into a multilingual inference string."""

    parts: List[str] = []
    for key, value in sorted(fields.items()):
        if value is None:
            continue
        parts.append(f"{key}: {value}")
    return "\n".join(parts)


def load_sentences(test_dataset: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """Transform raw JSON structure into inference-ready entries."""

    taxonomy_key = test_dataset.get("taxonomyKey")
    entries: List[Dict[str, Any]] = []
    for record in test_dataset.get("sentences", []):
        fields = record.get("fields", {})
        text = compose_inference_text(fields)
        entries.append(
            {
                "sentence_id": record.get("sentence_id"),
                "taxonomyKey": taxonomy_key,
                "fields": fields,
                "text": text,
            }
        )
    return entries, taxonomy_key


def compute_sentence_embeddings(
    test_sentences: List[Dict[str, Any]], model_name: str
) -> List[Dict[str, Any]]:
    """Embed all inference texts once for downstream independence."""

    texts = [record.get("text", "") for record in test_sentences]
    embeddings = embedding_utils.embed_texts(texts, model_name=model_name)
    enriched: List[Dict[str, Any]] = []
    for record, embedding in zip(test_sentences, embeddings):
        enriched_record = dict(record)
        enriched_record["embedding"] = embedding
        enriched.append(enriched_record)
    return enriched


def top_down_route(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Perform multilingual top-down routing with full Top-K context."""

    best_nodes: List[Dict[str, Any]] = []
    parent_code: str | None = None

    max_level = int(taxonomy_embedded["level"].max()) if not taxonomy_embedded.empty else 0
    for level in range(1, max_level + 1):
        level_nodes = taxonomy_utils.get_level_nodes(
            taxonomy_embedded, level, parent_code
        )
        if level_nodes.empty:
            break

        scored = scoring_utils.score_candidates(input_embedding, level_nodes)
        if not scored:
            break

        best = scored[0]
        best_nodes.append(best)
        parent_code = best.get("code")
        if parent_code is None:
            break

    best_leaf = best_nodes[-1] if best_nodes else None
    best_route = _build_route_annotations(best_nodes)

    leaves = taxonomy_embedded[taxonomy_embedded["isLeaf"].astype(bool)]
    leaf_scores = scoring_utils.score_candidates(input_embedding, leaves)
    topk_candidates = leaf_scores[:top_k]

    # Attach path_nodes to each candidate to avoid taxonomy walks later
    for candidate in topk_candidates:
        if candidate and not candidate.get("path_nodes"):
            parent_codes = taxonomy_utils.get_parent_chain(
                taxonomy_embedded, candidate.get("code")
            )
            path_nodes: List[Dict[str, Any]] = []
            for code in parent_codes:
                node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
                if not node_row.empty:
                    path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))
            # Add leaf itself
            leaf_node = dict(candidate)
            if "path_nodes" in leaf_node:
                del leaf_node["path_nodes"]
            path_nodes.append(leaf_node)
            candidate["path_nodes"] = path_nodes

    topk_routes = _build_topk_routes(
        topk_candidates, taxonomy_embedded, input_embedding
    )

    if not best_route and topk_routes:
        best_route = topk_routes[0].get("route", [])

    best_leaf_score = (
        float(best_leaf.get("score", 0.0)) if best_leaf else None
    )

    return {
        "best_leaf": best_leaf,
        "best_leaf_score": best_leaf_score,
        "best_route": best_route,
        "topk": topk_routes,
    }


def bottom_up_route(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    top_k: int = 5,
    parent_weight: float = 0.3,
) -> Dict[str, Any]:
    """Rank leaves directly with parent-aware scoring and sibling re-ranking.

    Args:
        input_embedding: Input text embedding vector
        taxonomy_embedded: Full taxonomy with embeddings
        top_k: Number of top candidates to return
        parent_weight: Weight for parent context (0.0-1.0), leaf weight is (1.0 - parent_weight)
    """

    leaves = taxonomy_embedded[taxonomy_embedded["isLeaf"].astype(bool)]

    # Parent-aware scoring (Improvement 1)
    enriched_scores = []
    for _, leaf in leaves.iterrows():
        # Score the leaf node itself
        leaf_dict = taxonomy_utils.row_to_candidate(leaf)
        leaf_embedding = leaf.get("embedding")

        if leaf_embedding is None or not isinstance(leaf_embedding, (list, np.ndarray)):
            continue

        leaf_score = float(np.dot(input_embedding, leaf_embedding) /
                          (np.linalg.norm(input_embedding) * np.linalg.norm(leaf_embedding)))

        # Get parent chain and score parents
        parent_codes = taxonomy_utils.get_parent_chain(taxonomy_embedded, leaf.get("code"))

        if parent_codes:
            parent_nodes = taxonomy_embedded[taxonomy_embedded["code"].isin(parent_codes)]
            parent_scores_list = scoring_utils.score_candidates(input_embedding, parent_nodes)

            if parent_scores_list:
                avg_parent_score = float(np.mean([p.get("score", 0.0) for p in parent_scores_list]))
            else:
                avg_parent_score = 0.0
        else:
            avg_parent_score = 0.0

        # Blend scores: default 70% leaf, 30% parent context
        blended_score = (1.0 - parent_weight) * leaf_score + parent_weight * avg_parent_score

        enriched_scores.append({
            **leaf_dict,
            "score": blended_score,
            "leaf_score": leaf_score,
            "parent_context_score": avg_parent_score,
        })

    # Sort by blended score
    enriched_scores.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Sibling-aware re-ranking (Improvement 3)
    topk_candidates = _rerank_by_sibling_diversity(enriched_scores, top_k)

    # Attach path_nodes to each candidate to avoid taxonomy walks later
    for candidate in topk_candidates:
        if candidate and not candidate.get("path_nodes"):
            parent_codes = taxonomy_utils.get_parent_chain(
                taxonomy_embedded, candidate.get("code")
            )
            path_nodes: List[Dict[str, Any]] = []
            for code in parent_codes:
                node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
                if not node_row.empty:
                    path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))
            # Add leaf itself
            leaf_node = dict(candidate)
            if "path_nodes" in leaf_node:
                del leaf_node["path_nodes"]
            path_nodes.append(leaf_node)
            candidate["path_nodes"] = path_nodes

    best_leaf = topk_candidates[0] if topk_candidates else None
    best_route = _build_route_annotations_for_leaf(
        best_leaf, taxonomy_embedded, input_embedding
    )
    topk_routes = _build_topk_routes(
        topk_candidates, taxonomy_embedded, input_embedding
    )

    best_leaf_score = (
        float(best_leaf.get("score", 0.0)) if best_leaf else None
    )

    if not best_route and topk_routes:
        best_route = topk_routes[0].get("route", [])

    return {
        "best_leaf": best_leaf,
        "best_leaf_score": best_leaf_score,
        "best_route": best_route,
        "topk": topk_routes,
    }


def flat_route(
    input_embedding: List[float],
    taxonomy_full_paths: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Return the best matching full path for a sentence."""

    scores = scoring_utils.score_candidates(
        input_embedding, taxonomy_full_paths
    )
    topk_paths = scores[:top_k]
    best_path = topk_paths[0] if topk_paths else None
    best_route = []
    best_leaf = None
    best_leaf_score = None

    if best_path:
        best_route = _build_route_annotations_from_path_nodes(
            best_path.get("path_nodes"),
            float(best_path.get("score", 0.0)),
        )
        path_nodes = best_path.get("path_nodes")
        if path_nodes is not None and len(path_nodes) > 0:
            best_leaf = dict(path_nodes[-1])
            best_leaf["score"] = float(best_path.get("score", 0.0))
        best_leaf_score = float(best_path.get("score", 0.0))

    topk_routes: List[Dict[str, Any]] = []
    for path_candidate in topk_paths:
        path_nodes = path_candidate.get("path_nodes")
        leaf_node = None
        if path_nodes is not None and len(path_nodes) > 0:
            leaf_node = dict(path_nodes[-1])
            leaf_node["score"] = float(path_candidate.get("score", 0.0))
        topk_routes.append(
            {
                "leaf": leaf_node,
                "score": float(path_candidate.get("score", 0.0)),
                "route": _build_route_annotations_from_path_nodes(
                    path_nodes,
                    float(path_candidate.get("score", 0.0)),
                ),
            }
        )

    return {
        "best_leaf": best_leaf,
        "best_leaf_score": best_leaf_score,
        "best_route": best_route,
        "topk": topk_routes,
    }


def compute_top_down_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_embedded: Dict[str, Any],
    taxonomy_key: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute top-down routes for every inference sample."""

    # Load the specific taxonomy partition
    taxonomy_df = taxonomy_embedded[taxonomy_key]()

    results: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        text = record.get("text", "")
        td = top_down_route(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_df,
            top_k=top_k,
        )
        results.append(
            {
                "sentenceId": record.get("sentence_id"),
                "text": text,
                "taxonomyKey": record.get("taxonomyKey"),
                "topdown_best_leaf": td.get("best_leaf"),
                "topdown_best_leaf_score": td.get("best_leaf_score"),
                "topdown_best_route": td.get("best_route", []),
                "topdown_topk_routes": td.get("topk", []),
            }
        )
    return results


def compute_bottom_up_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_embedded: Dict[str, Any],
    taxonomy_key: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute bottom-up routes leveraging existing embeddings."""

    # Load the specific taxonomy partition
    taxonomy_df = taxonomy_embedded[taxonomy_key]()

    outputs: List[Dict[str, Any]] = []

    for record in sentence_embeddings:
        bu = bottom_up_route(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_df,
            top_k=top_k,
        )
        outputs.append(
            {
                "sentenceId": record.get("sentence_id"),
                "text": record.get("text"),
                "taxonomyKey": record.get("taxonomyKey"),
                "bottomup_best_leaf": bu.get("best_leaf"),
                "bottomup_best_leaf_score": bu.get("best_leaf_score"),
                "bottomup_best_route": bu.get("best_route", []),
                "bottomup_topk_routes": bu.get("topk", []),
            }
        )
    return outputs


def hybrid_bottom_up_flat_route(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    taxonomy_full_paths: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Hybrid routing: bottom-up leaf candidates validated with full path scores.

    This combines the precision of bottom-up leaf scoring with the contextual
    validation of flat routing by re-scoring leaf candidates using their full paths.

    Args:
        input_embedding: Input text embedding vector
        taxonomy_embedded: Full taxonomy with embeddings
        taxonomy_full_paths: Pre-computed full paths with embeddings
        top_k: Number of top candidates to return
    """

    # Step 1: Get bottom-up leaf candidates
    leaves = taxonomy_embedded[taxonomy_embedded["isLeaf"].astype(bool)]
    leaf_scores = scoring_utils.score_candidates(input_embedding, leaves)
    topk_candidates = leaf_scores[:top_k * 2]  # Get 2x candidates for re-ranking

    # Step 2: For each candidate, find its full path and re-score
    path_rescores = []
    for candidate in topk_candidates:
        leaf_code = candidate.get("code")

        # Find matching path in taxonomy_full_paths
        matching_paths = taxonomy_full_paths[
            taxonomy_full_paths["leaf_code"] == leaf_code
        ]

        if not matching_paths.empty:
            path_record = matching_paths.iloc[0]
            path_embedding = path_record.get("embedding")

            if path_embedding is not None and isinstance(path_embedding, (list, np.ndarray)):
                # Re-score using full path embedding
                path_score = float(
                    np.dot(input_embedding, path_embedding) /
                    (np.linalg.norm(input_embedding) * np.linalg.norm(path_embedding))
                )

                # Blend: 50% leaf score, 50% path score
                hybrid_score = 0.5 * candidate.get("score", 0.0) + 0.5 * path_score

                path_rescores.append({
                    **candidate,
                    "leaf_score": candidate.get("score", 0.0),
                    "path_score": path_score,
                    "score": hybrid_score,  # Override with hybrid score
                    "path_nodes": path_record.get("path_nodes"),
                    "path_text": path_record.get("path_text"),
                })

    # Step 3: Sort by hybrid score
    path_rescores.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    topk_paths = path_rescores[:top_k]

    best_path = topk_paths[0] if topk_paths else None
    best_route = []
    best_leaf = None
    best_leaf_score = None

    if best_path:
        best_route = _build_route_annotations_from_path_nodes(
            best_path.get("path_nodes"),
            float(best_path.get("score", 0.0)),
        )
        path_nodes = best_path.get("path_nodes")
        if path_nodes is not None and len(path_nodes) > 0:
            best_leaf = dict(path_nodes[-1] if isinstance(path_nodes[-1], dict) else {})
            best_leaf["score"] = float(best_path.get("score", 0.0))
        best_leaf_score = float(best_path.get("score", 0.0))

    topk_routes: List[Dict[str, Any]] = []
    for path_candidate in topk_paths:
        path_nodes = path_candidate.get("path_nodes")
        leaf_node = None
        if path_nodes is not None and len(path_nodes) > 0:
            leaf_node = dict(path_nodes[-1] if isinstance(path_nodes[-1], dict) else {})
            leaf_node["score"] = float(path_candidate.get("score", 0.0))
        topk_routes.append(
            {
                "leaf": leaf_node,
                "score": float(path_candidate.get("score", 0.0)),
                "route": _build_route_annotations_from_path_nodes(
                    path_nodes,
                    float(path_candidate.get("score", 0.0)),
                ),
            }
        )

    return {
        "best_leaf": best_leaf,
        "best_leaf_score": best_leaf_score,
        "best_route": best_route,
        "topk": topk_routes,
    }


def compute_flat_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_full_paths: Dict[str, Any],
    taxonomy_key: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute flat Top-K similarity against full taxonomy paths."""

    # Load the specific taxonomy partition
    flat_taxonomy_df = taxonomy_full_paths[taxonomy_key]()

    results: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        fr = flat_route(
            input_embedding=record.get("embedding"),
            taxonomy_full_paths=flat_taxonomy_df,
            top_k=top_k,
        )
        results.append(
            {
                "sentenceId": record.get("sentence_id"),
                "text": record.get("text"),
                "taxonomyKey": record.get("taxonomyKey"),
                "flat_best_leaf": fr.get("best_leaf"),
                "flat_best_leaf_score": fr.get("best_leaf_score"),
                "flat_best_route": fr.get("best_route", []),
                "flat_topk_routes": fr.get("topk", []),
            }
        )
    return results


def compute_hybrid_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_embedded: Dict[str, Any],
    taxonomy_full_paths: Dict[str, Any],
    taxonomy_key: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute hybrid bottom-up + flat routes for improved accuracy."""

    # Load the specific taxonomy partitions
    taxonomy_df = taxonomy_embedded[taxonomy_key]()
    flat_taxonomy_df = taxonomy_full_paths[taxonomy_key]()

    results: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        hybrid = hybrid_bottom_up_flat_route(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_df,
            taxonomy_full_paths=flat_taxonomy_df,
            top_k=top_k,
        )
        results.append(
            {
                "sentenceId": record.get("sentence_id"),
                "text": record.get("text"),
                "taxonomyKey": record.get("taxonomyKey"),
                "hybrid_best_leaf": hybrid.get("best_leaf"),
                "hybrid_best_leaf_score": hybrid.get("best_leaf_score"),
                "hybrid_best_route": hybrid.get("best_route", []),
                "hybrid_topk_routes": hybrid.get("topk", []),
            }
        )
    return results


def compare_routes(
    topdown_results: List[Dict[str, Any]],
    bottomup_results: List[Dict[str, Any]],
    flat_results: List[Dict[str, Any]],
    hybrid_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge signals from top-down, bottom-up, flat, and hybrid routing."""

    bottom_map = {record["sentenceId"]: record for record in bottomup_results}
    flat_map = {record["sentenceId"]: record for record in flat_results}
    hybrid_map = {record["sentenceId"]: record for record in hybrid_results}
    compared: List[Dict[str, Any]] = []

    for top in topdown_results:
        sid = top.get("sentenceId")
        bottom = bottom_map.get(sid, {})
        flat = flat_map.get(sid, {})
        hybrid = hybrid_map.get(sid, {})

        top_best = top.get("topdown_best_leaf")
        bottom_best = bottom.get("bottomup_best_leaf")
        flat_best = flat.get("flat_best_leaf")
        hybrid_best = hybrid.get("hybrid_best_leaf")
        top_route = top.get("topdown_best_route") or []
        bottom_route = bottom.get("bottomup_best_route") or []
        flat_best_route = flat.get("flat_best_route") or []
        hybrid_route = hybrid.get("hybrid_best_route") or []

        validation_match = bool(
            _candidate_code(top_best)
            and _candidate_code(bottom_best)
            and _candidate_code(top_best) == _candidate_code(bottom_best)
        )

        compared.append(
            {
                "sentenceId": sid,
                "text": top.get("text"),
                "topdown_best_leaf": top_best,
                "bottomup_best_leaf": bottom_best,
                "flat_best_leaf": flat_best,
                "hybrid_best_leaf": hybrid_best,
                "topdown_topk_routes": top.get("topdown_topk_routes", []),
                "bottomup_topk_routes": bottom.get("bottomup_topk_routes", []),
                "flat_topk_routes": flat.get("flat_topk_routes", []),
                "hybrid_topk_routes": hybrid.get("hybrid_topk_routes", []),
                "validation_match": validation_match,
                "conflicts": {
                    "topdown_vs_bottomup": _conflict_flag(
                        _candidate_code(top_best), _candidate_code(bottom_best)
                    ),
                    "topdown_vs_flat": _conflict_flag(
                        _candidate_code(top_best), _candidate_code(flat_best)
                    ),
                    "bottomup_vs_flat": _conflict_flag(
                        _candidate_code(bottom_best), _candidate_code(flat_best)
                    ),
                    "hybrid_vs_bottomup": _conflict_flag(
                        _candidate_code(hybrid_best), _candidate_code(bottom_best)
                    ),
                    "hybrid_vs_flat": _conflict_flag(
                        _candidate_code(hybrid_best), _candidate_code(flat_best)
                    ),
                },
                "topdown_best_route": top_route,
                "bottomup_best_route": bottom_route,
                "flat_best_route": flat_best_route,
                "hybrid_best_route": hybrid_route,
            }
        )

    return compared


def finalize_predictions(
    compared_results: List[Dict[str, Any]],
    taxonomy_embedded: Dict[str, Any],
    taxonomy_key: str,
    judge_model_name: str,
    encoder_model_name: str,
    debug_level: str = "low",
    min_confidence: float = 0.4,
    judge_enabled: bool = True,
) -> Dict[str, Any]:
    """Produce the final decision with judge escalation when needed.

    Args:
        compared_results: Results from compare_routes
        taxonomy_embedded: Partitioned taxonomy dataset
        taxonomy_key: Key to load the specific taxonomy
        judge_model_name: Model name for judge
        encoder_model_name: Model name for encoder
        debug_level: Output verbosity ("low", "medium", "high")
        min_confidence: Minimum confidence threshold for accepting predictions (Improvement 4)
        judge_enabled: Whether to use judge for conflict resolution (default: True)
    """

    # Load the specific taxonomy partition
    taxonomy_df = taxonomy_embedded[taxonomy_key]()

    # Build lookup tables once for O(1) taxonomy access
    code_to_row, code_to_parent = _build_taxonomy_lookups(taxonomy_df)

    # Route reconstruction cache to avoid rebuilding identical routes
    route_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    suggestions: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for record in compared_results:
        # Confidence thresholding (Improvement 4)
        # Filter low-confidence predictions to trigger judge escalation
        record = _apply_confidence_threshold(record, min_confidence)
        flat_topk = record.get("flat_topk_routes", [])

        judge_output = None
        classifier_model = encoder_model_name

        should_judge = judge_enabled and decide_if_judge_needed(record)

        final_decision = None
        final_confidence = None

        # Call judge if enabled and needed
        if should_judge:
            candidate_pool = _deduplicate_candidates(
                _extract_leaf_candidates(record.get("topdown_topk_routes", []))
                + _extract_leaf_candidates(record.get("bottomup_topk_routes", []))
                + _extract_leaf_candidates(flat_topk)
            )
            if candidate_pool:
                context_segments = [
                    cand.get("path_text")
                    for cand in candidate_pool[:5]
                    if cand.get("path_text")
                ]
                if context_segments:
                    taxonomy_context = "\n\n".join(context_segments)
                else:
                    taxonomy_context = taxonomy_utils.build_taxonomy_context(
                        taxonomy_df, candidate_pool[:5]
                    )
                judge_output = judge_utils.run_judge(
                    input_text=record.get("text", ""),
                    candidates=candidate_pool[:5],
                    taxonomy_context=taxonomy_context,
                    model_name=judge_model_name,
                    route=record.get("bottomup_best_route", []),
                )
                classifier_model = judge_model_name

                # If judge made a decision, build route to that node
                if judge_output and judge_output.get("decision"):
                    decision_node = judge_output["decision"]
                    final_decision = decision_node
                    final_confidence = float(decision_node.get("score", 0.0))

        if final_decision is None:
            consensus = _select_consensus_leaf(
                record.get("bottomup_best_leaf"),
                record.get("flat_best_leaf"),
            )
            if consensus is not None:
                final_decision = consensus
            else:
                final_decision = _select_highest_scoring_candidate(
                    [
                        record.get("topdown_best_leaf"),
                        record.get("bottomup_best_leaf"),
                        record.get("flat_best_leaf"),
                    ]
                )
            if final_decision is not None:
                final_confidence = float(final_decision.get("score", 0.0))

        if final_decision:
            # Check cache first to avoid rebuilding identical routes
            decision_code = _candidate_code(final_decision)
            cache_key = (taxonomy_key, decision_code) if decision_code else None

            if cache_key and cache_key in route_cache:
                selected_route = route_cache[cache_key]
            else:
                selected_route = _select_route_for_decision(
                    final_decision, record, taxonomy_df,
                    code_to_row=code_to_row, code_to_parent=code_to_parent
                )
                if cache_key:
                    route_cache[cache_key] = selected_route
        else:
            selected_route = []

        # Build annotations from the selected route
        annotations = selected_route

        if not annotations:
            annotations = _default_unknown_annotation()

        # Determine if judge intervened
        judge_intervened = judge_output is not None

        # Optimize debug output: strip unnecessary data based on debug level
        if debug_level == "low":
            # For "low" debug level, strip all topk routes to reduce conversion overhead
            topdown_topk_routes = []
            bottomup_topk_routes = []
            flat_topk_routes_output = []
        elif debug_level == "medium":
            # For "medium", limit to top 3 candidates and strip nested route details
            topdown_topk_routes = [
                {"leaf": route.get("leaf"), "score": route.get("score")}
                for route in record.get("topdown_topk_routes", [])[:3]
            ]
            bottomup_topk_routes = [
                {"leaf": route.get("leaf"), "score": route.get("score")}
                for route in record.get("bottomup_topk_routes", [])[:3]
            ]
            flat_topk_routes_output = [
                {"leaf": route.get("leaf"), "score": route.get("score")}
                for route in flat_topk[:3]
            ]
        else:
            # For "high", include full topk routes
            topdown_topk_routes = record.get("topdown_topk_routes", [])
            bottomup_topk_routes = record.get("bottomup_topk_routes", [])
            flat_topk_routes_output = flat_topk

        # Build output based on debug level
        output = _build_output_by_debug_level(
            sentence_id=record.get("sentenceId"),
            text=record.get("text"),
            annotations=annotations,
            validation_match=bool(record.get("validation_match")),
            conflicts=record.get("conflicts", {}),
            classifier_model=classifier_model,
            judge_intervened=judge_intervened,
            topdown_topk_routes=topdown_topk_routes,
            bottomup_topk_routes=bottomup_topk_routes,
            flat_topk_routes=flat_topk_routes_output,
            debug_level=debug_level,
            final_decision=final_decision,
            final_confidence=final_confidence,
            judge_output=judge_output,
        )

        suggestions.append(_to_native(output))

    return {"suggestions": suggestions, "errors": errors}


def _build_output_by_debug_level(
    sentence_id: str,
    text: str,
    annotations: List[Dict[str, Any]],
    validation_match: bool,
    conflicts: Dict[str, Any],
    classifier_model: str,
    judge_intervened: bool,
    topdown_topk_routes: List[Dict[str, Any]],
    bottomup_topk_routes: List[Dict[str, Any]],
    flat_topk_routes: List[Dict[str, Any]],
    debug_level: str,
    final_decision: Dict[str, Any] | None,
    final_confidence: float | None,
    judge_output: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Build output structure based on debug level.

    Low: sentenceId, annotations (without labels)
    Medium: + text, validation_match, conflicts, classifierModel,
            annotations with labels, judge_intervened
    High: All fields including topk data from all approaches
    """
    # Low level: minimal output
    simplified_annotations = [
        {
            "level": ann.get("level"),
            "nodeCode": ann.get("nodeCode"),
            "confidence": ann.get("confidence"),
        }
        for ann in annotations
    ]

    output = {
        "sentenceId": sentence_id,
        "annotations": simplified_annotations,
    }

    # Medium level: add more context
    if debug_level in ["medium", "high"]:
        output["text"] = text
        output["validation_match"] = validation_match
        output["conflicts"] = conflicts
        output["classifierModel"] = classifier_model
        output["judge_intervened"] = judge_intervened
        output["final_decision"] = final_decision
        output["final_confidence"] = final_confidence
        output["judge_output"] = judge_output
        # Include labels in annotations for medium and high
        output["annotations"] = [
            {
                "level": ann.get("level"),
                "nodeCode": ann.get("nodeCode"),
                "confidence": ann.get("confidence"),
                "label": ann.get("label"),
            }
            for ann in annotations
        ]

    # High level: include all diagnostic data
    if debug_level == "high":
        output["topdown_topk_routes"] = topdown_topk_routes
        output["bottomup_topk_routes"] = bottomup_topk_routes
        output["flat_topk_routes"] = flat_topk_routes

    return output


def decide_if_judge_needed(record: Dict[str, Any]) -> bool:
    """Determine whether the judge model should be invoked."""

    td = record.get("topdown_best_leaf")
    bu = record.get("bottomup_best_leaf")
    fl = record.get("flat_best_leaf")
    flat_topk_routes = record.get("flat_topk_routes", [])
    flat_topk = _extract_leaf_candidates(flat_topk_routes)

    td_code = _candidate_code(td)
    bu_code = _candidate_code(bu)
    fl_code = _candidate_code(fl)

    # 1. Strong consensus: bottom-up and flat agree
    if bu_code and bu_code == fl_code:
        return False

    # 2. Missing evidence triggers judge
    if not td_code or not bu_code or not fl_code:
        return True

    # 3. Flat dominance suppresses judge
    if score_dominance(flat_topk) > 0.10:
        return False

    # 4. Clear confidence gap suppresses judge
    scores = [td.get("score", 0.0), bu.get("score", 0.0), fl.get("score", 0.0)]
    scores_sorted = sorted(scores)
    if len(scores_sorted) >= 2 and (scores_sorted[-1] - scores_sorted[-2]) > 0.05:
        return False

    # 5. All predictions are within the same branch
    if taxonomy_utils.same_branch(td_code, bu_code) and taxonomy_utils.same_branch(
        td_code, fl_code
    ):
        return False

    # 6. Otherwise, call judge
    return True


def judge_resolution(
    input_text: str,
    validated_result: Dict[str, Any],
    taxonomy_embedded: pd.DataFrame,
    judge_model_name: str,
) -> Dict[str, Any]:
    """Compatibility wrapper used by service runners."""

    candidates = validated_result.get("scores", [])[:3]
    taxonomy_context = taxonomy_utils.build_taxonomy_context(
        taxonomy_embedded, candidates
    )
    decision = judge_utils.run_judge(
        input_text=input_text,
        candidates=candidates,
        taxonomy_context=taxonomy_context,
        model_name=judge_model_name,
        route=validated_result.get("route", []),
    )
    return {
        "validated": validated_result.get("routes_match", False),
        "judge_decision": decision,
    }


def _build_taxonomy_lookups(
    taxonomy: pd.DataFrame,
) -> Tuple[Dict[str, pd.Series], Dict[str, str | None]]:
    """Build hash maps for O(1) taxonomy lookups.

    Returns:
        - code_to_row: Direct access to taxonomy row by code
        - code_to_parent: Direct access to parent code
    """
    code_to_row: Dict[str, pd.Series] = {}
    code_to_parent: Dict[str, str | None] = {}

    for _, row in taxonomy.iterrows():
        code = row.get("code")
        if code:
            code_to_row[code] = row
            code_to_parent[code] = row.get("parentCode")

    return code_to_row, code_to_parent


def _build_topk_routes(
    candidates: List[Dict[str, Any]],
    taxonomy: pd.DataFrame,
    input_embedding: List[float],
) -> List[Dict[str, Any]]:
    """Attach full routes to every Top-K candidate leaf."""

    topk_routes: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not candidate:
            continue
        route_annotations = _build_route_annotations_for_leaf(
            candidate, taxonomy, input_embedding
        )
        topk_routes.append(
            {
                "leaf": candidate,
                "score": float(candidate.get("score", 0.0)),
                "route": route_annotations,
            }
        )
    return topk_routes


def _build_route_annotations_for_leaf(
    leaf: Dict[str, Any] | None,
    taxonomy: pd.DataFrame,
    input_embedding: List[float] | None,
    code_to_row: Dict[str, pd.Series] | None = None,
    code_to_parent: Dict[str, str | None] | None = None,
) -> List[Dict[str, Any]]:
    """Construct annotation-style routes using taxonomy embeddings."""

    route_codes = _extract_route_codes(leaf, taxonomy, code_to_parent)
    if not route_codes:
        return []

    rows: List[pd.Series] = []
    # Use lookup table if available for O(1) row access
    if code_to_row is not None:
        for code in route_codes:
            if code is None:
                continue
            row = code_to_row.get(code)
            if row is not None:
                rows.append(row)
    else:
        # Fallback to pandas filtering
        for code in route_codes:
            if code is None:
                continue
            match = taxonomy[taxonomy["code"] == code]
            if match.empty:
                continue
            rows.append(match.iloc[0])

    if not rows:
        return []

    scored_lookup: Dict[str, Dict[str, Any]] = {}
    if input_embedding is not None:
        route_df = pd.DataFrame(rows)
        scored_nodes = scoring_utils.score_candidates(
            input_embedding, route_df
        )
        scored_lookup = {node.get("code"): node for node in scored_nodes}

    annotations: List[Dict[str, Any]] = []
    for code in route_codes:
        if code is None:
            continue
        node = scored_lookup.get(code)
        if not node:
            replacement = next(
                (row for row in rows if row.get("code") == code), None
            )
            if replacement is None:
                continue
            node = taxonomy_utils.row_to_candidate(replacement)
            node["score"] = float(leaf.get("score", 0.0)) if leaf else 0.0
        annotations.append(
            {
                "level": int(node.get("level", -1)),
                "nodeCode": node.get("code"),
                "confidence": float(node.get("score", 0.0)),
                "label": node.get("label"),
            }
        )
    return annotations


def _extract_route_codes(
    leaf: Dict[str, Any] | None,
    taxonomy: pd.DataFrame,
    code_to_parent: Dict[str, str | None] | None = None,
) -> List[str]:
    """Trace parent relationships to produce a root-to-leaf code path."""

    if not leaf:
        return []

    path_nodes = leaf.get("path_nodes") or []
    if path_nodes:
        return [
            node.get("code")
            for node in path_nodes
            if node.get("code") is not None
        ]

    codes: List[str] = []
    visited: set[str] = set()
    current_code = leaf.get("leaf_code") or leaf.get("code")

    # Use lookup table if available for O(1) parent access
    if code_to_parent is not None:
        while current_code and current_code not in visited:
            visited.add(current_code)
            codes.insert(0, current_code)
            current_code = code_to_parent.get(current_code)
    else:
        # Fallback to pandas filtering
        while current_code and current_code not in visited:
            visited.add(current_code)
            codes.insert(0, current_code)
            parent_row = taxonomy[taxonomy["code"] == current_code]
            if parent_row.empty:
                break
            current_code = parent_row.iloc[0].get("parentCode")

    return codes


def _build_route_annotations_from_path_nodes(
    path_nodes: List[Dict[str, Any]] | None, base_score: float
) -> List[Dict[str, Any]]:
    """Convert stored path nodes into annotation-style output."""

    annotations: List[Dict[str, Any]] = []
    if path_nodes is None:
        return annotations

    for node in path_nodes:
        annotations.append(
            {
                "level": int(node.get("level", -1)),
                "nodeCode": node.get("code"),
                "confidence": float(base_score),
                "label": node.get("label"),
            }
        )
    return annotations


def _build_route_annotations(
    route: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build annotations from a route (one best node per level)."""
    annotations: List[Dict[str, Any]] = []
    for node in route or []:
        annotations.append(
            {
                "level": int(node.get("level", -1)),
                "nodeCode": node.get("code"),
                "confidence": float(node.get("score", 0.0)),
                "label": node.get("label"),
            }
        )
    return annotations


def _build_annotations(
    levels: List[Dict[str, Any]], approach: str
) -> List[Dict[str, Any]]:
    annotations: List[Dict[str, Any]] = []
    for entry in levels or []:
        level = entry.get("level")
        for candidate in entry.get("topk", []) or []:
            annotations.append(
                {
                    "approach": approach,
                    "level": int(level) if level is not None else -1,
                    "nodeCode": candidate.get("code"),
                    "confidence": float(candidate.get("score", 0.0)),
                    "label": candidate.get("label"),
                }
            )
    return annotations


def _default_unknown_annotation() -> List[Dict[str, Any]]:
    return [
        {
            "approach": "topdown",
            "level": 1,
            "nodeCode": "-99",
            "confidence": 0.0,
            "label": "unknown",
        }
    ]


def _normalize_annotations(
    route: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Ensure routes are expressed in annotation format."""

    if not route:
        return []
    first = route[0] or {}
    if "nodeCode" in first:
        return route
    return _build_route_annotations(route)


def _deduplicate_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for candidate in candidates:
        code = candidate.get("code")
        if not code or code in seen:
            continue
        seen.add(code)
        deduped.append(candidate)
    return deduped


def _extract_leaf_candidates(
    topk_routes: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Flatten Top-K route entries into candidate leaf dictionaries."""

    candidates: List[Dict[str, Any]] = []
    for entry in topk_routes or []:
        leaf = entry.get("leaf")
        if not leaf:
            continue
        enriched = dict(leaf)
        if "score" not in enriched and entry.get("score") is not None:
            enriched["score"] = float(entry.get("score", 0.0))
        candidates.append(enriched)
    return candidates


def _select_consensus_leaf(
    bottom_leaf: Dict[str, Any] | None,
    flat_leaf: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    """Return the consensus leaf when bottom-up and flat agree."""

    bottom_code = _candidate_code(bottom_leaf)
    flat_code = _candidate_code(flat_leaf)
    if bottom_code and flat_code and bottom_code == flat_code:
        return bottom_leaf
    return None


def _select_highest_scoring_candidate(
    candidates: List[Dict[str, Any] | None],
) -> Dict[str, Any] | None:
    """Choose the candidate leaf with the highest confidence score."""

    best_candidate: Dict[str, Any] | None = None
    best_score = float("-inf")
    for candidate in candidates:
        if not candidate:
            continue
        score = _candidate_score(candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate


def _candidate_score(candidate: Dict[str, Any] | None) -> float:
    if not candidate:
        return 0.0
    return float(candidate.get("score", 0.0))


def _select_route_for_decision(
    decision: Dict[str, Any] | None,
    record: Dict[str, Any],
    taxonomy: pd.DataFrame,
    code_to_row: Dict[str, pd.Series] | None = None,
    code_to_parent: Dict[str, str | None] | None = None,
) -> List[Dict[str, Any]]:
    """Build annotations for the final decision route with per-level confidence scores."""

    if not decision:
        return []

    if decision.get("path_nodes"):
        base_route = _build_route_annotations(decision.get("path_nodes"))
        # Use the leaf score for all levels since path_nodes come from judge
        # which doesn't have per-level scores
        return _apply_confidence_to_route(base_route, _candidate_score(decision))

    # Try to reuse existing routes that already have per-level scores
    code = _candidate_code(decision)
    base_route = []
    candidate_routes = [
        record.get("topdown_best_route"),
        record.get("bottomup_best_route"),
        record.get("flat_best_route"),
    ]
    for route in candidate_routes:
        annotations = _normalize_annotations(route)
        if annotations and code and _route_matches_code(annotations, code):
            # Found a matching route with per-level scores - use it!
            base_route = annotations
            break

    if not base_route:
        # Fallback: build route without input embedding (will use leaf score for all levels)
        base_route = _build_route_annotations_for_leaf(
            decision, taxonomy, input_embedding=None,
            code_to_row=code_to_row, code_to_parent=code_to_parent
        )
        # Apply leaf confidence since we couldn't compute per-level scores
        return _apply_confidence_to_route(base_route, _candidate_score(decision))

    # Return the found route with its original per-level confidence scores
    return base_route


def _route_matches_code(route: List[Dict[str, Any]], code: str) -> bool:
    """Check that a route ends in the provided node code."""

    if not route:
        return False
    last = route[-1] or {}
    return last.get("nodeCode") == code


def _apply_confidence_to_route(
    route: List[Dict[str, Any]], confidence: float
) -> List[Dict[str, Any]]:
    """Override route confidence values with the final decision score."""

    annotations = _normalize_annotations(route)
    if not annotations:
        return []
    return [
        {
            "level": ann.get("level"),
            "nodeCode": ann.get("nodeCode"),
            "confidence": float(confidence),
            "label": ann.get("label"),
        }
        for ann in annotations
    ]


def _conflict_flag(left_code: str | None, right_code: str | None) -> bool | None:
    if not left_code or not right_code:
        return None
    return left_code != right_code


def _candidate_code(candidate: Dict[str, Any] | None) -> str | None:
    if not candidate:
        return None
    return candidate.get("leaf_code") or candidate.get("code")


def score_dominance(scores: List[Dict[str, Any]]) -> float:
    if not scores:
        return 0.0
    if len(scores) == 1:
        return float(scores[0].get("score", 0.0))
    return float(scores[0].get("score", 0.0)) - float(
        scores[1].get("score", 0.0)
    )


def _apply_confidence_threshold(
    record: Dict[str, Any], min_confidence: float
) -> Dict[str, Any]:
    """Filter low-confidence predictions to trigger judge escalation.

    Args:
        record: Comparison record with results from all three methods
        min_confidence: Minimum acceptable confidence score

    Returns:
        Modified record with low-confidence predictions set to None
    """
    # Check and filter top-down
    td_score = record.get("topdown_best_leaf_score")
    if td_score is not None and td_score < min_confidence:
        record["topdown_best_leaf"] = None
        record["topdown_best_leaf_score"] = None

    # Check and filter bottom-up
    bu_score = record.get("bottomup_best_leaf_score")
    if bu_score is not None and bu_score < min_confidence:
        record["bottomup_best_leaf"] = None
        record["bottomup_best_leaf_score"] = None

    # Check and filter flat
    flat_score = record.get("flat_best_leaf_score")
    if flat_score is not None and flat_score < min_confidence:
        record["flat_best_leaf"] = None
        record["flat_best_leaf_score"] = None

    return record


def _rerank_by_sibling_diversity(
    candidates: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """Re-rank candidates to prefer diverse parents (avoid all siblings from one parent).

    Args:
        candidates: List of scored candidate nodes
        top_k: Number of top candidates to return

    Returns:
        Re-ranked list prioritizing parent diversity
    """
    if not candidates or len(candidates) <= top_k:
        return candidates[:top_k]

    # Group by parent
    parent_groups: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        parent = candidate.get("parentCode")
        if parent not in parent_groups:
            parent_groups[parent] = []
        parent_groups[parent].append(candidate)

    # Re-rank: First pass - take best from each parent
    reranked: List[Dict[str, Any]] = []
    used_parents: set[str] = set()

    # Sort parent groups by their best candidate's score
    sorted_parents = sorted(
        parent_groups.items(),
        key=lambda x: max(s.get("score", 0.0) for s in x[1]),
        reverse=True
    )

    for parent, siblings in sorted_parents:
        if parent not in used_parents:
            best_sibling = max(siblings, key=lambda x: x.get("score", 0.0))
            reranked.append(best_sibling)
            used_parents.add(parent)
            if len(reranked) >= top_k:
                break

    # Second pass: Fill remaining slots with highest scores
    if len(reranked) < top_k:
        for candidate in candidates:
            if len(reranked) >= top_k:
                break
            if candidate not in reranked:
                reranked.append(candidate)

    return reranked[:top_k]


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to native types."""

    # Early returns for common primitive types (optimization)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {key: _to_native(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_native(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_to_native(item) for item in obj)
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def top_down_with_confidence_gating(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    min_confidence_level1: float = 0.2,
    stop_on_decrease: bool = True,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Top-down classification with adaptive confidence gating.

    Stops when similarity decreases from previous level (indicating over-specification).

    Args:
        input_embedding: Input text embedding vector
        taxonomy_embedded: Full taxonomy with embeddings
        min_confidence_level1: Minimum confidence to accept level 1 (default: 0.2)
                              This ensures we at least match something at the top level
        stop_on_decrease: If True, stop when confidence decreases from previous level
                         If False, continue even when confidence decreases
        top_k: Number of top candidates to return

    Returns:
        Dict with:
            - best_leaf: Best classification found
            - best_leaf_score: Confidence score
            - stopped_at_level: Level where classification stopped
            - stop_reason: Why classification stopped
            - best_route: Full path to best classification
            - confidence_per_level: Confidence at each level
            - topk: Top-k alternatives
    """
    best_nodes: List[Dict[str, Any]] = []
    confidence_per_level: List[float] = []
    parent_code: str | None = None
    previous_confidence: float | None = None
    stopped_at_level: int | None = None
    stop_reason: str = "reached_leaf"

    max_level = int(taxonomy_embedded["level"].max()) if not taxonomy_embedded.empty else 0

    for level in range(1, max_level + 1):
        level_nodes = taxonomy_utils.get_level_nodes(
            taxonomy_embedded, level, parent_code
        )
        if level_nodes.empty:
            stopped_at_level = level - 1
            stop_reason = "no_children"
            break

        scored = scoring_utils.score_candidates(input_embedding, level_nodes)
        if not scored:
            stopped_at_level = level - 1
            stop_reason = "no_scores"
            break

        best = scored[0]
        current_confidence = float(best.get("score", 0.0))

        # Level 1: Apply minimum confidence threshold
        if level == 1 and current_confidence < min_confidence_level1:
            stopped_at_level = 0
            stop_reason = f"level1_confidence_too_low_{current_confidence:.3f}"
            break

        # Level 2+: Check if confidence decreased from previous level
        if level > 1 and previous_confidence is not None and stop_on_decrease:
            if current_confidence < previous_confidence:
                # Confidence decreased - we've gone too specific
                stopped_at_level = level - 1
                confidence_drop = previous_confidence - current_confidence
                stop_reason = f"confidence_decreased_by_{confidence_drop:.3f}"
                break

        # Accept this level
        best_nodes.append(best)
        confidence_per_level.append(current_confidence)
        previous_confidence = current_confidence
        parent_code = best.get("code")

        if parent_code is None:
            stopped_at_level = level
            stop_reason = "invalid_parent_code"
            break

        # Check if this is a leaf node
        if best.get("isLeaf"):
            stopped_at_level = level
            stop_reason = "reached_leaf"
            break

    # If we didn't set stopped_at_level, we completed all levels
    if stopped_at_level is None:
        stopped_at_level = len(best_nodes)

    best_leaf = best_nodes[-1] if best_nodes else None
    best_route = _build_route_annotations(best_nodes)

    # Get top-k alternatives at the stopped level
    leaves = taxonomy_embedded[taxonomy_embedded["isLeaf"].astype(bool)]
    leaf_scores = scoring_utils.score_candidates(input_embedding, leaves)
    topk_candidates = leaf_scores[:top_k]

    # Attach path_nodes to each candidate
    for candidate in topk_candidates:
        if candidate and not candidate.get("path_nodes"):
            parent_codes = taxonomy_utils.get_parent_chain(
                taxonomy_embedded, candidate.get("code")
            )
            path_nodes: List[Dict[str, Any]] = []
            for code in parent_codes:
                node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
                if not node_row.empty:
                    path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))
            leaf_node = dict(candidate)
            if "path_nodes" in leaf_node:
                del leaf_node["path_nodes"]
            path_nodes.append(leaf_node)
            candidate["path_nodes"] = path_nodes

    topk_routes = _build_topk_routes(topk_candidates, taxonomy_embedded, input_embedding)

    best_leaf_score = float(best_leaf.get("score", 0.0)) if best_leaf else None

    return {
        "best_leaf": best_leaf,
        "best_leaf_score": best_leaf_score,
        "stopped_at_level": stopped_at_level,
        "stop_reason": stop_reason,
        "best_route": best_route,
        "confidence_per_level": confidence_per_level,
        "topk": topk_routes,
    }


def bottom_up_path_aggregation(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    aggregation_method: str = "mean_top_k",
    top_k_children: int = 3,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Bottom-up classification by aggregating leaf scores up the hierarchy.

    Args:
        input_embedding: Input text embedding vector
        taxonomy_embedded: Full taxonomy with embeddings
        aggregation_method: How to aggregate children scores:
            - "max": Take maximum child score
            - "mean": Average all children scores
            - "mean_top_k": Average top-k children scores (default)
        top_k_children: Number of top children to consider for mean_top_k (default: 3)
        top_k: Number of top candidates to return

    Returns:
        Dict with:
            - best_leaf: Best leaf classification
            - best_at_each_level: Best classification at each level
            - aggregated_scores_per_level: Aggregated scores for each level
            - best_route: Full path to best classification
            - topk: Top-k alternatives
    """
    # Step 1: Score all nodes at all levels
    all_scores_by_code: Dict[str, float] = {}
    all_nodes_by_code: Dict[str, Dict[str, Any]] = {}

    # First pass: score all nodes directly
    for _, node in taxonomy_embedded.iterrows():
        node_dict = taxonomy_utils.row_to_candidate(node)
        node_embedding = node.get("embedding")

        if node_embedding is None or not isinstance(node_embedding, (list, np.ndarray)):
            continue

        node_score = float(
            np.dot(input_embedding, node_embedding) /
            (np.linalg.norm(input_embedding) * np.linalg.norm(node_embedding))
        )

        code = node.get("code")
        all_scores_by_code[code] = node_score
        all_nodes_by_code[code] = {
            **node_dict,
            "direct_score": node_score,
            "score": node_score,  # Will be updated with aggregated score
        }

    # Step 2: Build parent-to-children mapping
    children_by_parent: Dict[str, List[str]] = {}
    for _, node in taxonomy_embedded.iterrows():
        parent_code = node.get("parentCode")
        code = node.get("code")
        if parent_code and code:
            if parent_code not in children_by_parent:
                children_by_parent[parent_code] = []
            children_by_parent[parent_code].append(code)

    # Step 3: Aggregate scores bottom-up (from max level to level 1)
    max_level = int(taxonomy_embedded["level"].max()) if not taxonomy_embedded.empty else 0
    aggregated_scores_by_code: Dict[str, float] = dict(all_scores_by_code)

    for level in range(max_level, 0, -1):
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == level]

        for _, node in level_nodes.iterrows():
            code = node.get("code")
            if not code:
                continue

            # Get children scores
            children_codes = children_by_parent.get(code, [])
            if not children_codes:
                # Leaf node or no children, keep direct score
                continue

            children_scores = [
                aggregated_scores_by_code.get(child_code, 0.0)
                for child_code in children_codes
            ]

            # Aggregate based on method
            if aggregation_method == "max":
                aggregated_score = max(children_scores) if children_scores else 0.0
            elif aggregation_method == "mean":
                aggregated_score = float(np.mean(children_scores)) if children_scores else 0.0
            elif aggregation_method == "mean_top_k":
                sorted_scores = sorted(children_scores, reverse=True)
                top_scores = sorted_scores[:top_k_children]
                aggregated_score = float(np.mean(top_scores)) if top_scores else 0.0
            else:
                aggregated_score = aggregated_scores_by_code.get(code, 0.0)

            # Update aggregated score
            aggregated_scores_by_code[code] = aggregated_score
            if code in all_nodes_by_code:
                all_nodes_by_code[code]["aggregated_score"] = aggregated_score
                all_nodes_by_code[code]["score"] = aggregated_score

    # Step 4: Find best classification at each level
    best_at_each_level: Dict[int, Dict[str, Any]] = {}
    for level in range(1, max_level + 1):
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == level]
        level_candidates = []

        for _, node in level_nodes.iterrows():
            code = node.get("code")
            if code and code in all_nodes_by_code:
                level_candidates.append(all_nodes_by_code[code])

        if level_candidates:
            best_at_level = max(level_candidates, key=lambda x: x.get("score", 0.0))
            best_at_each_level[level] = best_at_level

    # Step 5: Get best leaf (highest aggregated score among leaves)
    leaves = taxonomy_embedded[taxonomy_embedded["isLeaf"].astype(bool)]
    leaf_candidates = []
    for _, leaf in leaves.iterrows():
        code = leaf.get("code")
        if code and code in all_nodes_by_code:
            candidate = all_nodes_by_code[code]
            # Attach path_nodes
            if not candidate.get("path_nodes"):
                parent_codes = taxonomy_utils.get_parent_chain(taxonomy_embedded, code)
                path_nodes: List[Dict[str, Any]] = []
                for parent_code in parent_codes:
                    if parent_code in all_nodes_by_code:
                        path_nodes.append(all_nodes_by_code[parent_code])
                path_nodes.append(candidate)
                candidate["path_nodes"] = path_nodes
            leaf_candidates.append(candidate)

    # Sort by aggregated score
    leaf_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    topk_candidates = leaf_candidates[:top_k]

    best_leaf = topk_candidates[0] if topk_candidates else None
    best_route = _build_route_annotations_for_leaf(
        best_leaf, taxonomy_embedded, input_embedding
    )
    topk_routes = _build_topk_routes(topk_candidates, taxonomy_embedded, input_embedding)

    best_leaf_score = float(best_leaf.get("score", 0.0)) if best_leaf else None

    return {
        "best_leaf": best_leaf,
        "best_leaf_score": best_leaf_score,
        "best_at_each_level": best_at_each_level,
        "aggregated_scores_per_level": {
            level: float(node.get("score", 0.0))
            for level, node in best_at_each_level.items()
        },
        "best_route": best_route,
        "topk": topk_routes,
    }


def bottom_up_confidence_climbing(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    stop_on_decrease: bool = True,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Bottom-up classification by climbing from best leaf to optimal abstraction level.

    Starts at the best matching leaf and climbs up the hierarchy as long as
    the parent has a higher similarity score than the current node.
    Stops when score decreases (indicating we've found the right abstraction level).

    Args:
        input_embedding: Input text embedding vector
        taxonomy_embedded: Full taxonomy with embeddings
        stop_on_decrease: If True, stop when confidence decreases from child to parent
                         If False, continue to root regardless of score changes
        top_k: Number of top candidates to return

    Returns:
        Dict with:
            - best_leaf: Best classification found (may not be a leaf!)
            - best_leaf_score: Confidence score
            - stopped_at_level: Level where classification stopped
            - stop_reason: Why classification stopped
            - best_route: Full path to best classification
            - confidence_per_level: Confidence at each level (from leaf to root)
            - topk: Top-k alternatives
    """
    # Step 1: Find the best matching leaf
    leaves = taxonomy_embedded[taxonomy_embedded["isLeaf"].astype(bool)]
    leaf_scores = scoring_utils.score_candidates(input_embedding, leaves)

    if not leaf_scores:
        return {
            "best_leaf": None,
            "best_leaf_score": None,
            "stopped_at_level": 0,
            "stop_reason": "no_leaf_scores",
            "best_route": [],
            "confidence_per_level": [],
            "topk": [],
        }

    best_leaf_candidate = leaf_scores[0]
    current_node = best_leaf_candidate
    current_code = current_node.get("code")
    current_confidence = float(current_node.get("score", 0.0))
    current_level = int(current_node.get("level", 0))

    # Step 2: Build the climbing path (from leaf upwards)
    climbing_nodes: List[Dict[str, Any]] = [current_node]
    confidence_per_level: List[float] = [current_confidence]
    stopped_at_level = current_level
    stop_reason = "reached_root"

    # Step 3: Climb up the hierarchy
    while current_level > 1:
        # Get parent
        parent_code = current_node.get("parentCode")
        if not parent_code or parent_code == "__root__":
            stop_reason = "reached_root"
            break

        # Find parent node in taxonomy
        parent_row = taxonomy_embedded[taxonomy_embedded["code"] == parent_code]
        if parent_row.empty:
            stop_reason = "parent_not_found"
            break

        parent_node = taxonomy_utils.row_to_candidate(parent_row.iloc[0])
        parent_embedding = parent_row.iloc[0].get("embedding")

        if parent_embedding is None or not isinstance(parent_embedding, (list, np.ndarray)):
            stop_reason = "parent_no_embedding"
            break

        # Calculate parent's direct similarity score
        parent_confidence = float(
            np.dot(input_embedding, parent_embedding) /
            (np.linalg.norm(input_embedding) * np.linalg.norm(parent_embedding))
        )
        parent_node["score"] = parent_confidence
        parent_level = int(parent_node.get("level", 0))

        # Check if we should continue climbing
        if stop_on_decrease and parent_confidence < current_confidence:
            # Parent has lower score - stop here at current level
            confidence_drop = current_confidence - parent_confidence
            stop_reason = f"confidence_decreased_by_{confidence_drop:.3f}"
            break

        # Accept this parent level - continue climbing
        climbing_nodes.append(parent_node)
        confidence_per_level.append(parent_confidence)
        stopped_at_level = parent_level
        current_node = parent_node
        current_code = parent_code
        current_confidence = parent_confidence
        current_level = parent_level

    # Step 4: The best classification is the last accepted node
    best_leaf = climbing_nodes[-1] if climbing_nodes else None

    # Build route (need to reverse climbing_nodes to go from root to final node)
    route_nodes = list(reversed(climbing_nodes))
    best_route = _build_route_annotations(route_nodes)

    # Step 5: Get top-k alternatives at the stopped level
    if stopped_at_level > 0:
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == stopped_at_level]
        level_scores = scoring_utils.score_candidates(input_embedding, level_nodes)
        topk_candidates = level_scores[:top_k]

        # Attach path_nodes to each candidate
        for candidate in topk_candidates:
            if candidate and not candidate.get("path_nodes"):
                parent_codes = taxonomy_utils.get_parent_chain(
                    taxonomy_embedded, candidate.get("code")
                )
                path_nodes: List[Dict[str, Any]] = []
                for code in parent_codes:
                    node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
                    if not node_row.empty:
                        path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))
                leaf_node = dict(candidate)
                if "path_nodes" in leaf_node:
                    del leaf_node["path_nodes"]
                path_nodes.append(leaf_node)
                candidate["path_nodes"] = path_nodes

        topk_routes = _build_topk_routes(topk_candidates, taxonomy_embedded, input_embedding)
    else:
        topk_routes = []

    best_leaf_score = float(best_leaf.get("score", 0.0)) if best_leaf else None

    return {
        "best_leaf": best_leaf,
        "best_leaf_score": best_leaf_score,
        "stopped_at_level": stopped_at_level,
        "stop_reason": stop_reason,
        "best_route": best_route,
        "confidence_per_level": confidence_per_level,
        "topk": topk_routes,
    }


def multi_level_simultaneous_classification(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Classify at all levels simultaneously and return classification at the level with highest confidence.

    Args:
        input_embedding: Input text embedding vector
        taxonomy_embedded: Full taxonomy with embeddings
        top_k: Number of top candidates to return

    Returns:
        Dict with:
            - best_leaf: Best classification (may not be at leaf level)
            - best_leaf_score: Confidence score
            - best_level: Level where best match was found
            - best_at_each_level: Best match at each level with scores
            - best_route: Full path to best classification
            - topk: Top-k alternatives (at the best level)
    """
    max_level = int(taxonomy_embedded["level"].max()) if not taxonomy_embedded.empty else 0

    # Step 1: Find best match at each level
    best_at_each_level: Dict[int, Dict[str, Any]] = {}

    for level in range(1, max_level + 1):
        # Get ALL nodes at this level (not filtered by parent)
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == level]
        if level_nodes.empty:
            continue

        scored = scoring_utils.score_candidates(input_embedding, level_nodes)
        if scored:
            best_at_level = scored[0]
            best_at_each_level[level] = best_at_level

    # Step 2: Find the level with the highest confidence
    best_level = None
    best_node = None
    best_score = float("-inf")

    for level, node in best_at_each_level.items():
        score = float(node.get("score", 0.0))
        if score > best_score:
            best_score = score
            best_level = level
            best_node = node

    # Step 3: Build path to the best node
    if best_node and best_level:
        # Get parent chain for the best node
        parent_codes = taxonomy_utils.get_parent_chain(
            taxonomy_embedded, best_node.get("code")
        )
        path_nodes: List[Dict[str, Any]] = []

        # Build path from root to best node
        for code in parent_codes:
            node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
            if not node_row.empty:
                path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))

        # Add the best node itself
        best_node_copy = dict(best_node)
        if "path_nodes" in best_node_copy:
            del best_node_copy["path_nodes"]
        path_nodes.append(best_node_copy)
        best_node["path_nodes"] = path_nodes

        best_route = _build_route_annotations_for_leaf(
            best_node, taxonomy_embedded, input_embedding
        )
    else:
        best_route = []

    # Step 4: Get top-k alternatives at the best level
    topk_routes = []
    if best_level:
        # Get ALL nodes at the best level (not filtered by parent)
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == best_level]
        scored = scoring_utils.score_candidates(input_embedding, level_nodes)
        topk_candidates = scored[:top_k]

        # Attach path_nodes to each candidate
        for candidate in topk_candidates:
            if candidate and not candidate.get("path_nodes"):
                parent_codes = taxonomy_utils.get_parent_chain(
                    taxonomy_embedded, candidate.get("code")
                )
                path_nodes: List[Dict[str, Any]] = []
                for code in parent_codes:
                    node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
                    if not node_row.empty:
                        path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))
                candidate_copy = dict(candidate)
                if "path_nodes" in candidate_copy:
                    del candidate_copy["path_nodes"]
                path_nodes.append(candidate_copy)
                candidate["path_nodes"] = path_nodes

        topk_routes = _build_topk_routes(topk_candidates, taxonomy_embedded, input_embedding)

    return {
        "best_leaf": best_node,
        "best_leaf_score": best_score if best_score != float("-inf") else None,
        "best_level": best_level,
        "best_at_each_level": best_at_each_level,
        "best_route": best_route,
        "topk": topk_routes,
    }


def multi_level_simultaneous_with_path_context(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Multi-level classification using path-averaged embeddings.

    For each node, computes similarity using the average of:
    - The node's own embedding
    - All parent embeddings up to the root

    This incorporates hierarchical context into scoring, making nodes at different
    levels more comparable by considering their full ancestry.

    Args:
        input_embedding: Input text embedding vector
        taxonomy_embedded: Full taxonomy with embeddings
        top_k: Number of top candidates to return

    Returns:
        Dict with:
            - best_leaf: Best classification (may not be at leaf level)
            - best_leaf_score: Confidence score (using path-averaged embedding)
            - best_level: Level where best match was found
            - best_at_each_level: Best match at each level with scores
            - best_route: Full path to best classification
            - topk: Top-k alternatives (at the best level)
    """
    max_level = int(taxonomy_embedded["level"].max()) if not taxonomy_embedded.empty else 0

    # Cache for path-averaged embeddings
    path_avg_embeddings: Dict[str, np.ndarray] = {}

    def get_path_averaged_embedding(node_code: str) -> np.ndarray | None:
        """Get the average embedding of a node and all its parents."""
        if node_code in path_avg_embeddings:
            return path_avg_embeddings[node_code]

        # Get parent chain
        parent_codes = taxonomy_utils.get_parent_chain(taxonomy_embedded, node_code)

        # Collect embeddings from parents and the node itself
        embeddings_to_avg = []

        # Add parent embeddings
        for parent_code in parent_codes:
            parent_row = taxonomy_embedded[taxonomy_embedded["code"] == parent_code]
            if not parent_row.empty:
                parent_emb = parent_row.iloc[0].get("embedding")
                if parent_emb is not None and isinstance(parent_emb, (list, np.ndarray)):
                    embeddings_to_avg.append(np.array(parent_emb))

        # Add node's own embedding
        node_row = taxonomy_embedded[taxonomy_embedded["code"] == node_code]
        if not node_row.empty:
            node_emb = node_row.iloc[0].get("embedding")
            if node_emb is not None and isinstance(node_emb, (list, np.ndarray)):
                embeddings_to_avg.append(np.array(node_emb))

        if not embeddings_to_avg:
            return None

        # Compute average
        avg_embedding = np.mean(embeddings_to_avg, axis=0)
        path_avg_embeddings[node_code] = avg_embedding
        return avg_embedding

    # Step 1: Find best match at each level using path-averaged embeddings
    best_at_each_level: Dict[int, Dict[str, Any]] = {}

    for level in range(1, max_level + 1):
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == level]
        if level_nodes.empty:
            continue

        level_candidates = []
        for _, node in level_nodes.iterrows():
            node_code = node.get("code")
            if not node_code:
                continue

            # Get path-averaged embedding
            path_avg_emb = get_path_averaged_embedding(node_code)
            if path_avg_emb is None:
                continue

            # Calculate similarity using path-averaged embedding
            score = float(
                np.dot(input_embedding, path_avg_emb) /
                (np.linalg.norm(input_embedding) * np.linalg.norm(path_avg_emb))
            )

            candidate = taxonomy_utils.row_to_candidate(node)
            candidate["score"] = score
            candidate["path_context_score"] = score  # Mark this as using path context
            level_candidates.append(candidate)

        if level_candidates:
            best_at_level = max(level_candidates, key=lambda x: x.get("score", 0.0))
            best_at_each_level[level] = best_at_level

    # Step 2: Find the level with the highest confidence
    best_level = None
    best_node = None
    best_score = float("-inf")

    for level, node in best_at_each_level.items():
        score = float(node.get("score", 0.0))
        if score > best_score:
            best_score = score
            best_level = level
            best_node = node

    # Step 3: Build path to the best node
    if best_node and best_level:
        parent_codes = taxonomy_utils.get_parent_chain(
            taxonomy_embedded, best_node.get("code")
        )
        path_nodes: List[Dict[str, Any]] = []

        for code in parent_codes:
            node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
            if not node_row.empty:
                path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))

        best_node_copy = dict(best_node)
        if "path_nodes" in best_node_copy:
            del best_node_copy["path_nodes"]
        path_nodes.append(best_node_copy)
        best_node["path_nodes"] = path_nodes

        best_route = _build_route_annotations_for_leaf(
            best_node, taxonomy_embedded, input_embedding
        )
    else:
        best_route = []

    # Step 4: Get top-k alternatives at the best level (using path-averaged embeddings)
    topk_routes = []
    if best_level:
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == best_level]
        level_candidates = []

        for _, node in level_nodes.iterrows():
            node_code = node.get("code")
            if not node_code:
                continue

            path_avg_emb = get_path_averaged_embedding(node_code)
            if path_avg_emb is None:
                continue

            score = float(
                np.dot(input_embedding, path_avg_emb) /
                (np.linalg.norm(input_embedding) * np.linalg.norm(path_avg_emb))
            )

            candidate = taxonomy_utils.row_to_candidate(node)
            candidate["score"] = score
            level_candidates.append(candidate)

        level_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        topk_candidates = level_candidates[:top_k]

        # Attach path_nodes to each candidate
        for candidate in topk_candidates:
            if candidate and not candidate.get("path_nodes"):
                parent_codes = taxonomy_utils.get_parent_chain(
                    taxonomy_embedded, candidate.get("code")
                )
                path_nodes: List[Dict[str, Any]] = []
                for code in parent_codes:
                    node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
                    if not node_row.empty:
                        path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))
                candidate_copy = dict(candidate)
                if "path_nodes" in candidate_copy:
                    del candidate_copy["path_nodes"]
                path_nodes.append(candidate_copy)
                candidate["path_nodes"] = path_nodes

        topk_routes = _build_topk_routes(topk_candidates, taxonomy_embedded, input_embedding)

    return {
        "best_leaf": best_node,
        "best_leaf_score": best_score if best_score != float("-inf") else None,
        "best_level": best_level,
        "best_at_each_level": best_at_each_level,
        "best_route": best_route,
        "topk": topk_routes,
    }


def bottom_up_with_disagreement_detection(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    top_k: int = 5,
    disagreement_threshold: int = 3,
) -> Dict[str, Any]:
    """
    Bottom-up classification with disagreement detection.

    Starts at the deepest level (leaves) and moves up the hierarchy when top-K candidates
    are scattered across multiple different parents, indicating the query is too abstract
    for that level.

    Algorithm:
    1. Start at level 4 (leaves) and get top-K candidates
    2. Check parent diversity: if top-K candidates come from disagreement_threshold+ different parents → disagreement detected
    3. Move up to level 3 and repeat the check
    4. Continue until finding a level where top-K candidates show parent consensus
    5. Return classification at the consensus level

    Args:
        input_embedding: Input text embedding vector
        taxonomy_embedded: Full taxonomy with embeddings
        top_k: Number of top candidates to consider for disagreement detection
        disagreement_threshold: Minimum number of different parents to consider as disagreement (default: 3)

    Returns:
        Dict with:
            - best_leaf: Best classification at consensus level
            - best_leaf_score: Confidence score
            - final_level: Level where consensus was found
            - disagreement_detected_at: Levels where disagreement was detected
            - parent_diversity_per_level: Number of different parents at each level
            - best_route: Full path to best classification
            - topk: Top-k alternatives at the consensus level
    """
    max_level = int(taxonomy_embedded["level"].max()) if not taxonomy_embedded.empty else 0

    if max_level == 0:
        return {
            "best_leaf": None,
            "best_leaf_score": None,
            "final_level": 0,
            "disagreement_detected_at": [],
            "parent_diversity_per_level": {},
            "best_route": [],
            "topk": [],
        }

    disagreement_detected_at = []
    parent_diversity_per_level = {}

    # Start from the deepest level and move up
    for level in range(max_level, 0, -1):
        # Get ALL nodes at this level
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == level]
        if level_nodes.empty:
            continue

        # Score all nodes at this level
        scored = scoring_utils.score_candidates(input_embedding, level_nodes)
        if not scored or len(scored) == 0:
            continue

        # Get top-K candidates
        topk_candidates = scored[:min(top_k, len(scored))]

        # Count unique parents among top-K candidates
        unique_parents = set()
        for candidate in topk_candidates:
            parent_code = candidate.get("parentCode")
            if parent_code:
                unique_parents.add(parent_code)

        num_unique_parents = len(unique_parents)
        parent_diversity_per_level[level] = num_unique_parents

        # Check for disagreement
        if num_unique_parents >= disagreement_threshold:
            # Disagreement detected - too many different parents
            disagreement_detected_at.append(level)
            # Continue to next level up
            continue
        else:
            # Consensus found - top-K candidates are concentrated under few parents
            # Accept this level as the final classification level
            best_node = topk_candidates[0]
            best_score = float(best_node.get("score", 0.0))

            # Build path to the best node
            parent_codes = taxonomy_utils.get_parent_chain(
                taxonomy_embedded, best_node.get("code")
            )
            path_nodes: List[Dict[str, Any]] = []

            for code in parent_codes:
                node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
                if not node_row.empty:
                    path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))

            best_node_copy = dict(best_node)
            if "path_nodes" in best_node_copy:
                del best_node_copy["path_nodes"]
            path_nodes.append(best_node_copy)
            best_node["path_nodes"] = path_nodes

            best_route = _build_route_annotations_for_leaf(
                best_node, taxonomy_embedded, input_embedding
            )

            # Build top-k routes
            for candidate in topk_candidates:
                if candidate and not candidate.get("path_nodes"):
                    parent_codes = taxonomy_utils.get_parent_chain(
                        taxonomy_embedded, candidate.get("code")
                    )
                    path_nodes: List[Dict[str, Any]] = []
                    for code in parent_codes:
                        node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
                        if not node_row.empty:
                            path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))
                    candidate_copy = dict(candidate)
                    if "path_nodes" in candidate_copy:
                        del candidate_copy["path_nodes"]
                    path_nodes.append(candidate_copy)
                    candidate["path_nodes"] = path_nodes

            topk_routes = _build_topk_routes(topk_candidates, taxonomy_embedded, input_embedding)

            return {
                "best_leaf": best_node,
                "best_leaf_score": best_score,
                "final_level": level,
                "disagreement_detected_at": disagreement_detected_at,
                "parent_diversity_per_level": parent_diversity_per_level,
                "best_route": best_route,
                "topk": topk_routes,
            }

    # If we reached level 1 and still have disagreement, return level 1 best match
    level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == 1]
    if not level_nodes.empty:
        scored = scoring_utils.score_candidates(input_embedding, level_nodes)
        if scored:
            best_node = scored[0]
            best_score = float(best_node.get("score", 0.0))

            # No parents at level 1, just return the node
            best_node["path_nodes"] = [best_node]
            best_route = _build_route_annotations_for_leaf(
                best_node, taxonomy_embedded, input_embedding
            )

            topk_candidates = scored[:min(top_k, len(scored))]
            for candidate in topk_candidates:
                if not candidate.get("path_nodes"):
                    candidate["path_nodes"] = [candidate]

            topk_routes = _build_topk_routes(topk_candidates, taxonomy_embedded, input_embedding)

            return {
                "best_leaf": best_node,
                "best_leaf_score": best_score,
                "final_level": 1,
                "disagreement_detected_at": disagreement_detected_at,
                "parent_diversity_per_level": parent_diversity_per_level,
                "best_route": best_route,
                "topk": topk_routes,
            }

    # Fallback if nothing worked
    return {
        "best_leaf": None,
        "best_leaf_score": None,
        "final_level": 0,
        "disagreement_detected_at": disagreement_detected_at,
        "parent_diversity_per_level": parent_diversity_per_level,
        "best_route": [],
        "topk": [],
    }


def hybrid_disagreement_climbing(
    input_embedding: List[float],
    taxonomy_embedded: pd.DataFrame,
    top_k: int = 5,
    disagreement_threshold: int = 3,
    stop_on_decrease: bool = True,
) -> Dict[str, Any]:
    """
    Hybrid approach: Disagreement Detection + Bottom-Up Climbing.

    Algorithm:
    1. Use disagreement detection to find the "consensus level" (where top-K candidates converge)
    2. From that consensus level, climb up using bottom-up climbing logic
    3. Stop when parent has lower score than current node (or reach root)

    This addresses the issue where disagreement detection might stop one level too early.

    Args:
        input_embedding: Input text embedding vector
        taxonomy_embedded: Full taxonomy with embeddings
        top_k: Number of top candidates to consider
        disagreement_threshold: Minimum number of different parents to consider as disagreement
        stop_on_decrease: If True, stop climbing when confidence decreases

    Returns:
        Dict with:
            - best_leaf: Best classification at final level
            - best_leaf_score: Confidence score
            - final_level: Level where climbing stopped
            - consensus_level: Level where disagreement detection found consensus
            - disagreement_detected_at: Levels where disagreement was detected
            - parent_diversity_per_level: Number of different parents at each level
            - climbing_path: Nodes visited during climbing phase
            - best_route: Full path to best classification
            - topk: Top-k alternatives at the final level
    """
    max_level = int(taxonomy_embedded["level"].max()) if not taxonomy_embedded.empty else 0

    if max_level == 0:
        return {
            "best_leaf": None,
            "best_leaf_score": None,
            "final_level": 0,
            "consensus_level": 0,
            "disagreement_detected_at": [],
            "parent_diversity_per_level": {},
            "climbing_path": [],
            "best_route": [],
            "topk": [],
        }

    disagreement_detected_at = []
    parent_diversity_per_level = {}
    consensus_level = None
    consensus_node = None

    # PHASE 1: Disagreement Detection to find consensus level
    for level in range(max_level, 0, -1):
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == level]
        if level_nodes.empty:
            continue

        scored = scoring_utils.score_candidates(input_embedding, level_nodes)
        if not scored or len(scored) == 0:
            continue

        topk_candidates = scored[:min(top_k, len(scored))]

        # Count unique parents among top-K candidates
        unique_parents = set()
        for candidate in topk_candidates:
            parent_code = candidate.get("parentCode")
            if parent_code:
                unique_parents.add(parent_code)

        num_unique_parents = len(unique_parents)
        parent_diversity_per_level[level] = num_unique_parents

        # Check for disagreement
        if num_unique_parents >= disagreement_threshold:
            disagreement_detected_at.append(level)
            continue
        else:
            # Consensus found!
            consensus_level = level
            consensus_node = topk_candidates[0]
            break

    # If no consensus found at any level, fall back to level 1
    if consensus_node is None:
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == 1]
        if not level_nodes.empty:
            scored = scoring_utils.score_candidates(input_embedding, level_nodes)
            if scored:
                consensus_level = 1
                consensus_node = scored[0]

    if consensus_node is None:
        return {
            "best_leaf": None,
            "best_leaf_score": None,
            "final_level": 0,
            "consensus_level": 0,
            "disagreement_detected_at": disagreement_detected_at,
            "parent_diversity_per_level": parent_diversity_per_level,
            "climbing_path": [],
            "best_route": [],
            "topk": [],
        }

    # PHASE 2: Bottom-Up Climbing from consensus level
    current_node = consensus_node
    current_level = consensus_level
    current_confidence = float(current_node.get("score", 0.0))
    climbing_path = [dict(current_node)]
    stop_reason = "reached_root"

    while current_level > 1:
        parent_code = current_node.get("parentCode")
        if not parent_code or parent_code == "__root__":
            stop_reason = "reached_root"
            break

        # Find parent node
        parent_row = taxonomy_embedded[taxonomy_embedded["code"] == parent_code]
        if parent_row.empty:
            stop_reason = "parent_not_found"
            break

        parent_node = taxonomy_utils.row_to_candidate(parent_row.iloc[0])
        parent_embedding = parent_row.iloc[0].get("embedding")

        if parent_embedding is None or not isinstance(parent_embedding, (list, np.ndarray)):
            stop_reason = "parent_no_embedding"
            break

        # Calculate parent's similarity score
        parent_confidence = float(
            np.dot(input_embedding, parent_embedding) /
            (np.linalg.norm(input_embedding) * np.linalg.norm(parent_embedding))
        )
        parent_node["score"] = parent_confidence
        parent_level = int(parent_node.get("level", 0))

        # Check if we should continue climbing
        if stop_on_decrease and parent_confidence < current_confidence:
            confidence_drop = current_confidence - parent_confidence
            stop_reason = f"confidence_decreased_by_{confidence_drop:.3f}"
            break

        # Accept this parent level - continue climbing
        climbing_path.append(dict(parent_node))
        current_node = parent_node
        current_level = parent_level
        current_confidence = parent_confidence

    # The best classification is the last accepted node
    final_level = current_level
    best_node = climbing_path[-1] if climbing_path else None
    best_score = float(best_node.get("score", 0.0)) if best_node else None

    # Build path to the best node
    if best_node:
        parent_codes = taxonomy_utils.get_parent_chain(
            taxonomy_embedded, best_node.get("code")
        )
        path_nodes: List[Dict[str, Any]] = []

        for code in parent_codes:
            node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
            if not node_row.empty:
                path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))

        best_node_copy = dict(best_node)
        if "path_nodes" in best_node_copy:
            del best_node_copy["path_nodes"]
        path_nodes.append(best_node_copy)
        best_node["path_nodes"] = path_nodes

        best_route = _build_route_annotations_for_leaf(
            best_node, taxonomy_embedded, input_embedding
        )
    else:
        best_route = []

    # Get top-k alternatives at the final level
    if final_level > 0:
        level_nodes = taxonomy_embedded[taxonomy_embedded["level"] == final_level]
        level_scores = scoring_utils.score_candidates(input_embedding, level_nodes)
        topk_candidates = level_scores[:top_k]

        # Attach path_nodes to each candidate
        for candidate in topk_candidates:
            if candidate and not candidate.get("path_nodes"):
                parent_codes = taxonomy_utils.get_parent_chain(
                    taxonomy_embedded, candidate.get("code")
                )
                path_nodes: List[Dict[str, Any]] = []
                for code in parent_codes:
                    node_row = taxonomy_embedded[taxonomy_embedded["code"] == code]
                    if not node_row.empty:
                        path_nodes.append(taxonomy_utils.row_to_candidate(node_row.iloc[0]))
                leaf_node = dict(candidate)
                if "path_nodes" in leaf_node:
                    del leaf_node["path_nodes"]
                path_nodes.append(leaf_node)
                candidate["path_nodes"] = path_nodes

        topk_routes = _build_topk_routes(topk_candidates, taxonomy_embedded, input_embedding)
    else:
        topk_routes = []

    return {
        "best_leaf": best_node,
        "best_leaf_score": best_score,
        "final_level": final_level,
        "consensus_level": consensus_level,
        "disagreement_detected_at": disagreement_detected_at,
        "parent_diversity_per_level": parent_diversity_per_level,
        "climbing_path": climbing_path,
        "stop_reason": stop_reason,
        "best_route": best_route,
        "topk": topk_routes,
    }


def compute_top_down_gated_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_embedded: Dict[str, Any],
    taxonomy_key: str,
    min_confidence_level1: float,
    stop_on_decrease: bool,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute top-down routes with adaptive confidence gating for every inference sample."""

    taxonomy_df = taxonomy_embedded[taxonomy_key]()

    results: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        text = record.get("text", "")
        td_gated = top_down_with_confidence_gating(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_df,
            min_confidence_level1=min_confidence_level1,
            stop_on_decrease=stop_on_decrease,
            top_k=top_k,
        )
        results.append(
            {
                "sentenceId": record.get("sentence_id"),
                "text": text,
                "taxonomyKey": record.get("taxonomyKey"),
                "topdown_gated_best_leaf": td_gated.get("best_leaf"),
                "topdown_gated_best_leaf_score": td_gated.get("best_leaf_score"),
                "topdown_gated_stopped_at_level": td_gated.get("stopped_at_level"),
                "topdown_gated_stop_reason": td_gated.get("stop_reason"),
                "topdown_gated_best_route": td_gated.get("best_route", []),
                "topdown_gated_confidence_per_level": td_gated.get("confidence_per_level", []),
                "topdown_gated_topk_routes": td_gated.get("topk", []),
            }
        )
    return results


def compute_bottom_up_aggregation_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_embedded: Dict[str, Any],
    taxonomy_key: str,
    aggregation_method: str,
    top_k_children: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute bottom-up aggregation routes for every inference sample."""

    taxonomy_df = taxonomy_embedded[taxonomy_key]()

    outputs: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        bu_agg = bottom_up_path_aggregation(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_df,
            aggregation_method=aggregation_method,
            top_k_children=top_k_children,
            top_k=top_k,
        )
        outputs.append(
            {
                "sentenceId": record.get("sentence_id"),
                "text": record.get("text"),
                "taxonomyKey": record.get("taxonomyKey"),
                "bottomup_agg_best_leaf": bu_agg.get("best_leaf"),
                "bottomup_agg_best_leaf_score": bu_agg.get("best_leaf_score"),
                "bottomup_agg_best_at_each_level": bu_agg.get("best_at_each_level"),
                "bottomup_agg_aggregated_scores": bu_agg.get("aggregated_scores_per_level"),
                "bottomup_agg_best_route": bu_agg.get("best_route", []),
                "bottomup_agg_topk_routes": bu_agg.get("topk", []),
            }
        )
    return outputs


def compute_multilevel_simultaneous_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_embedded: Dict[str, Any],
    taxonomy_key: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute multi-level simultaneous classification routes for every inference sample."""

    taxonomy_df = taxonomy_embedded[taxonomy_key]()

    results: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        ml_sim = multi_level_simultaneous_classification(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_df,
            top_k=top_k,
        )
        results.append(
            {
                "sentenceId": record.get("sentence_id"),
                "text": record.get("text"),
                "taxonomyKey": record.get("taxonomyKey"),
                "multilevel_best_leaf": ml_sim.get("best_leaf"),
                "multilevel_best_leaf_score": ml_sim.get("best_leaf_score"),
                "multilevel_best_level": ml_sim.get("best_level"),
                "multilevel_best_at_each_level": ml_sim.get("best_at_each_level"),
                "multilevel_best_route": ml_sim.get("best_route", []),
                "multilevel_topk_routes": ml_sim.get("topk", []),
            }
        )
    return results


def compute_bottom_up_climbing_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_embedded: Dict[str, Any],
    taxonomy_key: str,
    stop_on_decrease: bool,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute bottom-up confidence climbing routes for every inference sample."""

    taxonomy_df = taxonomy_embedded[taxonomy_key]()

    results: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        bu_climb = bottom_up_confidence_climbing(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_df,
            stop_on_decrease=stop_on_decrease,
            top_k=top_k,
        )
        results.append(
            {
                "sentenceId": record.get("sentence_id"),
                "text": record.get("text"),
                "taxonomyKey": record.get("taxonomyKey"),
                "bottomup_climb_best_leaf": bu_climb.get("best_leaf"),
                "bottomup_climb_best_leaf_score": bu_climb.get("best_leaf_score"),
                "bottomup_climb_final_level": bu_climb.get("final_level"),
                "bottomup_climb_stop_reason": bu_climb.get("stop_reason"),
                "bottomup_climb_climbing_nodes": bu_climb.get("climbing_nodes"),
                "bottomup_climb_best_route": bu_climb.get("best_route", []),
                "bottomup_climb_topk_routes": bu_climb.get("topk", []),
            }
        )
    return results


def compute_disagreement_detection_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_embedded: Dict[str, Any],
    taxonomy_key: str,
    top_k: int,
    disagreement_threshold: int,
) -> List[Dict[str, Any]]:
    """Compute disagreement detection routes for every inference sample."""

    taxonomy_df = taxonomy_embedded[taxonomy_key]()

    results: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        disagree = bottom_up_with_disagreement_detection(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_df,
            top_k=top_k,
            disagreement_threshold=disagreement_threshold,
        )
        results.append(
            {
                "sentenceId": record.get("sentence_id"),
                "text": record.get("text"),
                "taxonomyKey": record.get("taxonomyKey"),
                "disagreement_best_leaf": disagree.get("best_leaf"),
                "disagreement_best_leaf_score": disagree.get("best_leaf_score"),
                "disagreement_final_level": disagree.get("final_level"),
                "disagreement_detected_at": disagree.get("disagreement_detected_at"),
                "disagreement_parent_diversity": disagree.get("parent_diversity_per_level"),
                "disagreement_best_route": disagree.get("best_route", []),
                "disagreement_topk_routes": disagree.get("topk", []),
            }
        )
    return results
