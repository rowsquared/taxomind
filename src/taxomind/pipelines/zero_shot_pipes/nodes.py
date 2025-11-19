"""Zero-shot routing nodes with multilingual guarantees."""

from __future__ import annotations

from typing import Any, Dict, List

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


def load_test_sentences(test_dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    return entries


def build_full_paths(taxonomy_embedded: pd.DataFrame) -> pd.DataFrame:
    """Enumerate every root-to-leaf path in the taxonomy."""

    return taxonomy_utils.build_full_paths(taxonomy_embedded)


def embed_full_paths(paths: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Embed the textual representation of all taxonomy paths."""

    paths = paths.copy()
    if paths.empty:
        paths["embedding"] = [[] for _ in range(len(paths))]
        return paths
    embeddings = embedding_utils.embed_texts(
        paths["path_text"].fillna("").tolist(), model_name=model_name
    )
    paths["embedding"] = embeddings
    paths["embedding_model_name"] = model_name
    return paths


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
) -> Dict[str, Any]:
    """Rank leaves directly and reconstruct the best route bottom-up."""

    leaves = taxonomy_embedded[taxonomy_embedded["isLeaf"].astype(bool)]
    leaf_scores = scoring_utils.score_candidates(input_embedding, leaves)
    topk_candidates = leaf_scores[:top_k]

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
        path_nodes = best_path.get("path_nodes") or []
        if path_nodes:
            best_leaf = dict(path_nodes[-1])
            best_leaf["score"] = float(best_path.get("score", 0.0))
        best_leaf_score = float(best_path.get("score", 0.0))

    topk_routes: List[Dict[str, Any]] = []
    for path_candidate in topk_paths:
        path_nodes = path_candidate.get("path_nodes") or []
        leaf_node = None
        if path_nodes:
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
    taxonomy_embedded: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute top-down routes for every inference sample."""

    results: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        text = record.get("text", "")
        td = top_down_route(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_embedded,
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
    taxonomy_embedded: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute bottom-up routes leveraging existing embeddings."""

    outputs: List[Dict[str, Any]] = []

    for record in sentence_embeddings:
        bu = bottom_up_route(
            input_embedding=record.get("embedding"),
            taxonomy_embedded=taxonomy_embedded,
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


def compute_flat_routes(
    sentence_embeddings: List[Dict[str, Any]],
    taxonomy_full_paths: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Compute flat Top-K similarity against full taxonomy paths."""

    paths_df = taxonomy_full_paths.copy()
    results: List[Dict[str, Any]] = []
    for record in sentence_embeddings:
        fr = flat_route(
            input_embedding=record.get("embedding"),
            taxonomy_full_paths=paths_df,
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


def compare_routes(
    topdown_results: List[Dict[str, Any]],
    bottomup_results: List[Dict[str, Any]],
    flat_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge signals from top-down, bottom-up, and flat similarity."""

    bottom_map = {record["sentenceId"]: record for record in bottomup_results}
    flat_map = {record["sentenceId"]: record for record in flat_results}
    compared: List[Dict[str, Any]] = []

    for top in topdown_results:
        sid = top.get("sentenceId")
        bottom = bottom_map.get(sid, {})
        flat = flat_map.get(sid, {})

        top_best = top.get("topdown_best_leaf")
        bottom_best = bottom.get("bottomup_best_leaf")
        flat_best = flat.get("flat_best_leaf")
        top_route = top.get("topdown_best_route") or []
        bottom_route = bottom.get("bottomup_best_route") or []
        flat_best_route = flat.get("flat_best_route") or []
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
                "topdown_topk_routes": top.get("topdown_topk_routes", []),
                "bottomup_topk_routes": bottom.get("bottomup_topk_routes", []),
                "flat_topk_routes": flat.get("flat_topk_routes", []),
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
                },
                "topdown_best_route": top_route,
                "bottomup_best_route": bottom_route,
                "flat_best_route": flat_best_route,
            }
        )

    return compared


def finalize_predictions(
    compared_results: List[Dict[str, Any]],
    taxonomy_embedded: pd.DataFrame,
    judge_model_name: str,
    encoder_model_name: str,
    debug_level: str = "low",
) -> Dict[str, Any]:
    """Produce the final decision with judge escalation when needed.

    Args:
        compared_results: Results from compare_routes
        taxonomy_embedded: Embedded taxonomy DataFrame
        judge_model_name: Model name for judge
        encoder_model_name: Model name for encoder
        debug_level: Output verbosity ("low", "medium", "high")
    """

    suggestions: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for record in compared_results:
        flat_topk = record.get("flat_topk_routes", [])

        judge_output = None
        classifier_model = encoder_model_name

        should_judge = decide_if_judge_needed(record)

        final_decision = None
        final_confidence = None

        # Call judge if needed
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
                        taxonomy_embedded, candidate_pool[:5]
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
            selected_route = _select_route_for_decision(
                final_decision, record, taxonomy_embedded
            )
        else:
            selected_route = []

        # Build annotations from the selected route
        annotations = selected_route

        if not annotations:
            annotations = _default_unknown_annotation()

        # Determine if judge intervened
        judge_intervened = judge_output is not None

        # Build output based on debug level
        output = _build_output_by_debug_level(
            sentence_id=record.get("sentenceId"),
            text=record.get("text"),
            annotations=annotations,
            validation_match=bool(record.get("validation_match")),
            conflicts=record.get("conflicts", {}),
            classifier_model=classifier_model,
            judge_intervened=judge_intervened,
            topdown_topk_routes=record.get("topdown_topk_routes", []),
            bottomup_topk_routes=record.get("bottomup_topk_routes", []),
            flat_topk_routes=flat_topk,
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
) -> List[Dict[str, Any]]:
    """Construct annotation-style routes using taxonomy embeddings."""

    route_codes = _extract_route_codes(leaf, taxonomy)
    if not route_codes:
        return []

    rows: List[pd.Series] = []
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
    leaf: Dict[str, Any] | None, taxonomy: pd.DataFrame
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
    for node in path_nodes or []:
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
) -> List[Dict[str, Any]]:
    """Build annotations for the final decision route."""

    if not decision:
        return []

    if decision.get("path_nodes"):
        base_route = _build_route_annotations(decision.get("path_nodes"))
    else:
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
                base_route = annotations
                break

        if not base_route:
            base_route = _build_route_annotations_for_leaf(
                decision, taxonomy, input_embedding=None
            )

    return _apply_confidence_to_route(
        base_route, _candidate_score(decision)
    )


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


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to native types."""

    if isinstance(obj, dict):
        return {key: _to_native(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_native(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_to_native(item) for item in obj)
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj
