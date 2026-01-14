# Module 2 — Execution contract (Codex)

## Pipeline intent (Kedro)
Typical node chain (names indicative):

1. validate_payload
2. convert_to_dataframe
3. load_taxonomy_index
4. prepare_scoring_views (build O(1) dicts for embeddings + evidence)
5. embed_queries
6. retrieve_candidates (label-only recall; return R and ancestor-closure A; optional beam_roots)
7. route_query_topdown (Variant 2: sibling ambiguity across ALL taxonomy children; multi-view scoring)
8. validate_prediction_scoped (HiRAG-style, scoped to candidate set)
9. format_results

## Multi-view scoring (routing + validation)
Score defaults to **max across available views**, with short-query protection.

Views:
- sim_label_effective (label blended with evidence when evidence_count>0)
- sim_def
- sim_examples
- sim_evidence_centroid (optional)

Rule:
- if query tokens <= `short_query_tokens` (configurable, default 2):
  - score = max(sim_label_effective, sim_evidence_centroid)
- else:
  - score = max(sim_label_effective, sim_def, sim_examples, sim_evidence_centroid)

## Candidate refinement (structural)
- Always compute ancestor closure A: all ancestors from retrieved nodes up to `__root__`.

## Routing variant (chosen)
**Variant 2 (taxonomy-complete ambiguity):**
- At each parent, ambiguity is computed over **all** children in `taxonomy_graph[parent]`.
- Routing remains top-down and level-by-level.
- Ancestor evidence may be used as a *veto* but not as the primary descent signal.

## Scoped validation (HiRAG-style)
- Compare top-down result TD to best leaf L* in candidate set V.
- Also compute best leaf under TD subtree, L_sub (within V).
- If (score(L*) - score(L_sub)) exceeds threshold and L* is outside TD subtree:
  - OVERRIDE to L*
- Else:
  - keep TD (CONSISTENT)
- If evidence insufficient (no leaves under TD in V):
  - return TD with weak/ambiguous validation (or increase retrieval K upstream).
