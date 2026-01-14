# Module 2 — Inference Specification

This document defines the REQUIRED behavior of inference.

---

## 2.1 Candidate Retrieval (Recall)

Purpose: high recall, stable, fast.

- Use label embeddings only.
- Exact or ANN dot-product search.
- Retrieve top-K nodes (K configurable).
- Retrieved nodes may be at any level.

---

## 2.2 Structural Refinement (Semantic + Structural)

After retrieval:

- Compute ancestor closure:
  - A = all ancestors of retrieved nodes up to roots
- Candidate set:
  - V = Retrieved ∪ Ancestors

Sibling expansion is NOT required here.

Optional:
- Aggregate retrieved evidence by L1 root
- Use as beam prior (NOT routing signal)

---

## 2.3 Multi-View Scoring

For any node n, compute:

- sim_label
- sim_definition
- sim_examples
- sim_evidence (if available)

Scoring rule:

- For short queries (≤2 tokens):
  - score(n) = max(sim_label, sim_evidence)
- Otherwise:
  - score(n) = max(sim_label, sim_definition, sim_examples, sim_evidence)

Weighted sums are allowed but NOT required.

---

## 2.4 Hierarchical Routing (Top-Down)

Routing MUST be level-by-level.

Algorithmic constraints:

- Start at root (or beam root).
- At each step:
  - Consider ALL children of current parent
  - Score each child using multi-view scoring
- Decide STOP vs DESCEND explicitly.

Stopping logic (asymmetric):

1. **Sibling ambiguity**
   - If best_child_score - second_best < min_gap → STOP
2. **Parent competitiveness (veto)**
   - If parent_score + margin ≥ best_child_score → STOP

If descending:
- Move to best child
- Repeat until stop or leaf

---

## 2.5 Scoped Validation (HiRAG-style)

After routing:

- Let TD = top-down result
- Let L* = best leaf in V (candidate set)
- Let L_sub = best leaf under TD (if exists)

Decision:

- If L* outside TD subtree AND
  score(L*) - score(L_sub) > threshold:
    → OVERRIDE to L*
- Else:
    → KEEP TD

Validation is scoped to V.
No full taxonomy scans.

---

## 2.6 Outputs & Explainability

Return:

- final_code
- path
- score
- stopping_reason
- ambiguous flag
- alternatives (if ambiguous)
- routing_trace