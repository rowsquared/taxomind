# Module 2 — Inference (Zero‑shot + Incremental)

This spec defines the intended behavior of Module 2 **independent of code**.

## What Module 2 must do
For each query:
1. retrieve a small candidate set (recall)
2. refine candidates (semantic + structural)
3. route top‑down with explicit stopping
4. optionally validate scoped to candidates (HiRAG‑style)
5. return an explainable result (path, scores, stopping reason)

The taxonomy is a tree/forest with columns:
`level, code, parentCode, label, definition, examples`
`definition` is mandatory, `examples` is optional.

---

## Core principles
- **Non‑leaf outputs are valid**: the system must be able to stop early.
- **Retrieval is not classification**: retrieval provides recall; decisions use hierarchical logic.
- **Sibling ambiguity must be measured on complete sibling sets** (per visited parent), otherwise stopping is unreliable.
- **Ancestor evidence aggregation is allowed only as a secondary signal** (beam/priors/veto), never as the sole descent driver.

---

## Step 2.1 Candidate retrieval (recall)
- Use a stable, short representation for fast retrieval (typically `embedding_label`).
- Retrieve top‑K nodes globally (any level).
- Keep retrieval scores for later diagnostics and (optionally) beam/root priors.

---

## Step 2.2 Candidate refinement (semantic + structural)

### Semantic refinement
- Re‑score retrieved nodes using **multi‑view scoring** over:
  - label
  - definition
  - examples (if present)
  - evidence centroid (if present)

### Structural refinement (critical)
Build an induced candidate set **V** that guarantees routing can:
- traverse from roots to candidates (connectivity)
- compute ambiguity at each parent using the **full sibling set under that parent**

To do this:
1. **Ancestor closure**: include all ancestors for each retrieved node up to `__root__`.
2. **Sibling completion (within induced subgraph)**: for every ancestor parent on those paths (including `__root__`), include **all of its children**.

This yields the induced set:
- `R` = retrieved codes
- `A` = ancestors(R)
- `S` = children(A ∪ {__root__})
- `V = R ∪ A ∪ S`

Clarification (matches your “Variant 2” intent):
- Once root `2` is selected/considered, ambiguity is computed across **all children of `2`**.
- We do **not** require scanning siblings from unrelated roots that are not in the induced set.

### Optional: beam roots (anti‑dilution)
If retrieval returns mixed topics (e.g., “Legislators” also retrieves “Lawyers/Judges/Economists”), allow a small **beam over L1 roots**:
- aggregate evidence per root from retrieved items
- keep top‑B roots
- route each root beam independently and select the best final output
- if beam selection is disabled, route all L1 roots in V

---

## Step 2.3 Hierarchical routing with explicit stopping
Induced set with sibling completion:
- `S = children(A ∪ {__root__})`
- `V = R ∪ A ∪ S`

Route **top‑down** over `V`:
- start at `__root__` (or at each beam root)
- at each parent, score **all candidate children** = `children(parent) ∩ V`
- decide **STOP vs DESCEND** explicitly

Stopping logic is asymmetric:
- **Sibling separation drives descent** (need a clear best child vs runner‑up)
- **Parent competitiveness acts as a veto/brake** (optional via `enable_parent_veto`)

This directly targets:
- over‑specification (near‑tied children → stop)
- under‑specification (clear winner → descend)

---

## Step 2.4 Scoped validation (HiRAG‑style)
After top‑down routing returns `TD`:
- validate against strong evidence elsewhere in `V` (no full scans)
- detect wrong‑branch cases (TD subtree vs best leaf outside)
- decide: CONSISTENT / OVERRIDE / CONFLICT

Optional stability guard:
- require `score(L*) - score(L2) >= validation_stability_margin` before overriding.

---

## Multi‑view scoring (normative)
Score a node as:
- `score(n) = max(sim_label, sim_def, sim_ex, sim_emp)`
- short‑query protection (≤2 tokens): `max(sim_label, sim_emp)`

Evidence is persisted per node and blended into the effective label embedding using a bounded weight based on `evidence_count`.

Missing views:
- If definition/examples/evidence are missing, ignore those views in max pooling.

---

## Parameters
- `retrieval_k`
- `beam_count`
- `enable_beam_selection`
- `short_query_tokens`
- `min_descent_gap`
- `enable_parent_veto` (bool)
- `parent_veto_margin`
- `validation_threshold`
- `validation_stability_margin` (optional)
- `evidence_tau`
- `evidence_max_beta`
