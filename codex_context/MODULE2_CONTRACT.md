# MODULE 2 CONTRACT — Inference (Zero‑shot + Incremental)

This file is the **implementation contract** for `taxomind.pipelines.inference`.
It is written so a developer (or Codex) can implement/modify Module 2 without any prior chat context.

## Objective
Given:
- `taxonomy_index`: a per‑taxonomy DataFrame built by Module 1 (multi‑view embeddings)
- `queries`: short multilingual texts (typically 1–10 words)

Produce, per query:
- a predicted taxonomy node `final_code` (leaf **or internal**) with an explanation trail.

## Key design choices (must hold)
1. **Top‑down routing + explicit stopping**: descend level by level; may stop at an internal node.
2. **Retrieval is for recall**: fast, label‑anchored retrieval to limit the work.
3. **Multi‑view scoring for decisions**: routing/validation use label + definition + examples (+ evidence).
4. **Induced candidate subgraph**: routing and validation operate on a small induced set that preserves
   (a) ancestor connectivity and (b) full sibling sets for ambiguity checks.
5. **Ancestor evidence aggregation is secondary**: may guide beam/root selection or confidence; it must not be the sole routing signal.

---

## Inputs and outputs

### Inputs
- `taxonomy_df`: DataFrame (one taxonomy partition) with required columns:
  - structural: `level, code, parentCode, label, definition, examples, isLeaf`
  - embeddings: `embedding_label` (np.ndarray), `embedding_definition` (np.ndarray|None), `embedding_examples` (np.ndarray|None)
  - evidence state (if enabled): `evidence_centroid` (np.ndarray|None), `evidence_count` (int, default 0)
- `taxonomy_graph`: adjacency dict `parent_code -> [child_codes]` (include `"__root__" -> L1 roots`)
- `queries_df`: DataFrame with at least `query_id` and `query_text`
- `model`: embedding model used to embed queries (SentenceTransformer, normalized output)
- `params`: thresholds (see Parameters section)

### Outputs (per query)
Return a row with:
- `final_code`
- `final_score` (the score used at decision time)
- `path_codes` (root→…→final)
- `stopping_reason` (string)
- `ambiguous` (bool)
- `alternatives` (optional list)
- `routing_trace` (compact trace per level: siblings considered, top scores, gaps)
- `validation` (optional: CONSISTENT / OVERRIDE / CONFLICT and supporting scores)
- Optional stability gate for overrides using the best-leaf gap within `V`.

---

## Scoring contract

### Multi‑view node score
For a query embedding `q` and node `n`:
- `sim_label = cos(q, label_effective(n))`
- `sim_def   = cos(q, def_emb(n))` if exists else `-inf`
- `sim_ex    = cos(q, ex_emb(n))` if exists else `-inf`
- `sim_emp   = cos(q, evidence_centroid(n))` if exists else `-inf`

Default pooling:
- `score(n) = max(sim_label, sim_def, sim_ex, sim_emp)`

Short‑query rule (configurable):
- if query token count `<= short_query_tokens` then use only label/evidence:
  - `score(n) = max(sim_label, sim_emp)`

### Evidence blending (label_effective)
Evidence is persisted across runs via `evidence_centroid` + `evidence_count`.
For node `n` with `k = evidence_count`:
- `beta = min(k / (k + evidence_tau), evidence_max_beta)`
- `label_effective(n) = normalize((1-beta) * embedding_label(n) + beta * evidence_centroid(n))`

If no evidence: `label_effective(n) = embedding_label(n)`.

Missing views:
- If definition/examples/evidence are missing, they are ignored in max pooling
  (do not treat as zeros).

---

## Candidate retrieval and induced subgraph

### Step 1 — Retrieval (recall)
- Retrieve top‑K codes using **label embeddings only** (fast and stable).
- Output: `R` (retrieved codes) with retrieval scores.

### Step 2 — Structural closure (connectivity)
- Build ancestor closure `A`: all ancestors on paths from each `r in R` to `__root__`.

### Step 3 — Candidate set (induced subgraph)
Sibling completion:
- `S = children(A ∪ {__root__})`

Induced node set:
- `V = R ∪ A ∪ S`

Notes:
- Ambiguity is computed over `children(p) ∩ V`, and `V` contains full sibling sets
  for every ancestor you might traverse.

### Optional — Beam roots (to reduce semantic dilution)
Use retrieval evidence to select a small set of L1 roots (beam) before routing:
- For each retrieved code `r`, map to its L1 root `root(r)`.
- Aggregate evidence per root (e.g., sum of positive retrieval scores, or sum of top‑N per root).
- Keep top `beam_count` roots (when beam selection is enabled).

Routing can be run (when beam selection is enabled):
- per beam root independently, then pick best final,
- or as a single run that starts at `__root__` but prioritizes beam roots.

If beam selection is disabled, route all L1 roots in V.

Guard:
- Beam roots are chosen from retrieval evidence, but routing must still include
  `__root__` and L1 siblings via `S` so beam selection does not become a hard filter.

---

## Routing contract (top‑down with explicit stopping)

Routing operates **only on the induced subgraph** `V`.
At each step with current parent `p`:
1. Candidate children = `children(p) ∩ V`.
2. Score every candidate child with the multi‑view score.
3. Let `c1, c2` be top‑1/top‑2 child scores.
4. Stopping is **asymmetric**:
   - **Sibling separation drives descent**: descend only if `(c1 - c2) >= min_descent_gap`.
   - **Parent competitiveness is a veto/brake** (optional): if `enable_parent_veto` and `score(p) + parent_veto_margin >= c1`, stop at `p`.

If stop: output `p`.
If descend: set `p = c1_child` and repeat until leaf or stop.

---

## Scoped validation contract (HiRAG‑style, within V)
After routing yields `TD`:
- Score a set of strong candidates within `V` (typically leaves in `V`; optionally also internal nodes).
- Compare best evidence inside TD subtree vs best evidence outside:
  - `L_sub = best leaf within subtree(TD) ∩ V`
  - `L_star = best leaf within V`

Decision:
- if `L_star` is outside subtree(TD) and `score(L_star) - score(L_sub) >= validation_threshold`
  and (optional) `score(L_star) - score(L_2) >= validation_stability_margin`,
  then **OVERRIDE** to `L_star`.
- else **KEEP** TD (CONSISTENT) or **CONFLICT** if evidence is weak/unclear.

Important: validation must not require full taxonomy scans; it is restricted to `V`.

---

## Parameters (minimum set)
- `retrieval_k`
- `beam_count`
- `enable_beam_selection`
- `short_query_tokens`
- `min_descent_gap`
- `enable_parent_veto` (bool; if false, ignore parent veto and rely on sibling separation only)
- `parent_veto_margin`
- `validation_threshold`
- `validation_stability_margin` (optional)
- `evidence_tau`
- `evidence_max_beta`
