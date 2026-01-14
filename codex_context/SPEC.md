# Unified Hierarchical Classification System — Working Spec (Codex Context)

## 0. Scope and invariants

We classify short, multilingual text inputs (typically 2–10 words; sometimes 1–2) into deep hierarchical taxonomies (e.g., ISCO/ISIC):

- Up to ~5 levels
- Hundreds of nodes
- Usually 10–50 children per parent

Each taxonomy node has the schema:

`level, code, parentCode, label, definition, examples (optional)`

**Definition is mandatory** for every node and is part of the *taxonomy prior*.
**Examples are optional**.

### Root sentinel
Use a single sentinel for roots: `__root__`.

- Any missing/empty parentCode must be normalized to `__root__`.
- No other root tokens (e.g., `"**root**"`) should appear after normalization.

---

## 1. Functional requirements

1. **Variable specificity**
   - Inputs range from vague (“Education”) to precise (“React Native mobile developer”).
   - The system decides how deep to go.
   - **Non-leaf nodes are valid final outputs** (explicit stopping).

2. **Zero-shot readiness**
   - Works immediately after taxonomy load (day 0).
   - No labeled examples required.

3. **Incremental improvement from sparse corrections**
   - New text examples arrive over time with only some nodes labeled.
   - Improve without retraining the embedding model or full retraining cycles.

4. **Efficiency**
   - Sub-second per query (avoid full scans when possible).

5. **Explainability**
   - At each level, return (when possible):
     - node score (primary)
     - parent comparison
     - stopping reason (e.g., “siblings near-tied”, “parent competitive”)

---

## 2. Failure modes to prevent (with examples)

1. **Over-specification (“Education problem”)**
   - When children under a parent are near-tied, forcing a child is arbitrary.
   - Example: Query “Education” with children {Primary, Secondary, Higher, Adult} all similar → **stop at Education**.

2. **Under-specification**
   - When input is specific but model stops too early.
   - Example: “React Native mobile developer” stops at “Software Developers” instead of “Mobile Application Developers” → **descend while evidence is strong**.

3. **Top-down error propagation**
   - Greedy routing locks into wrong branch early.
   - Example: “Shop Manager” routes “Managers → Hospitality Managers”, missing “Shop Supervisors” → **retrieval must surface alternative branches** and validation can override.

4. **Incremental learning instability**
   - Sparse corrections must not distort parents.
   - Example: Few corrections for “Shop Supervisors” drift “Managers” centroid → **update only the corrected node**.

5. **Semantic dilution from broad-topic aggregation**
   - Broad parents gather weak evidence from many descendants and dominate incorrectly.
   - Example: Query “Legislators” retrieves Lawyers/Judges/Economists; naive ancestor aggregation makes “Professionals” win over “Legislators” → **ancestor evidence must be secondary and safety-oriented, not primary routing**.

---

## 3. Architecture overview

Three modules, with clear separation:

### Module 1 — Taxonomy preparation (Day 0)
Purpose: build a stable semantic prior.

- Parse taxonomy tree/forest.
- Build node text views:
  - label (primary anchor)
  - definition (mandatory, secondary view)
  - examples (optional, tertiary view)
- Compute embeddings per view (same model; L2-normalized).
- Prepare fast inference structures:
  - parent/child adjacency (or functions + cached maps)
  - ancestry checks
  - optional retrieval index over label embeddings (stable)

**Key principle:** taxonomy embeddings define a prior, not a classifier.

### Module 2 — Inference (zero-shot + incremental)
Purpose: produce predictions immediately and improve smoothly with evidence.

1. **Candidate retrieval (recall)**
   - Fast recall using label embeddings only (stable).

2. **Candidate refinement (semantic + structural)**
   - Re-score retrieved candidates using multi-view semantic scoring (label/definition/examples + evidence).
   - Structural closure:
     - include all ancestors needed to connect candidates to the root
   - Ambiguity assessment:
     - **Variant 2 / recommended:** sibling ambiguity computed over **all taxonomy children** for the current parent (not only retrieved).
   - Optional light ancestor evidence:
     - secondary signal only (confidence modulation / veto), never sufficient alone to force descent.

3. **Hierarchical routing with explicit stopping**
   - True top-down traversal:
     - score only children of current parent at each step
     - decide STOP vs DESCEND
   - Stopping is asymmetric:
     - sibling separation drives descent
     - parent competitiveness acts as a veto/brake

4. **Scoped validation (HiRAG-style)**
   - Validates top-down result against strong evidence elsewhere in candidate set (no full taxonomy scans).
   - Outputs: CONSISTENT / OVERRIDE / CONFLICT(AMBIGUOUS)

**Key principle:** routing is structural; similarity alone never decides depth.

### Module 3 — Incremental learning from corrections
Purpose: adapt to real user language without retraining.

- Maintain per-node evidence:
  - `evidence_centroid` (nullable vector)
  - `evidence_count` (int)
  - optional `evidence_last_updated`
- Update on correction: centroid update for that node only (no ancestor propagation).
- Evidence blending in scoring:
  - dynamic β_n = min(k/(k+τ), β_max)
  - effective label embedding = normalize((1-β_n)E_label + β_n evidence_centroid)

**Key principle:** learning adjusts confidence/stopping, not taxonomy structure.

---

## 4. Explicit non-goals

- Flat classification over all nodes
- Letting global evidence override local structural decisions
- Using ancestor vote aggregation as the sole routing signal
- Retraining embedding models on sparse corrections
- Full-leaf scans during inference
