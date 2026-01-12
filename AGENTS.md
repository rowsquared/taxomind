AGENT.md

Scope and Purpose

This document defines the authoritative classification architecture for this project and serves as the contributor guide for AI agents and developers.

The system performs zero-shot, hierarchical, multilingual text classification over deep taxonomies (e.g. ISCO, ISIC) and incrementally improves from human corrections.

Mandatory technologies:
	•	Kedro for pipeline structure, reproducibility, and orchestration
	•	txtai for embeddings, ANN retrieval, and optional graph-aware expansion

Agents MUST follow this specification. Deviations require explicit justification.

⸻

Architectural Principles (Validated)

This design is self-consistent and correct under the following principles:
	1.	Short-query realism: queries may be 1–3 tokens; node representations must not be diluted by long text.
	2.	Stable priors + adaptive evidence: taxonomy labels are stable anchors; empirical phrasing is learned incrementally.
	3.	Retrieval ≠ decision: ANN retrieval is for recall only; final decisions always use a unified scoring function.
	4.	Hierarchy-aware stopping: stopping is a first-class decision, not a post-hoc heuristic.
	5.	Local evidence only: learning updates affect only the corrected node, not its ancestors.

⸻

Phase 0 — Initialization (Day 0)

Step 0.1 Build node prototype views (multi-view)

Rationale
Taxonomy definitions and examples may be long, while queries are often extremely short. A single concatenated representation causes semantic dilution and harms short-query recall.

For each taxonomy node n, construct the following independent textual views (if available):
	•	text_label(n) → node label (mandatory)
	•	text_def(n) → definition (optional)
	•	text_ex(n) → examples (optional)

Missing views are simply omitted.

⸻

Step 0.2 Embed taxonomy prototypes (zero-shot, multi-view)

Compute and store normalized embeddings:
	•	E_label[n] = normalize(embed(text_label(n)))
	•	E_def[n] = normalize(embed(text_def(n))) (if present)
	•	E_ex[n] = normalize(embed(text_ex(n))) (if present)

Hard rule: do NOT embed a single long concatenation as the sole node representation.

Initialize incremental evidence storage per node:
	•	C_emb_node[n] = None
	•	k_emb_node[n] = 0

⸻

Step 0.3 Define effective evidence embedding (prior + empirical)

At inference time, define the effective empirical embedding:
	•	If k_emb_node[n] == 0:
C_emb_node_eff[n] = None
	•	Else:
	•	β_n = k_emb_node[n] / (k_emb_node[n] + τ) (e.g. τ = 10)
	•	C_emb_node_eff[n] = normalize((1 − β_n) · E_label[n] + β_n · C_emb_node[n])

Interpretation:
	•	E_label[n] is the immutable semantic prior
	•	C_emb_node[n] captures confirmed empirical phrasing
	•	Definitions/examples remain separate views and are not merged into the centroid

⸻

Step 0.4 Unified node scoring function (view pooling)

Define a single scoring function used everywhere (routing, stopping, validation):

For query embedding q:
	•	s_label = cos(q, E_label[n])
	•	s_def = cos(q, E_def[n]) (if available)
	•	s_ex = cos(q, E_ex[n]) (if available)
	•	s_emp = cos(q, C_emb_node_eff[n]) (if available)

Pooling:

score(n) = max(s_label, s_def, s_ex, s_emp)

(Optional smoother variant: mean of top-2 available scores.)

Short-query rule (recommended):
	•	If query length ≤ 2 tokens:
	•	prioritize label and empirical views
	•	treat definition/example scores as secondary

⸻

Step 0.5 Retrieval index construction

Build ANN indices strictly for candidate recall:
	•	Primary index: E_label[n] for all nodes (stable, short, comparable)
	•	Optional secondary index: C_emb_node_eff[n] for nodes with evidence

Invariant:
	•	Retrieval only proposes candidates
	•	Final decisions ALWAYS use score(n)

⸻

Phase 1 — Inference (Routing + Validation)

Step 1.1 Taxonomy-specific query views

Queries may be multi-field and taxonomy-dependent.

Examples:
	•	ISCO: job_text | industry_text
	•	ISIC: industry_text | job_text

Embed and normalize:
	•	q = normalize(embed(q_text))

⸻

Step 1.2 Global retrieval → candidate set C
	1.	Retrieve top-K nodes globally:
	•	C = ANN.search(q, K) (K ≈ 50–200)
	2.	Optional recall expansion:
	•	C := C ∪ neighbors(C, hops=1) using taxonomy graph

Graph expansion is recall-only; hierarchy semantics remain unchanged.

⸻

Step 1.3 Induced taxonomy subgraph

Stopping decisions require sibling visibility.

Compute:
	•	A = ancestors(C) (walk to level-1 roots)
	•	R = { r ∈ A | level(r) = 1 }
	•	S = children(A) (direct children of all ancestors)

Induced node set:
	•	V = C ∪ A ∪ S

Routing is restricted to parent–child edges within V.

⸻

Step 1.4 Top-down traversal with stopping

For each active root r ∈ R (beam over roots):
	1.	Set p = r
	2.	Let children(p) = children(p) ∩ V
	3.	Score each child using score(c)
	4.	Compute decision signals:
	•	s_parent = score(p)
	•	s1, s2 = top-1 / top-2 child scores
	•	margin = s1 − s2
	•	parent_gap = s1 − s_parent
	•	dispersion = std(child_scores)
	•	metadata: level, child_count, is_leaf
	5.	Decide STOP vs DESCEND:
	•	Prefer learned gate if available (Phase 3)
	•	Else fallback rules:
	•	stop if margin is small AND parent is competitive
	•	stop if dispersion is very low (children indistinguishable)
	•	optional depth-confidence decay
	6.	If STOP → return p
	7.	If DESCEND → p = argmax_c score(c) and repeat

Select the best beam by final score and stability criteria.

⸻

Step 1.5 Bottom-up validation (scoped)

Purpose: detect wrong-branch errors without full leaf scans.

Definitions:
	•	TD = top-down result (possibly internal)
	•	Leaves(TD) = descendant leaves under TD

Compute:
	•	L* = argmax_{leaf ∈ C} score(leaf)
	•	L_sub = argmax_{leaf ∈ Leaves(TD) ∩ C} score(leaf) (if any)

Decision logic:
	1.	If L_sub exists and L* == L_sub → CONSISTENT
	2.	If L_sub exists and L* outside TD subtree:
	•	override only if margin and stability exceed thresholds
	•	else mark CONFLICT
	3.	Optional: deepen to L_sub if strongly supported
	4.	If L_sub missing: return TD with low validation strength

Return final node, alternatives, and explanation.

⸻

Phase 2 — Incremental Learning (Evidence Layer)

Step 2.1 Update node evidence centroid

Given correction (input_text, correct_node = n):
	1.	x = normalize(embed(input_text))
	2.	Update:
	•	if k == 0: C_emb_node[n] = x
	•	else: C_emb_node[n] = normalize((k · C_emb_node[n] + x) / (k + 1))
	3.	k_emb_node[n] += 1

Critical rule: do NOT propagate embeddings to ancestors.

⸻

Phase 3 — Learned Gate (Optional, Controlled)

Step 3.1 Decision logging

Log at each routing decision:
	•	taxonomy, level, parent id
	•	child scores
	•	s_parent, margin, parent_gap, dispersion
	•	query length
	•	chosen action
	•	later correction (if any)

⸻

Step 3.2 Training labels from corrections

For corrected final node Y, at each visited parent P:
	•	if P not ancestor of Y: skip
	•	if P == Y: label = STOP
	•	else: label = DESCEND

⸻

Step 3.3 Gate training
	•	Maintain buffers per (taxonomy, level)
	•	Train only when both classes are sufficiently populated
	•	Use balanced batches

Initial features: score-only. Add embeddings only if necessary.

⸻

Step 3.4 Inference with gate

Use gate probabilities to decide STOP vs DESCEND.

If gate confidence is low or unavailable, fallback to handcrafted rules.

Optional reranker may be added if sibling mis-selection dominates.

⸻

Kedro & txtai Enforcement
	•	All phases MUST be implemented as Kedro pipelines or nodes
	•	txtai is the sole component for:
	•	embeddings
	•	ANN retrieval
	•	optional graph expansion

Agents should assume:
	•	deterministic pipelines
	•	explicit dataset versioning
	•	no hidden global state

⸻

Agent Rules (Non-Negotiable)
	•	Never collapse multi-view node representations into one vector
	•	Never train on ancestors implicitly
	•	Never use ANN scores directly as final scores
	•	Always return uncertainty when evidence is insufficient

This document is the single source of truth for the system. Any ambiguities must be resolved by referring back to this specification.