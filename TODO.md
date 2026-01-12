# Kedro version

## **Unified Hierarchical Classification System — Final Spec (Clean & Internally Consistent)**

### **Problem Statement**

We need to classify short, multilingual text inputs (typically 2–10 words, sometimes just 1–2 words) into deep hierarchical taxonomies (e.g., ISCO/ISIC with up to 5 levels, hundreds of nodes, and generally 10–50 nodes per parent). The system must handle these constraints:

- Variable specificity: Inputs range from vague (“Education”) to precise (“React Native mobile developer”). The system must determine the appropriate depth and be allowed to stop early at an internal node.
- Sparse supervision: Most labels have very few examples; many have none. The system must work zero-shot on day 0 and improve incrementally from high-confidence corrections.
- Multiple taxonomies: Handle up to 3 taxonomies simultaneously with consistent, fast classification.
- Efficiency: Sub-second inference; avoid expensive full scans where possible.
- Explainability: Provide a score at each inferred node and its respective parents.

⸻

### **Main failure modes to prevent**

1. **Over-specification (“Education problem”).** When children under a parent are near-tied, forcing a child choice becomes random and erodes trust.
2. **Under-specification.** When the input is specific, stopping too early returns a generic parent.
3. **Top-down error propagation.** Greedy top-down routing can lock into the wrong branch early (e.g., “Shop Manager” matching “Managers” and missing “Shop Supervisors”).
4. **Incremental learning without retraining.** Corrections must improve future outputs without nightly retraining or embedding model fine-tuning.

⸻

### **Architecture overview**

The system has three layers:

- **Layer 1 (Zero-shot backbone):** taxonomy prototypes + candidate-driven hierarchical routing + stopping + bottom-up validation
- **Layer 2 (Incremental memory):** per-node centroid updates from high-confidence corrections
- **Layer 3 (Small learned calibrator):** a gate that learns stop vs descend from decision-point features (not a full classifier)

Key design decisions:

- **Greedy single-path L1 selection is unsafe.**
- Use **global retrieval** to build a **candidate set C**, then run **top-down beam only inside candidate-supported subtrees** (roots derived from ancestors of C).
- Bottom-up validation is computed **within C** (not the full taxonomy) to remain efficient.

⸻

## **Phase 0 — Initialization (Day 0)**

### **Step 0.1 Build node prototype fields (multi-view)**

Rationale: taxonomy definitions/examples can be long, while queries can be very short. To avoid prototype dilution and improve short-query matching, represent each node with multiple prototype views.

For each taxonomy node n, create these texts (if available):

- text_label(n) = label
- text_def(n) = definition
- text_ex(n) = examples

(If definition/examples are missing, omit them.)

### **Step 0.2 Embed taxonomy prototypes (zero-shot, multi-view)**

Compute and store:

- E_label[n] = normalize(embed(text_label(n)))
- E_def[n] = normalize(embed(text_def(n))) if present
- E_ex[n] = normalize(embed(text_ex(n))) if present

Do not embed a single concatenated long paragraph as the sole node representation.

Initialize incremental storage **per node**:

- C_emb_node[n] = None
- k_emb_node[n] = 0

### **Step 0.3 Define effective evidence embedding (prior + empirical evidence)**

At inference time:

- If k_emb_node[n] == 0: C_emb_node_eff[n] = None
- Else:
    - β_n = k_emb_node[n] / (k_emb_node[n] + τ) (e.g., τ=10)
    - C_emb_node_eff[n] = normalize((1-β_n)·E_label[n] + β_n·C_emb_node[n])

Notes:

- E_label[n] remains the stable prior anchor.
- C_emb_node[n] captures empirical phrasing confirmed for node n.
- Definitions/examples remain separate views, not merged into the centroid.

### **Step 0.4 Node scoring function (pooling across views)**

Define a single node score function used everywhere (routing, stopping, validation):

Compute:

- s_label = cos(q, E_label[n])
- s_def = cos(q, E_def[n]) (if available)
- s_ex = cos(q, E_ex[n]) (if available)
- s_emp = cos(q, C_emb_node_eff[n]) (if available)

Pool:

- score(n) = max(s_label, s_def, s_ex, s_emp)
    - (or mean of top-2 among available scores for smoother behavior)

Short-query rule (recommended):

- If query is extremely short (≤2 tokens), prefer label/evidence views:
    - score(n) = max(s_label, s_emp) and treat definition/examples as secondary.

### **Step 0.5 Build retrieval index (recommended)**

Index all nodes using a stable retrieval vector:

- index E_label[n] and index C_emb_node_eff[n] for nodes with evidence (maintained as a small “delta” index)

Retrieval provides candidates; final scoring always uses score(n).

⸻

## **Phase 1 — Inference (Layer 1 + Layer 2)**

### **Step 1.1 Build taxonomy-specific query views (cross-field context)**

For each taxonomy:

- ISCO query: q_isco_text = job_text + " | " + industry_text
- ISIC query: q_isic_text = industry_text + " | " + job_text

Embed and normalize:

- q = normalize(embed(q_*_text))

### **Step 1.2 Global retrieval → candidate set C (any level)**

1. Retrieve top-K nodes globally:
    - C = ANN.search(q, K) (e.g., K=50–200 depending on taxonomy size/recall needs)
2. (Optional) Expand candidate pool via semantic graph neighbors:
    - C := C ∪ neighbors(C, hops=1)
        
        (graph expansion is for recall only; hierarchy logic uses taxonomy edges)
        

### **Step 1.3 Build induced taxonomy subgraph for routing and stopping**

To make stopping signals (margin/dispersion) well-defined, routing must see sibling alternatives.

Compute:

- A = ancestors(C) (walk parentCode up to the L1 roots)
- R = { r ∈ A | level(r)=1 } (active L1 roots)
- S = children(A) (all direct children of each ancestor in A)

Induced node set:

- V = C ∪ A ∪ S

Routing uses only taxonomy parent–child edges restricted to V.

### **Step 1.4 Top-down traversal with stopping (beam over active roots)**

For each active root r ∈ R (beam over roots):

1. Set current node p = r.
2. Let candidate children be children(p) ∩ V (siblings must be present due to children(A)).
3. Score each child c:
    - s(c) = score(c)
4. Compute decision-point signals:
    - s_parent = score(p)
    - s1, s2 = top-1 and top-2 child scores
    - margin = s1 - s2
    - parent_gap = s1 - s_parent
    - dispersion = std(child_scores) (across all available children)
    - metadata: level, child_count, is_leaf
5. Decide STOP vs DESCEND:
    - If Layer 3 gate is available and trusted: use it (Phase 3).
    - Else fallback rules:
        - stop if **margin is small** AND **parent is competitive** (parent_gap small/negative)
        - stop if **dispersion is very low** (children indistinguishable)
        - stop if confidence degrades with depth (optional)
6. If stop: return p as final for this beam path.
7. If descend: set p = argmax_c s(c) and repeat until leaf or stop.

Select the best path across beams (e.g., highest final-node score + stability tie-breakers).

### **Step 1.5 Bottom-up validation (scoped to C)**

Purpose: catch wrong-branch selections and validate depth without full-leaf scanning.

Define:

- TD = top-down predicted node (may be internal)
- Leaves(TD) = leaf descendants under TD (within the induced set when possible)

Compute scoped best leaf in candidate pool:

- L* = argmax_{leaf ∈ C} score(leaf) (if C contains leaves; else use best node in C with a leaf preference rule)
- L_sub = argmax_{leaf ∈ Leaves(TD) ∩ C} score(leaf) (if any; else treat as missing)

Decision logic (with explicit ancestry guards):

1. If L_sub exists and L* == L_sub and L_sub is descendant of TD: → **CONSISTENT**
2. If L_sub exists and L* is not under TD:
    - margin = score(L*) - score(L_sub)
    - stability = score(L*) - score(second_best_leaf_in_C)
    - If margin > override_margin AND stability > stability_margin: → **OVERRIDE** to L*
    - Else: → **CONFLICT** (judge or return ambiguous with alternatives)
3. Optional deepen-within-subtree rule:
    - If L_sub is deeper than TD and strongly supported: → **DEEPER** to L_sub
4. If L_sub does not exist (no leaf evidence for TD in C):
    - treat as insufficient scoped evidence; return TD but report low validation strength (or expand K / neighbors and retry)

Return:

- final node
- alternatives (TD, L*, L_sub if defined)
- explanation (“best candidate leaf lies outside TD subtree”, “candidates cluster under TD”, etc.)

⸻

## **Phase 2 — Incremental learning from corrections (Layer 2)**

### **Step 2.1 Update emb_node centroid for the corrected node (O(1))**

Given correction (input_text, correct_node_id=n):

1. x = normalize(embed(input_text))
2. Update centroid:
    - if k_emb_node[n] == 0: C_emb_node[n] = x
    - else: C_emb_node[n] = normalize((k·C_emb_node[n] + x)/(k+1))
3. k_emb_node[n] += 1

Do not push the same example embedding into all ancestors.

⸻

## **Phase 3 — Layer 3 calibrator (gate + optional reranker)**

### **Step 3.1 Log decision points during inference**

Log:

- taxonomy id, level, parent node id
- child scores (top few) via score(child)
- s_parent, s1, s2, margin, parent_gap, dispersion
- query length (tokens/chars)
- chosen action
- later, corrected final node (if feedback arrives)

### **Step 3.2 Generate training labels from corrections**

When a correction arrives with final_correct_node Y:

For each decision point at parent P encountered during the predicted traversal:

1. If P is not ancestor-or-equal to Y: skip (wrong branch)
2. If P == Y: label = STOP
3. Else: label = DESCEND

### **Step 3.3 Train the gate (safe, balanced)**

Maintain buffers per (taxonomy, level) for stop/descend.

Update only when both classes have ≥ m examples; train on balanced batches.

### **Step 3.4 Gate features**

Use score-only features first. Add embeddings later only if needed.

### **Step 3.5 Inference with the gate**

Use gate probabilities to decide stop/descend; fallback to handcrafted stopping if gate isn’t trusted.

Optional Step 3.6 Reranker if sibling-choice errors dominate.

---