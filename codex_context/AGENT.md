# AGENT INSTRUCTIONS — Hierarchical Classification System

You are an expert engineer working on a hierarchical text classification system.
You MUST follow the architectural principles and constraints defined below.

This file is authoritative. Do not invent alternative designs.

---

## SYSTEM GOAL

Classify short, multilingual text inputs (1–10 tokens) into deep hierarchical
taxonomies (e.g. ISCO / ISIC), allowing explicit stopping at non-leaf nodes.

---

## HARD CONSTRAINTS (DO NOT VIOLATE)

- Routing MUST be hierarchical and top-down (level by level).
- Non-leaf nodes MUST be valid final predictions.
- No flat classification.
- No forced leaf prediction.
- No full taxonomy scans at inference time.
- No retraining or fine-tuning embedding models.
- Incremental learning MUST be per-node only (no ancestor drift).
- Ancestor evidence may be used ONLY as a secondary signal.
- Broad-topic aggregation must NEVER dominate narrow semantic matches.

---

## CORE DESIGN PRINCIPLES

- The taxonomy defines a **semantic prior**, not a classifier.
- Semantic similarity alone NEVER decides depth.
- Routing decisions are structural; semantics guide but do not override structure.
- Ambiguity must result in stopping, not guessing.
- Evidence accumulation must be stable and monotonic over time.

---

## ALLOWED SIGNALS (WITH PRIORITY)

1. **Sibling separation** (primary signal for descent)
2. **Parent competitiveness** (veto / braking signal)
3. **Ancestor evidence aggregation** (confidence / beam prior only)
4. **Bottom-up validation** (HiRAG-style, scoped)

---

## EXPLICIT NON-GOALS

- Treating the taxonomy as a flat label space
- Global ancestor voting as the main routing signal
- Propagating evidence embeddings to ancestors
- Using entropy alone as a stopping criterion
- Training a classifier to replace routing logic

---

## DEVELOPMENT RULES

- Prefer clarity over cleverness
- Follow the written specs in MODULE2_SPEC.md
- If unsure, STOP and ask instead of guessing