# Hierarchical Classification System — Architecture

## Problem

We classify short, multilingual text into deep hierarchical taxonomies
with sparse or zero supervision.

Inputs may be vague or highly specific.
The system must decide the correct depth and stop explicitly when needed.

---

## Key Challenges

- Over-specification (forcing arbitrary children)
- Under-specification (stopping too early)
- Top-down error propagation
- Semantic dilution from broad ancestors
- Incremental learning without retraining

---

## Architectural Modules

### Module 1 — Taxonomy Preparation
- Parse taxonomy structure
- Embed node label, definition, examples (multi-view)
- Initialize incremental learning state per node

### Module 2 — Inference
- Candidate retrieval (recall)
- Multi-view scoring
- Top-down hierarchical routing with explicit stopping
- Scoped bottom-up validation

### Module 3 — Incremental Learning
- Update node-specific evidence centroids
- Blend evidence with taxonomy prior using bounded confidence

---

## Guiding Principle

> The taxonomy is a structured semantic space.
> Inference is navigation, not classification.