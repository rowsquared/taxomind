"""
Module 2 - Inference Pipeline.

This module implements the runtime inference system for hierarchical
classification over deep taxonomies. It follows the approved specification
for Module 2 with:

1. Label-based retrieval for initial candidate recall
2. Multi-view scoring with on-demand embedding_effective computation
3. Top-down routing with asymmetric stopping (parent veto)
4. Explainability at each decision point

Key Design Principles:
- Retrieval uses stable label embeddings (NOT multi-view)
- Multi-view scoring happens during re-ranking only
- embedding_effective = (1-β) * embedding_label + β * evidence_centroid
- Stopping criteria: sibling separation drives, ancestors veto
- Non-leaf nodes are valid outputs

Spec Reference: Module 2 — Inference
Failure Modes Prevented:
- Over-specification (Education problem) via parent veto
- Under-specification via sibling separation requirements
- Top-down error propagation via candidate expansion
"""

__all__ = ["pipeline", "nodes"]
