# Incremental Learning — Specification

## Stored Per Node

- evidence_centroid (vector or null)
- evidence_count (int)
- evidence_last_updated (timestamp)

---

## Update Rule (On Correction)

Given corrected example x for node n:

- Embed x → e
- If evidence_count == 0:
    centroid = e
- Else:
    centroid = normalize((k * centroid + e) / (k + 1))
- Increment evidence_count

No ancestor updates.

---

## Evidence Blending (Inference Time)

Dynamic beta per node:

beta_n = min(k / (k + tau), max_beta)

Effective label embedding:

E_eff = (1 - beta_n) * E_label + beta_n * evidence_centroid

---

## Key Rule

> Evidence adjusts confidence and routing behavior,
> not taxonomy structure.