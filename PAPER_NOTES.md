# Mind the Gap — Condensed Method Notes (arXiv:2609.29779)

**Paper:** Drwiega, Szymanski, Wodzinski — *Mind the Gap: Mesh-Guided Repair of Broken Vessels* (arXiv:2609.29779, Sep 2026).  
**Blog:** https://pycad.co/blog/broken-vessel-mask-in-connected-overlay-out-centerlines-graphs/

## Problem
nnU-Net vessel masks can have high Dice but broken connectivity (missing thin bridges). Downstream centerlines / graphs / CFD fail. Repair is **post-process only** on a hard binary NIfTI mask (no intensities / probs).

## Full paper pipeline (spirit)
1. Preserve NIfTI spacing / origin / direction.
2. Optional distant FP island filtering (artifact keep/remove distances).
3. Extract surface (marching cubes), normalize by physical bbox.
4. Fit anatomy-specific deformable template mesh (GCN decoder + Chamfer + regularizers); FOMAML warm-start. Templates: cylinder (aorta), torus-like (TopCoW), sphere (PARSE).
5. **Connectivity repair (§2.4)** — do NOT voxelize the whole mesh:
   - Find disconnected components (26-connectivity).
   - Propose thin bridges:
     - **Mesh-graph:** anchor components to nearby mesh vertices; shortest paths on mesh graph (aorta / TopCoW).
     - **Endpoint:** short endpoint-to-endpoint bridges; mesh validates support (PARSE).
   - Rasterize as thin tubes (physical mm radius).
   - Accept only if merges components AND added voxels ≤ max foreground-growth fraction.
   - Optional mesh-supported cleanup of small unsupported islands.
6. Output repaired binary mask; Betti-0 ≈ component count.

## Table 2 defaults (CLI-overridable inspiration)
| Param | Aorta | TopCoW | PARSE |
|-------|-------|--------|-------|
| Variant | mesh-graph | mesh-graph | endpoint |
| Tube radius base/min/max mm | 1.0 / 2.0 / 5.0 | 0.25 / 0.35 / 1.0 | 0.4 / 0.7 / 1.8 |
| Max added FG fraction | 0.03 | 0.15 | 0.06 |
| Artifact keep/remove mm | 57 / 57.5 | 57 / 57.5 | 20 / 25 |
| Cleanup max removed voxels | 1000 | 1000 | 200 |
| Connectivity | 26 | 26 | 26 |
| Min component size | 1 | 1 | 1 |

Shared defaults from §2.4 text: adaptive local bridge radii with 6.0 mm window; post-repair mesh distance 2.0 mm; min mesh-close fraction 0.01.

## What this repository implements
This open prototype does **not** include the full GCN mesh decoder or FOMAML. It focuses on a practical geometric prior for connectivity repair:

- Boundary / endpoint pairs in physical mm, thin tubes, and a foreground-growth gate
- Optional island filtering and small-component cleanup

See the README for install, CLI flags, and how this maps to the paper.
