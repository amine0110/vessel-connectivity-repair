# Mind the Gap — vessel connectivity repair (from the paper)

Broken vessel segmentations are a quiet killer for anything that needs a centerline, a branch graph, or a connected overlay in a viewer. A mask can look fine on Dice and still fall apart into dozens of components because of a one-voxel gap.

I came across **[Mind the Gap: Mesh-Guided Repair of Broken Vessels](https://arxiv.org/abs/2609.29779)** (Drwiega, Szymanski, Wodzinski — Sano / AGH). The idea is exactly right for this problem: keep your nnU-Net (or any) binary mask, fit a geometric prior, and reconnect components with thin bridges under a foreground-growth budget — instead of replacing the mask with a voxelized mesh.

The authors have not released public code yet. I liked the solution enough that I handed the paper to **GPT-6 Astra** and asked it to implement a workable research prototype from the description alone. This repository is that result, cleaned up so anyone can try it.

> This is **not** the official paper implementation. It deliberately focuses on the connectivity-repair stage (components → thin bridges → growth checks → optional cleanup). The full paper pipeline (GCN mesh decoder, FOMAML meta-init, anatomy-specific templates) is **not** reimplemented here. Treat this as an open, paper-inspired starting point until the authors ship their repo.

Write-up on the PYCAD side: [Broken vessel mask in, connected overlay out](https://pycad.co/blog/broken-vessel-mask-in-connected-overlay-out-centerlines-graphs/).

## What it does

**Input:** binary vessel NIfTI (or any `>0` foreground)  
**Output:** repaired binary NIfTI with local tubular bridges, same geometry (spacing / origin / direction)

Prior used here: nearest **boundary-endpoint** pairs in physical mm (KD-tree), straight thin tubes, accept only if components merge and cumulative added voxels stay under a max foreground-growth fraction. Optional island filter / small-component cleanup.

On a real single-class vessel mask I tried: **43 → 2** connected components with ~2% foreground growth (default settings). Your mileage will vary — always QC the bridges visually.

## Install

```bash
git clone https://github.com/amine0110/mind-the-gap-vessel-repair.git
cd mind-the-gap-vessel-repair
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# or: pip install -e .
```

Python 3.10+. CPU only — NumPy / SciPy / scikit-image / NiBabel.

## Quick demo (synthetic broken mask)

```bash
python scripts/run_demo.py
# or
python -m vessel_repair.cli --demo --outdir demo_out
```

## Repair your own mask

```bash
python -m vessel_repair.cli \
  --input broken.nii.gz \
  --output repaired.nii.gz \
  --tube-radius-mm 1.0 \
  --max-added-fg-fraction 0.05 \
  --max-gap-mm 20.0 \
  --cleanup
```

| Flag | Default | Meaning |
|---|---:|---|
| `--tube-radius-mm` | 1.0 | Bridge tube radius (mm) |
| `--max-added-fg-fraction` | 0.05 | Max added voxels / original foreground |
| `--max-gap-mm` | 20.0 | Max endpoint gap to consider (mm) |
| `--filter-island-mm` | 0 | Drop islands farther than this from the largest component (0 = off) |
| `--min-component-size` | 1 | Cleanup size threshold |
| `--cleanup` | off | Remove tiny leftovers after bridging |

The CLI prints JSON with before/after component counts, voxels added, and accepted bridge lengths.

## How this maps to the paper

See [PAPER_NOTES.md](PAPER_NOTES.md) for a short method digest. Roughly:

| Paper (§2.4) | This repo |
|---|---|
| Fitted deformable mesh as scaffold | Simpler distance / boundary prior |
| Mesh-graph or endpoint + mesh validation | Endpoint pairs + growth gate |
| Thin tube + max FG growth | Same idea |
| FOMAML / GCN decoder / Tables 1–2 | Not included |

If you need production topology repair, wait for (or rebuild) the authors’ mesh-guided path — and validate on your own protocol.

## Citation

Please cite the original work if you use these ideas:

```bibtex
@article{drwiega2026mindthegap,
  title   = {Mind the Gap: Mesh-Guided Repair of Broken Vessels},
  author  = {Drwiega, Gniewosz and Szymanski, Wojciech and Wodzinski, Marek},
  journal = {arXiv preprint arXiv:2609.29779},
  year    = {2026}
}
```

Paper: https://arxiv.org/abs/2609.29779 · PDF: https://arxiv.org/pdf/2609.29779

## Disclaimer

Research software for experimentation and education. **Not** a medical device. Do not use for clinical decisions without your own validation, senior review of bridge sites, and appropriate regulatory path.

## License

MIT — see [LICENSE](LICENSE).

—
Mohammed El Amine Mokhtari ([@amine0110](https://github.com/amine0110)) · [PYCAD](https://pycad.co)
