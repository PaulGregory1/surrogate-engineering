# fcso-chairs

Flow-Corrected Shape Optimization (FCSO, arXiv 2608.07199) against plain latent gradient descent (GD), reducing
the volume of 10 ShapeNet chairs by up to 70 %, on a MacBook Pro M3 Pro (MPS, 18 GB), with no training: the
public 3DShape2VecSet weights serve both as the shape decoder (KL autoencoder `kl_d512_m512_l8`) and as the
corrector (class-conditioned EDM diffusion `kl_d512_m512_l8_d24_edm`).

- **Results:** [`REPORT.md`](REPORT.md). To see the chairs, open [`report/viewer.html`](report/viewer.html) by
  double-click: one chair at a time, one row per method, columns initial → 10 … 70 % reduction → final, with
  `V/V₀`, step and time under each shape, all cells sharing one camera, detached fragments in red. Static versions:
  [`report/trajectories/`](report/trajectories/) (one image per chair) and [`report/grid_70.png`](report/grid_70.png).
- **Contract:** [`SPEC.md`](SPEC.md). Every choice it leaves open: [`DECISIONS.md`](DECISIONS.md).

| at 70 % reduction | FCSO (x0 guidance) | GD |
| --- | --- | --- |
| chairs reaching it | 6/10 | 10/10 |
| median time | 254 s | 76 s |
| reached chairs in one piece | 2/6 (median stray volume 1.2 %); the 4 that stall stay whole | 0/10 (mean 55.6 stray components, 29 % of the volume) |

## Run it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
git clone https://github.com/1zb/3DShape2VecSet third_party/3DShape2VecSet   # tested at 8df9b7a, left unmodified
.venv/bin/python scripts/fetch_data.py lists                 # chair split lists, by HTTP range requests
.venv/bin/python scripts/fetch_data.py shapes --split test   # ~390 MB, never the 600 GB archives
```

Weights: `checkpoint-199.pth` (AE) and `checkpoint-499.pth` (EDM) from the "Pretrained model" Google Drive link of
the 3DShape2VecSet README, under `weights/ae/kl_d512_m512_l8/` and `weights/dm/kl_d512_m512_l8_d24_edm/`;
`scripts/check_env.py` checks that they load. Then, with `PYTORCH_ENABLE_MPS_FALLBACK=1`:

```bash
.venv/bin/python scripts/run_opt.py --config configs/fcso_x0.yaml   # or gd, gd_reg, fcso, fcso_noguide, fmg
.venv/bin/python scripts/make_report.py
.venv/bin/python -m pytest tests
```

`runs/` keeps the per-chair logs, metrics and renders; the latents (`*.pt`) and evaluation meshes (`*.npz`) are
not versioned. A full run of the six methods on 10 chairs took about 7 h.
