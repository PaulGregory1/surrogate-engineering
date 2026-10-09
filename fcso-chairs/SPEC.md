# FCSO spec — chair volume reduction on a Mac

Oct 8, 2026 · @paul

## Goal and ground rules

Build an FCSO pipeline **that works** on a MacBook Pro M3 Pro (18 GB unified memory). It reduces the volume of chairs, in the spirit of section 5.2 of the paper, and compares against a simple baseline. Reproducing the full benchmark is out of scope: about ten chairs are enough.

- **No training**: only the open-source 3DShape2VecSet weights, plus inference and optimization.
- **Comparison**: FCSO versus plain gradient descent on the latent (GD), the standard method with no correction. FMG is optional: it is the fastest method in the paper (Table 4).

The paper's PDF (*Flow-Corrected Shape Optimization: Taming Manifold Drift in High-Dimensional 3D Models*, arXiv 2608.07199) is provided to you. It is the reference for anything this spec leaves unspecified. Read these first:

- Algorithm 1 and Equations 5 to 9 (Section 4);
- Section 5.2 and Figure 3;
- Appendix B (baselines), Appendix D (why the correction works), Appendix E (guidance ablation), Appendix F.1 and Table 6 (chair hyperparameters).

Rules:

- No training or fine-tuning. If a step seems to require it, stop and propose an alternative before writing code.
- Where this spec departs from the paper, it does so on purpose and says so explicitly. Everywhere else, follow the paper.
- Every decision not covered here or in the paper goes into `DECISIONS.md`, one line per decision, with its rationale.
- Before any run longer than 30 minutes, measure the speed on 1 or 2 chairs and report the estimated duration.

## Open-source model

Use a single model: **the official 3DShape2VecSet release**. The paper's chair models were never published, so these public weights stand in for them.

| Item | Detail |
| --- | --- |
| Code and weights | [github.com/1zb/3DShape2VecSet](https://github.com/1zb/3DShape2VecSet), “Pretrained model” link in the README |
| Shape decoder | KL autoencoder `kl_d512_m512_l8`: 512 latents × 8, occupancy logits |
| Generative prior (corrector) | EDM diffusion `kl_d512_m512_l8_d24_edm`, conditioned on the ShapeNet class (chair) |
| Guidance | Option B from the paper (gradient, Eq. 8), since the prior is not conditioned on volume |
| Data | ShapeNet as preprocessed by the repo (`occupancies.zip`, `surfaces.zip`) |
| Departures from the paper | EDM diffusion instead of flow matching; 512×8 instead of 256×8; occupancy instead of SDF; Option B instead of CFG |

Option B is what the paper prescribes for a pretrained prior that is not conditioned on the objective (Section 4.2, paragraph after Eq. 8). Hunyuan3D is out of scope: heavier, image-conditioned, and not needed for this goal.

Check every model name, function and path cited here against the repo's actual code: they come from its README and may have changed.

## Mac constraints

Everything must run on a MacBook Pro M3 Pro with 18 GB of unified memory and no CUDA GPU. Keep peak memory below roughly 10 GB to avoid swapping.

- **Device**: `mps`, falling back to `cpu`. Run with `PYTORCH_ENABLE_MPS_FALLBACK=1` and list in `DECISIONS.md` the ops that fall back to CPU.
- **Precision**: fp32 by default. fp16 is allowed only for prior forward passes, and only if a test shows a negligible difference in decoded volume.
- **Forbidden dependencies** (CUDA-only): flash-attn, xformers, pytorch3d, kaolin, nvdiffrast, torchmcubes, FlashVDM.
- **`torch_cluster`**: if the VecSet code imports it for FPS, replace it with a pure-PyTorch FPS. Point clouds of 2048 points make this cheap.
- **Attention**: `torch.nn.functional.scaled_dot_product_attention`, which works on MPS.
- **Marching cubes**: `skimage.measure.marching_cubes`, on CPU.
- **Rendering**: PyVista in offscreen mode (VTK). No EGL or OSMesa on macOS.
- **Long runs**:
  - resumable checkpoints, at least one per optimized shape;
  - `caffeinate -i` to prevent sleep;
  - logs written to a file;
  - a `--debug` mode on 3 shapes and a few iterations.
- **No pointless heavy compute**: decode the volume grid in chunks (see the volume section), never all 64³ queries at once with gradients.

## Data and splits

Use the ShapeNet data already preprocessed by the 3DShape2VecSet repo: watertight, normalized meshes with occupancies and surface point clouds. No mesh preprocessing is needed. Keep only the chair category (synset `03001627`); if the archives cover all categories, extract only that folder.

Start from the repo's train/val/test splits and take, outside the autoencoder's training set:

- `opt`: 10 chairs to optimize (3 in `--debug` mode), chosen to be varied (thin legs, armrests, central pedestal);
- `ref`: 200 other chairs, only if you do the optional KD metric.

Fix the seed and write the ID lists to `splits/`, so every method uses the same shapes.

Encoding a shape: 2048 points sampled from `surfaces.zip` → KL encoder. Use the posterior **mean** as the initial latent `z_s`, not a sample. Check in the diffusion training code (`main_class_cond.py` and its engine) how latents are produced (sample or mean, any scaling factor) and reproduce that normalization exactly: the prior must see latents at the scale it was trained on.

## Adapting Phase 2 to an EDM diffusion prior

The paper corrects with a rectified flow, but the chosen model only ships an EDM diffusion. You therefore replace the ODE of Eq. 9 with EDM's probability-flow ODE, started at an equivalent noise level. Both are parameterizations of the same transport. Appendix D justifies the correction through the score, which EDM also provides.

**Equivalent noise level.** In a rectified flow, a state at t = 1 − τ satisfies `z_t / (1−τ) = ẑ + (τ/(1−τ))·ε`. In EDM, `x_σ = ẑ + σ·ε`. Hence:

```latex
\sigma_{\text{start}} = \frac{\tau}{1-\tau}
```

With τ = 0.2 (Table 6), this gives σ\_start = 0.25. The equivalence assumes latents at the prior's training scale (see Data). Also check `sigma_data` in `EDMPrecond`.

**Unguided correction.** Noise `x = ẑ + σ_start·ε`, then integrate from σ\_start down to 0 in N = 10 steps. Use the model's denoiser `D(x; σ, class=chair)`, the derivative `d = (x − D(x; σ)) / σ` and an Euler step `x ← x + (σ_next − σ)·d`. Take the intermediate σ values from the repo's EDM schedule (ρ = 7), truncated to \[0, σ\_start\]. Heun is allowed as a variant.

**Guidance, Option B (Eq. 8).** The condition is `c = V(ẑ)`, the volume reached at the end of Phase 1. At each step, add a term that decreases `(V(x) − c)²`:

```latex
d_{\text{guided}} = d + \lambda \, \nabla_x \big(V(\mathrm{Dec}(x)) - c\big)^2
```

The sign is “+” because σ decreases: (σ\_next − σ) is negative. **Test it**: from the same x, a guided step must reduce `(V − c)²` more than an unguided step.

- By default, as in Eq. 8, decode the noisy state x directly. Variant `x0`: decode `D(x; σ)` and differentiate through the denoiser. It is more expensive and often more stable.
- λ does not transfer from the paper (0.001 for compliance). Calibrate it so the guidance term's norm is about 0.1 to 1 times the norm of d at the first step, then try ×3 and ÷3 on the 3 chairs of milestone 6.
- The guidance gradient must never reach the model's parameters: `requires_grad=False` everywhere, and `torch.no_grad()` for the denoiser in the default mode.

## Differentiable volume estimator

This follows Eq. 16 of the paper, adapted to occupancy logits instead of an SDF:

```latex
V(z) = |B| \cdot \frac{1}{|G|} \sum_{x \in G} \sigma\big(\beta \, \ell_\theta(x, z)\big)
```

- `ℓ_θ(x, z)` is the decoder's occupancy logit. Check in the code that positive means inside.
- `G` is a uniform grid over the data's normalization box (check its bounds in the repo's evaluation code), and `|B|` is the volume of that box.
- Resolution: 64³ for evaluation and thresholds. 48³ is allowed during optimization if a test on 3 chairs shows less than 2% difference from 64³.
- `β = 1` by default, since the logits are already sharp. Increase β only if the sum of sigmoids drifts away from the volume of the extracted mesh.
- Report volume as a ratio (`V / V_initial`). The paper's absolute scale (V ≈ 4.5) is not reproducible.

**Chunked computation.** The decoder has two stages: self-attention layers over the latents, then cross-attention to the queries. Proceed as follows:

1. Compute `h = self_attn(post_kl(z))` once, with the graph.
2. Detach: `h_leaf = h.detach().requires_grad_()`.
3. For each chunk of about 16k to 32k queries, compute its volume contribution from `h_leaf`, then call `backward()`. Gradients accumulate in `h_leaf.grad`.
4. Finish with `h.backward(h_leaf.grad)` to propagate back to z.

Volume is a mean, so this decomposition is exact. The same mechanism serves the Phase 2 guidance.

**Required tests.**

- `torch.autograd.gradcheck`, or a finite difference, on a small grid.
- On 5 chairs from `opt`, `V(z_s)` must be within ±5% of the `trimesh` volume of the mesh extracted by marching cubes.

## FCSO algorithm and hyperparameters

Implement Algorithm 1 of the paper: K cycles, each made of M gradient steps on V (Phase 1) followed by a prior-guided correction (Phase 2, previous section). The goal is to minimize V, starting from `z_s` and targeting a 70% reduction.

| Parameter | Default | Source |
| --- | --- | --- |
| Gradient steps per cycle M | 15 | Table 6 |
| Learning rate η | 0.005 | Table 6 |
| Noise ratio τ | 0.2 (σ\_start = 0.25) | Table 6 |
| ODE steps N | 10 | Table 6 |
| Guidance λ | calibrated (Phase 2 section) | — |
| Volume grid | 64³ (48³ if validated) | Appendix F.1 |
| Max budget | 1000 gradient steps | Appendix E (Table 5) |

Execution rules:

- **Phase 1**: literal SGD from Eq. 5 by default. Adam is a configurable option; log it in `DECISIONS.md` if you keep it. With Adam, reset the optimizer state at every cycle, since Phase 2 moves z.
- **Progress measure**: after each cycle, so after Phase 2, compute `r = 1 − V / V_initial` at 64³.
- **Matched snapshots**: the first time r crosses 10, 20, …, 70%, save the latent. This is the “matched optimization progress” of Section 5.1: all methods are evaluated at the same reduction level.
- **Stopping**: at r ≥ 70% or at the max budget. Shapes that miss a threshold count as failures at that threshold (success rate, as in Table 5), not silently dropped.
- **Seed**: fixed and logged per shape. The Phase 2 noise ε is redrawn at every cycle.
- **Per-shape log**: V, r, `(V − c)²` before and after Phase 2, norm of `z − z_s`, time per cycle.

## Methods compared

The required comparison is FCSO versus GD. The other two methods are optional and cost only a flag in the same script. All methods use the same volume estimator, the same chairs and the same budget.

| Method | Status | Definition |
| --- | --- | --- |
| FCSO | Required | Full algorithm (previous section) |
| GD | Required | Phase 1 only: gradient descent on the latent, same η, no regularization (λ\_reg = 0), then a second run with `λ_reg·‖z − z_s‖²`, λ\_reg = 1e-3 (Appendix B) |
| FMG | Optional (fastest in the paper) | Phase 2 only, repeated: noise to σ\_start, then a guided ODE toward `c = 0.97·V_current` |
| FCSO without guidance | Optional | FCSO with λ = 0: shows what the guidance is for (Appendix E) |

Unregularized GD should exhibit drift: floating fragments, disappearing legs (Figure 3, 3DShape2VecSet row). If it does not, raise η or the budget until it actually reaches 70%, before comparing.

D-Flow, ICTM and SGO are out of scope: they would have to be ported to this latent space.

## Evaluation

Methods are compared on two axes: time to produce a result, and shape quality at equal reduction. Measurements are taken at the 50% and 70% reduction thresholds (matched progress, Section 5.1).

**Meshes.** Marching cubes at 128³ on the logits, at level 0. Keep every connected component: floating fragments are part of what is being measured.

**Measurements per chair and per method.**

| Measure | Definition |
| --- | --- |
| Success | Threshold reached within budget (yes/no) |
| Time | Seconds to reach the threshold (wall clock, same machine) |
| Stray components | Number of connected components other than the largest, and their share of the volume |
| Distance to initial | Chamfer distance (10k points) between the optimized and the initial shape |
| Relative volume | `V / V_initial` reached |

Aggregate per method: success rate, median time, mean of the other columns.

**Visual grid.** For all 10 chairs: the initial shape, then each method at 70%, with the relative volume shown, as in Figure 3. PyVista offscreen rendering, 3/4 view, white background, identical camera everywhere. This is the main deliverable for judging quality.

**Interactive 3D trajectory viewer (required, added by @paul on Oct 8).** One self-contained HTML page (`report/viewer.html`, three.js loaded from a CDN, no server needed) for comparing the method with correction against the one without, step by step:

- A chair selector (the 10 `opt` chairs). For the selected chair, one row per method: FCSO on top, GD (λ\_reg = 0) below. Other runs (GD λ\_reg = 1e-3, FMG, FCSO without guidance) appear as extra rows behind a toggle when they exist.
- Columns are the matched snapshots: the initial shape, then 10%, 20%, …, 70% reduction. Every cell is a rotatable 3D mesh. All cells share one camera, so rotating or zooming one moves them all.
- Under each cell: `V / V_initial` (two decimals) and the gradient-step count and wall-clock time at that snapshot. A threshold the method never reached shows an empty cell marked "not reached".
- Meshes come from the evaluation meshes (marching cubes at 128³, every component kept, so fragments stay visible), decimated for the web (target ~20k faces each). Browsers block `fetch` on `file://`, so store each chair's meshes as a `report/meshes/<id>.js` file (base64 typed arrays assigned to a global) loaded on demand with a `<script>` tag: the page must work by double-click. Colour stray components differently from the largest one.
- Add a static PNG version of the same grid (initial + 10…70%, both methods, volume under each) for every chair to `report/`, and link the viewer from `REPORT.md`.

**Optional.** KD with DINOv2 ViT-B/14 on 10 views per shape, against the `ref` set (formula in Appendix A). With 10 chairs, treat it as indicative only.

## Repo layout and milestones

The repo is a plain Python project. 3DShape2VecSet is cloned into `third_party/` and left untouched, except for minimal Mac-compatibility fixes, which must be listed.

```
fcso-chairs/
  configs/            # one YAML per method
  third_party/        # 3DShape2VecSet
  splits/             # opt.txt (+ ref.txt if KD)
  fcso/
    prior.py          # load AE + EDM, encode/decode, denoiser
    volume.py         # estimator + chunked decoding
    correct.py        # Phase 2: noising, ODE, guidance
    optimize.py       # FCSO, GD (+ FMG, FCSO without guidance)
    mesh_render.py    # marching cubes, PyVista renders
    metrics.py        # components, Chamfer, timing (+ KD)
  scripts/            # run_opt.py, make_report.py
  tests/              # guidance sign, volume gradcheck
  DECISIONS.md
  REPORT.md
  report/             # grid PNGs, viewer.html + its meshes
```

Milestones, in order. Move on only once the criterion is met.

1. **Environment.** Python 3.10+ venv, PyTorch with MPS, repo cloned, weights and chair data downloaded.
   - Criterion: `torch.backends.mps.is_available()` is true and both checkpoints load.
2. **Reconstruction.** Encode then decode 3 chairs from `opt` and render them.
   - Criterion: the renders look like the original chairs.
3. **Prior.** Sample 4 chairs from noise, then apply an unguided correction at σ\_start = 0.25 to 3 encoded chairs.
   - Criterion: the samples are plausible chairs, and the correction changes volume by less than 5%.
4. **Volume.** Estimator and tests (dedicated section).
   - Criterion: tests pass, and the peak memory of one gradient step is logged.
5. **GD.** On 3 chairs, down to 70%.
   - Criterion: drift is visible, or a written explanation of why it is not.
6. **FCSO.** Guidance sign test, λ calibration, then FCSO on the same 3 chairs.
   - Criterion: 70% reached with no floating fragments or lost legs on at least 2 of the 3 chairs.
7. **Comparison.** All 10 chairs × FCSO and GD (plus the optional methods if time allows), then `REPORT.md` with the grid, the measurement table and a summary of `DECISIONS.md`, plus the interactive 3D trajectory viewer (Evaluation section). Every run must save the latent at each matched snapshot (initial, 10…70%) with its step count and time, so the viewer can be built without rerunning anything.

**Order of magnitude, to be checked at milestone 6.** One FCSO cycle costs about 25 forward/backward decoder passes (15 in Phase 1, 10 in Phase 2), and about twenty cycles get to 70%. If one 64³ pass takes 0.3 to 1 s on the M3 Pro, expect a few minutes per chair, and milestone 7 fits in one to two hours. If it is much slower, switch to 48³.

## Paper ambiguities and default decisions

The paper leaves several points open or contradictory. Apply the default decision, make it configurable, and log any change in `DECISIONS.md`.

| Point | What the paper says | Default decision |
| --- | --- | --- |
| Phase 1 optimizer | Eq. 5: plain gradient descent | SGD; Adam as an option |
| Number of ODE steps | Table 6: “100 schedule steps” and N = 10 actual | Algorithm 1: N steps of size τ/N between 1 − τ and 1 |
| CFG scale | Text: ω > 1; Table 6: 1.0 | Not applicable (Option B) |
| What is decoded in guidance | Eq. 8: ∇ of O(D(z\_t)) on the noisy state | Noisy state; `x0` variant (denoised prediction) |
| “Epochs” in Appendix E | Undefined | Read as gradient steps |
| Volume units | V ≈ 4.5, undefined | Relative volume only |
| Prior | Rectified flow conditioned on volume (CFG) | EDM diffusion conditioned on class + Option B |
| Representation | SDF, 256 × 8 latents | Occupancy, 512 × 8 latents (public weights) |
| GD regularization | λ not given | Two runs: 0, then 1e-3 |
| FMG target | Undefined outside CFG | `c = 0.97·V_current` per repetition |

When in doubt on an unlisted point, follow the paper if possible. Otherwise, take the simplest option and log it.

## Expected outcome

The deliverable is a pipeline that runs end to end on the Mac, with a `REPORT.md` that answers one question: at equal volume reduction, how long does FCSO take compared with GD, and are the chairs in better shape?

What to expect if the method behaves here as it does in the paper:

- **Quality**: FCSO chairs stay whole at 70%, with thinner legs and backrest. GD chairs show floating fragments or missing parts, hence more stray components.
- **Time**: FCSO is slower than GD, roughly 2 to 4 times according to Table 4 of the paper. The report must quantify this overhead.

If that is not what you see, do not tune the hyperparameters until you get it. Report the results as they are, with your hypotheses about the cause: EDM prior instead of a flow, λ, σ\_start, 512×8 latents.

## Sources

- FCSO paper (PDF provided): arXiv 2608.07199
- [3DShape2VecSet, official repo](https://github.com/1zb/3DShape2VecSet)
