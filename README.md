# A surrogate-modelling project

One self-contained repository taking an expensive physics solver and asking how far a
learned model can replace it — and, on top of it, an interactive platform that puts the
result in an engineer's hands.

> **Other labs live on other branches.**  `main` deliberately carries the flow project
> alone.  [`stress-lab`](../../tree/stress-lab) holds the companion study — 2-D linear
> elasticity on unstructured meshes, mesh graph networks, and a design-ranking result
> that cuts against the premise it was built on — together with the two-project
> comparison this README used to open with.  `bridge-lab` and `thermal-pinn` hold work
> in progress.

| | [`flow-surrogate`](flow-surrogate) |
|---|---|
| physics | 2-D incompressible flow (lattice Boltzmann) |
| geometry | parametric obstacle, **fixed Cartesian lattice** |
| what varies | 7 shape coefficients + Reynolds number |
| surrogate input | rasterised level set |
| models compared | POD + ridge / POD + kernel ridge / U-Net / U-Net + physics loss / FNO |
| the question | *can I differentiate through the surrogate to optimise a shape, and does the optimum survive contact with the solver?* |

It runs end to end on a laptop (Apple M-series, MPS).  Data generation is ~20-25 min;
training is minutes per model.

> **Not familiar with these models?**  The repo carries a `BACKGROUND.md` that explains
> everything its README assumes, from first principles and with figures:
> [`flow-surrogate/BACKGROUND.md`](flow-surrogate/BACKGROUND.md) covers reduced-order
> models (POD + ridge / kernel ridge), U-Nets, physics-informed losses, Fourier neural
> operators, shape derivatives and the winner's curse.  It assumes no prior CFD or deep
> learning.

![flow-surrogate setup](flow-surrogate/figs/setup.png)

*The problem: a body in a channel, parametrised by six Fourier coefficients on a fixed
192x96 lattice.*

## Results at a glance

**`flow-surrogate`** — 640 solver runs, 128 unseen test geometries.

| | field rel L2 | drag error | vs solver |
|---|---|---|---|
| POD + kernel ridge (ROM baseline) | 0.107 | 3.78 % | 19 800x |
| U-Net + continuity loss | **0.078** | **1.03 %** | 713x |

Gradient descent through the surrogate cuts drag **18.6 % at constant area, verified by
the solver** — 4 000 design evaluations in 17 s against ~2 h of solver time.  The
surrogate was 11 % wrong about its own optimum, and one round of active learning halved
that gap without moving the optimum: it bought reliability, not a better design.  The
ROM/network crossover sits at ~320 solver runs.

---

## The problem, formally

A design vector $d$ fixes a geometry $\Omega(d)$; a PDE gives a field; a functional of
that field is the number an engineer actually wants:

$$\mathcal A\big(u;\,\Omega(d)\big) = 0,\qquad J(d) = Q\big(u_d\big).$$

Here $\mathcal A$ is steady Navier–Stokes and $J$ is the drag coefficient $C_d$.  $J$ is
a *boundary* functional of a *derivative* of $u$ — which is why field accuracy and
engineering accuracy disagree throughout the project.  One evaluation costs 1.7 s, so a
thousand-design loop is hours.

**Stage 1 — sample and solve.**  Draw $\{d_i\}_{i=1}^N$, run the solver, keep both the
field and $J$.  **Stage 2 — fit** an encoder $X(d)$ and a network:

$$\theta^\star = \arg\min_\theta\ \frac1N\sum_{i=1}^N \ell\Big(N_\theta\big(X(d_i)\big),\ \big(u_{d_i},\, J(d_i)\big)\Big),$$

with $\ell$ carrying a field term *and* a term on $J$ itself, because a model fitted to
the field alone is not fitted to the quantity of interest.  **Stage 3 — put it in a
design loop:**

$$\min_{d\,\in\,\mathcal D}\ \hat J_{\theta^\star}(d)\qquad\text{s.t.}\qquad g(d) = 0,$$

the constraint being constant area, so that the optimiser cannot cheat by removing
material.

Everything about the method follows from **the encoder $X$**.  A fixed Cartesian lattice
makes $X(d)$ a tensor of constant shape and, crucially, a *differentiable* function of
$d$, so stage 3 is gradient descent through the whole pipeline.  (Change that one choice
— an unstructured mesh, where node count and connectivity are combinatorial functions of
$d$ — and nothing differentiates, stage 3 becomes a search over sampled candidates, and a
different toolbox applies.  That is the companion study on `stress-lab`.)

### The failure mode

**Optimising against your own error.**  Write $\hat J = J + e$.  Minimising $\hat J$
does not sample $e$ neutrally — it actively seeks out the designs where $e$ is most
negative, so

$$\mathbb E\big[\,e(d)\ \big|\ d = \arg\min \hat J\,\big] \;<\; \mathbb E\big[\,e(d)\,\big]$$

even for a surrogate that is unbiased on its training distribution.  This is the
surrogate's winner's curse, and it is a statement about the predicted *value*, not about
the design.  Measured here: 11.3 % optimistic at the optimum, while the optimum itself is
genuinely 18.6 % better, verified by the solver.  One active-learning round on the 32
designs the optimiser proposed halves the gap.

The consequence the whole project is built around: a surrogate used as an *objective*
must be verified by the solver.

## Method

The discipline, which is the part that transfers:

- **the solver is verified before anything is learned from it** — `flow-surrogate`
  reproduces published cylinder drag to a few percent after the standard blockage
  correction;
- **a non-neural baseline is always in the table** — a POD reduced-order model on
  seven parameters is hard to beat in-distribution.  A neural surrogate reported without
  one is not a result;
- **the metric is the engineering quantity**, not the loss — drag coefficient — reported
  next to the field error, because the two disagree;
- **generalisation is measured off-distribution**, on splits built to be outside the
  training range, not only on a random test split;
- **cost is measured end to end**, including the data generation, not just the forward
  pass.

## Layout

```
flow-surrogate/     LBM solver, surrogates, gradient-based shape optimisation
engine-copilot/     FastAPI + React platform on top, with a Claude Agent SDK copilot
```

`flow-surrogate` has its own `README.md` (the problem in equations, how to run it, what
the results mean), `BACKGROUND.md` (the mathematics behind the models, assuming none of
it), `RESULTS.md` (generated tables), `figs/` (all committed), `tests/`, and a
`run_all.py`.  Reading the results requires nothing to be re-run; only `data/` and
`runs/` are left out of the repo, being a few hundred megabytes of solver output and
checkpoints.

`engine-copilot` treats it as a read-only library and turns it into an interactive
demo — see [`engine-copilot/README.md`](engine-copilot/README.md), and
[`KNOWLEDGE-GRAPH.md`](KNOWLEDGE-GRAPH.md) for the navigation map of the whole repo.
