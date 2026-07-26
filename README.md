# Two surrogate-modelling projects

Two self-contained repositories, each one taking an expensive physics solver and
asking a different question about how to replace it with a learned model.

| | [`flow-surrogate`](flow-surrogate) | [`stress-gnn`](stress-gnn) |
|---|---|---|
| physics | 2-D incompressible flow (lattice Boltzmann) | 2-D linear elasticity (P1 finite elements) |
| geometry | parametric obstacle, **fixed Cartesian lattice** | perforated plate, **unstructured mesh, variable topology** |
| what varies | 7 shape coefficients + Reynolds number | 2-6 holes (position, radius) + load direction |
| surrogate input | rasterised level set | the mesh itself, as a graph |
| models compared | POD + ridge / POD + kernel ridge / U-Net / U-Net + physics loss / FNO | pointwise MLP / grid U-Net / MeshGraphNet (+ global latent, + coarse-solve input) |
| the question | *can I differentiate through the surrogate to optimise a shape, and does the optimum survive contact with the solver?* | *can a surrogate on a 1k-node mesh give 25k-node answers, and does it rank candidate designs better than a cheap solve?* |

Both run end to end on a laptop (Apple M-series, MPS).  Data generation is ~20-25 min
per project; training is minutes per model.

> **Not familiar with these models?**  Each repo carries a `BACKGROUND.md` that explains
> everything its README assumes, from first principles and with figures:
> [`flow-surrogate/BACKGROUND.md`](flow-surrogate/BACKGROUND.md) covers reduced-order
> models (POD + ridge / kernel ridge), U-Nets, physics-informed losses, Fourier neural
> operators, shape derivatives and the winner's curse;
> [`stress-gnn/BACKGROUND.md`](stress-gnn/BACKGROUND.md) covers linear elasticity, the
> finite element method and why its stress error behaves nothing like its energy error,
> message-passing graph networks, multi-fidelity correction models, and the statistics of
> ranking.  Neither assumes any prior CFD, FEM or deep learning.

![flow-surrogate setup](flow-surrogate/figs/setup.png)
![stress-gnn setup](stress-gnn/figs/setup.png)

*The two problems, top and bottom: a body in a channel, parametrised by six Fourier
coefficients on a fixed 192x96 lattice; and a perforated plate, parametrised by 2-6
holes, meshed differently for every design.*

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

**`stress-gnn`** — 400 training designs, 120 unseen test designs.

| | peak stress error | unseen mesh size | off-distribution |
|---|---|---|---|
| coarse FEM, no ML | 35.8 % | 24.4 % | **35.5 %** |
| U-Net on a Cartesian grid | **5.2 %** | **5.1 %** | 88.1 % |
| MeshGraphNet + global, fed the coarse solve | 6.0 % | 7.6 % | 51.0 % |

The surrogates are worth ~6x on their own distribution.  Everything else cuts against
the premise the project was built on, and is reported as such: the Cartesian baseline
wins on this geometry family; graph-network discretisation invariance degrades outside
the trained mesh-size range while the mesh-blind baseline does not; off-distribution
every surrogate is worse than the coarse solve it replaces; and in a 150-candidate
design study the *biased* coarse solve ranks designs slightly better (Spearman +0.892)
than the debiased surrogate (+0.856), because ranking cares about scatter and not about
bias.

The conclusion that survives all of it: a surrogate earns its place where an *absolute*
number is needed inside a trained envelope, not where a cheap solve can already order
the candidates.

---

## The common problem, formally

Both repos instantiate the same three-stage template.  A design vector $d$ fixes a
geometry $\Omega(d)$; a PDE gives a field; a functional of that field is the number an
engineer actually wants:

$$\mathcal A\big(u;\,\Omega(d)\big) = 0,\qquad J(d) = Q\big(u_d\big).$$

Here $\mathcal A$ is steady Navier–Stokes or linear elastostatics, and $J$ is the drag
coefficient $C_d$ or the peak von Mises stress $\max_\Omega \sigma_{vm}$.  In both cases
$J$ is a *boundary* or *pointwise* functional of a *derivative* of $u$ — which is why
field accuracy and engineering accuracy disagree throughout both projects.  One
evaluation costs 1.7 s and 0.83 s respectively, so a thousand-design loop is hours.

**Stage 1 — sample and solve.**  Draw $\{d_i\}_{i=1}^N$, run the solver, keep both the
field and $J$.  **Stage 2 — fit** an encoder $X(d)$ and a network:

$$\theta^\star = \arg\min_\theta\ \frac1N\sum_{i=1}^N \ell\Big(N_\theta\big(X(d_i)\big),\ \big(u_{d_i},\, J(d_i)\big)\Big),$$

with $\ell$ carrying a field term *and* a term on $J$ itself, because a model fitted to
the field alone is not fitted to the quantity of interest.  **Stage 3 — put it in a
design loop:**

$$\min_{d\,\in\,\mathcal D}\ \hat J_{\theta^\star}(d)\qquad\text{s.t.}\qquad g(d) = 0,$$

the constraint being constant area in one project and constant removed area in the
other — in both cases so that the optimiser cannot cheat by removing material.

Everything that differs between the two repos follows from **the encoder $X$**.  A fixed
Cartesian lattice makes $X(d)$ a tensor of constant shape and, crucially, a
*differentiable* function of $d$, so stage 3 is gradient descent through the whole
pipeline.  An unstructured mesh makes $X(d)$ a graph whose node count and connectivity
are combinatorial functions of $d$; nothing differentiates, and stage 3 becomes a
search over sampled candidates.

### The two failure modes, one per project

**Optimising against your own error.**  Write $\hat J = J + e$.  Minimising $\hat J$
does not sample $e$ neutrally — it actively seeks out the designs where $e$ is most
negative, so

$$\mathbb E\big[\,e(d)\ \big|\ d = \arg\min \hat J\,\big] \;<\; \mathbb E\big[\,e(d)\,\big]$$

even for a surrogate that is unbiased on its training distribution.  This is the
surrogate's winner's curse, and it is a statement about the predicted *value*, not about
the design.  Measured in `flow-surrogate`: 11.3 % optimistic at the optimum, while the
optimum itself is genuinely 18.6 % better, verified by the solver.  One active-learning
round on the 32 designs the optimiser proposed halves the gap.

**Ranking is not accuracy.**  Write a strategy's score as $\hat s = \psi(s) + \varepsilon$
with $\psi$ increasing.  Rank correlation is *invariant* to $\psi$ and destroyed only by
$\varepsilon$: a solver that is 35 % low everywhere ranks designs almost perfectly, while
an unbiased surrogate with more scatter ranks them slightly worse.  Measured in
`stress-gnn`: Spearman +0.892 for the coarse solve against +0.856 for the surrogate that
corrects its bias.

The two are duals of each other.  The first says that a surrogate used as an *objective*
must be verified by the solver.  The second says that a surrogate used as a *ranker* may
not be needed at all — and that the case for it is the certification question ("does this
part pass at 250 MPa?"), where the bias that ranking forgives is exactly what matters.

## "Isn't that the same project twice?"

Mathematically, yes, and the section above says so up front: both are the same recipe.
Swap the operator and the loss and the two codebases would converge.

As engineering, no — and the reason is the one thing that actually decides which
techniques are available: **how the geometry is represented**.

**`flow-surrogate` lives on a fixed grid.** Every sample has the same shape, so the
problem is an image-to-image problem and the whole convolutional / neural-operator
toolbox applies.  That buys something specific: the pipeline from shape coefficients
to predicted drag is one differentiable graph, so shape optimisation is
back-propagation, not a search.  The project is built around that — and around the
failure mode it creates, which is that an optimiser will happily walk into the
surrogate's own error until you close the loop with the solver.

**`stress-gnn` lives on unstructured meshes.** Node count, connectivity and even the
number of holes change from sample to sample, so a CNN has nowhere to stand: it can
only see the part after resampling onto a grid.  The expectation going in was that this
would cost it accuracy at the stress concentrations on curved hole boundaries.  It was
measured rather than asserted — the grid U-Net is trained on the same data, scored at
the same mesh nodes, and reported in the same table — and it won.  Reading *why* it won,
and what would have to change for it not to, is most of what that repo is about.  The
second question there is different from the first project's: a design loop does not need
exact stresses, it needs the *ranking* of candidates to be right, and it turns out a
cheap biased solve already provides that.

The second project is the closer analogue of learning directly on CAD-derived meshes;
the first is the closer analogue of the operator-learning literature and of
differentiable design.  Doing only one of them would leave out either the mesh-native
representation problem or the differentiable-optimisation story — and, as it turned
out, only the pair of them shows that the same recipe gives a clean win in one setting
and a negative result in the other.

## Shared method, and what is deliberately not shared

Both repos follow the same discipline, because it is the part that transfers:

- **the solver is verified before anything is learned from it** — `flow-surrogate`
  reproduces published cylinder drag to a few percent after the standard blockage
  correction; `stress-gnn` passes a constant-stress patch test to 1e-9 and an
  energy-balance check;
- **a non-neural baseline is always in the table** — a POD reduced-order model on
  seven parameters is hard to beat in-distribution, and a coarse FEM solve is already
  a usable answer.  A neural surrogate reported without them is not a result;
- **the metric is the engineering quantity**, not the loss — drag coefficient, peak
  von Mises stress — reported next to the field error, because the two disagree;
- **generalisation is measured off-distribution**, on splits built to be outside the
  training range, not only on a random test split;
- **cost is measured end to end**, including the meshing and the data generation, not
  just the forward pass.

What is not shared: the two repos have separate environments, separate dependencies
and no common package.  They are meant to be read independently.

## Layout

```
flow-surrogate/     LBM solver, surrogates, gradient-based shape optimisation
stress-gnn/         FEM solver, mesh graph networks, design-ranking study
```

Each has its own `README.md` (the problem in equations, how to run it, what the results
mean), `BACKGROUND.md` (the mathematics behind the models, assuming none of it),
`RESULTS.md` (generated tables), `figs/` (all committed), `tests/`, and a `run_all.py`.  Reading the results requires nothing to be re-run; only `data/` and
`runs/` are left out of the repos, being a few hundred megabytes of solver output and
checkpoints.
