<p align="center">
  <img src="docs/banner.png" alt="Surrogate Engineering — an expensive physics solver, a learned replacement for it, and the question of whether the optimum survives contact with the solver" width="880">
</p>

<p align="center">
  <a href="flow-surrogate">flow-surrogate</a>&nbsp; · &nbsp;<a href="stress-gnn">stress-gnn</a>&nbsp; · &nbsp;<a href="engine-copilot">engine-copilot</a>&nbsp; · &nbsp;<a href="flow-surrogate/RESULTS.md">results</a>&nbsp; · &nbsp;<a href="KNOWLEDGE-GRAPH.md">code&nbsp;map</a>
</p>

<p align="center">
  <sub>
    lattice Boltzmann &nbsp;·&nbsp; finite elements &nbsp;·&nbsp; POD / U-Net / FNO / MeshGraphNet
    <br>
    PyTorch (Apple MPS) &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; React + Vite &nbsp;·&nbsp; Claude Agent SDK
  </sub>
</p>

---

Two self-contained research repositories, each taking an expensive physics solver and
asking a different question about how to replace it with a learned model — and, on top of
them, an interactive platform that puts both results in an engineer's hands.

> **One more branch holds work in progress.**  `thermal-pinn` is a physics-informed network
> with no platform wiring yet.  It is not on `main`.

| | [`flow-surrogate`](flow-surrogate) | [`stress-gnn`](stress-gnn) |
|---|---|---|
| physics | 2-D incompressible flow (lattice Boltzmann) | 2-D linear elasticity (P1 finite elements) |
| geometry | parametric obstacle, **fixed Cartesian lattice** | perforated plate, **unstructured mesh, variable topology** |
| what varies | 7 shape coefficients + Reynolds number | 2-6 holes (position, radius) + load direction |
| surrogate input | rasterised level set | the mesh itself, as a graph |
| models compared | POD + ridge / POD + kernel ridge / U-Net / U-Net + physics loss / FNO | pointwise MLP / grid U-Net / MeshGraphNet (+ global latent, + coarse-solve input) |
| the question | *can I differentiate through the surrogate to optimise a shape, and does the optimum survive contact with the solver?* | *can a surrogate on a 1k-node mesh give 25k-node answers, and does it rank candidate designs better than a cheap solve?* |

Both run end to end on a laptop (Apple M-series, MPS).  Data generation is ~20-25 min per
project; training is minutes per model.

![flow-surrogate setup](flow-surrogate/figs/setup.png)
![stress-gnn setup](stress-gnn/figs/setup.png)

*The two problems, top and bottom: a body in a channel, parametrised by six Fourier
coefficients on a fixed 192x96 lattice; and a perforated plate, parametrised by 2-6 holes,
meshed differently for every design.*

## The platform

[`engine-copilot`](engine-copilot) wraps both research repos — untouched, as read-only
libraries — in a FastAPI backend and a React interface, so the surrogates can be *used*
rather than read about.  Move a Fourier coefficient and the drag coefficient follows in
about 60 ms; the fields, the streamlines and the interval move with it.  Drag a hole
across the plate and the von Mises map is re-meshed, re-solved and re-predicted in about
230 ms.

<p align="center">
  <img src="docs/flow-lab.png" alt="The Flow Lab: shape editor, predicted drag with its interval, velocity and pressure fields, streamlines, and the shape optimiser with its convergence trace" width="880">
</p>

*One real session, captured by [`docs/capture.py`](docs/capture.py) against a running
backend — not a mock-up.*

What the lab does, top to bottom:

- **Predict** — six sliders and a Reynolds number in, a drag coefficient out, next to the
  model's measured error, its confidence interval and the latency it took.  The three
  fields (axial velocity, transverse velocity, pressure) are the network's actual output.
- **Animate** — client-side RK4 streamlines through the predicted field, or the true LBM
  streamed frame by frame over SSE, in the same panel, on a toggle.
- **Optimise** — Adam through the frozen surrogate at constant projected area, streaming
  its convergence live.
- **Verify** — re-run the optimised design in the real solver.  This is the button the
  whole project is about, and it is deliberately not automatic.
- **Report** — export the run as Markdown.  A report produced before the verification
  says in as many words that nothing has confirmed the surrogate.
- **Ask** — a Claude Agent SDK copilot (⌘/Ctrl-J) that sees the design on screen and
  reaches the same services the interface does, through eight read-only tools called
  in-process.  It drives the local `claude` CLI, so there is no API key anywhere in this
  project.

The **Stress Lab** is the same loop on the other repo, minus the one thing the geometry
forbids: click to place a hole, drag it, scroll to resize it, and the von Mises field
re-meshes and re-predicts under it, next to the coarse FEM solve it has to beat — the
baseline typically misses the peak by a quarter to a third, and the panel says by how
much.  *Optimise* is a randomised local search over the surrogate at constant removed
area, not a gradient descent: the mesh topology changes on every hole move, so there is no
gradient to descend (which is the whole point of the comparison below).

<p align="center">
  <img src="docs/copilot.png" alt="The copilot sidebar: a question about the design on screen, the flow_predict call it makes, and an answer that quotes the expected error and the training range" width="400">
</p>

*The copilot is given the design on screen, so a question needs no preamble.  It never
states a drag coefficient without the error attached to it, and every tool call it makes
stays in the transcript — this one is a real turn, captured with the rest.*

<p align="center">
  <img src="docs/optimizer.png" alt="The optimiser panel: start shape, optimised shape, convergence trace, and the surrogate's prediction next to the solver's verdict" width="880">
</p>

*The optimiser lands on a design it scores at $C_d = 1.549$; the solver, re-run on that
same design, returns 1.432 — an 8.9 % gap between the surrogate's account of its own
optimum and the physics.  The committed run in `results/optimization.json` measures the
same effect at 11.3 %, against a drag cut of 18.6 % that the solver does confirm.
Surfacing that gap rather than hiding it is the point of the panel.*

## Results at a glance

**`flow-surrogate`** — 640 solver runs, 128 unseen test geometries.

| | field rel L2 | drag error | vs solver |
|---|---|---|---|
| POD + kernel ridge (ROM baseline) | 0.107 | 3.78 % | 19 800x |
| U-Net + continuity loss | **0.078** | **1.03 %** | 700x |

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

The surrogates are worth ~6x on their own distribution.  Everything else cuts against the
premise the project was built on, and is reported as such: the Cartesian baseline wins on
this geometry family; graph-network discretisation invariance degrades outside the trained
mesh-size range while the mesh-blind baseline does not; off-distribution every surrogate is
worse than the coarse solve it replaces; and in a 150-candidate design study the *biased*
coarse solve ranks designs slightly better (Spearman +0.892) than the debiased surrogate
(+0.856), because ranking cares about scatter and not about bias.

The conclusion that survives all of it: a surrogate earns its place where an *absolute*
number is needed inside a trained envelope, not where a cheap solve can already order the
candidates.

---

## The common problem, formally

Both repos instantiate the same three-stage template.  A design vector $d$ fixes a
geometry $\Omega(d)$; a PDE gives a field; a functional of that field is the number an
engineer actually wants:

$$\mathcal A\big(u;\,\Omega(d)\big) = 0,\qquad J(d) = Q\big(u_d\big).$$

Here $\mathcal A$ is steady Navier–Stokes or linear elastostatics, and $J$ is the drag
coefficient $C_d$ or the peak von Mises stress $\max_\Omega \sigma_{vm}$.  In both cases
$J$ is a *boundary* or *pointwise* functional of a *derivative* of $u$ — which is why
field accuracy and engineering accuracy disagree throughout both projects.  One evaluation
costs 1.7 s and 0.83 s respectively, so a thousand-design loop is hours.

**Stage 1 — sample and solve.**  Draw $\{d_i\}_{i=1}^N$, run the solver, keep both the
field and $J$.  **Stage 2 — fit** an encoder $X(d)$ and a network:

$$\theta^\star = \arg\min_\theta\ \frac1N\sum_{i=1}^N \ell\Big(N_\theta\big(X(d_i)\big),\ \big(u_{d_i},\, J(d_i)\big)\Big),$$

with $\ell$ carrying a field term *and* a term on $J$ itself, because a model fitted to
the field alone is not fitted to the quantity of interest.  **Stage 3 — put it in a
design loop:**

$$\min_{d\,\in\,\mathcal D}\ \hat J_{\theta^\star}(d)\qquad\text{s.t.}\qquad g(d) = 0,$$

the constraint being constant area in one project and constant removed area in the other —
in both cases so that the optimiser cannot cheat by removing material.

Everything that differs between the two repos follows from **the encoder $X$**.  A fixed
Cartesian lattice makes $X(d)$ a tensor of constant shape and, crucially, a
*differentiable* function of $d$, so stage 3 is gradient descent through the whole
pipeline.  An unstructured mesh makes $X(d)$ a graph whose node count and connectivity are
combinatorial functions of $d$; nothing differentiates, and stage 3 becomes a search over
sampled candidates.  That single choice is why the Flow Lab's *Optimise* button runs Adam
and the Stress Lab's runs a randomised search.

### The two failure modes, one per project

**Optimising against your own error.**  Write $\hat J = J + e$.  Minimising $\hat J$
does not sample $e$ neutrally — it actively seeks out the designs where $e$ is most
negative, so

$$\mathbb E\big[\,e(d)\ \big|\ d = \arg\min \hat J\,\big] \;<\; \mathbb E\big[\,e(d)\,\big]$$

even for a surrogate that is unbiased on its training distribution.  This is the
surrogate's winner's curse, and it is a statement about the predicted *value*, not about
the design.

Measured in `flow-surrogate`: **11.3 % off at the optimum, against 1.03 % on the test
split** — the error where the optimiser chose to stand is an order of magnitude larger than
the error the model advertises for itself, and the optimum sits on the boundary of the box
the model was trained in.  (In this run it lands on the pessimistic side: the solver finds
*less* drag than the surrogate promised.  The magnitude is the finding, not the sign.)
The design itself is genuinely 18.6 % better, verified by the solver, and one
active-learning round on the 32 designs the optimiser proposed halves the gap.

**Ranking is not accuracy.**  Write a strategy's score as $\hat s = \psi(s) + \varepsilon$
with $\psi$ increasing.  Rank correlation is *invariant* to $\psi$ and destroyed only by
$\varepsilon$: a solver that is 35 % low everywhere ranks designs almost perfectly, while
an unbiased surrogate with more scatter ranks them slightly worse.  Measured in
`stress-gnn`: Spearman +0.892 for the coarse solve against +0.856 for the surrogate that
corrects its bias.

The two are duals of each other.  The first says that a surrogate used as an *objective*
must be verified by the solver — which is why *Verify* is a button in the Flow Lab and not
an automatic step.  The second says that a surrogate used as a *ranker* may not be needed
at all, and that the case for it is the certification question ("does this part pass at
250 MPa?"), where the bias that ranking forgives is exactly what matters — which is why the
Stress Lab shows the coarse solve next to the prediction rather than hiding it.

## "Isn't that the same project twice?"

Mathematically, yes, and the section above says so up front: both are the same recipe.
Swap the operator and the loss and the two codebases would converge.

As engineering, no — and the reason is the one thing that actually decides which techniques
are available: **how the geometry is represented**.

**`flow-surrogate` lives on a fixed grid.**  Every sample has the same shape, so the problem
is an image-to-image problem and the whole convolutional / neural-operator toolbox applies.
That buys something specific: the pipeline from shape coefficients to predicted drag is one
differentiable graph, so shape optimisation is back-propagation, not a search.  The project
is built around that — and around the failure mode it creates, which is that an optimiser
will happily walk into the surrogate's own error until you close the loop with the solver.

**`stress-gnn` lives on unstructured meshes.**  Node count, connectivity and even the number
of holes change from sample to sample, so a CNN has nowhere to stand: it can only see the
part after resampling onto a grid.  The expectation going in was that this would cost it
accuracy at the stress concentrations on curved hole boundaries.  It was measured rather
than asserted — the grid U-Net is trained on the same data, scored at the same mesh nodes,
and reported in the same table — and it won.  Reading *why* it won, and what would have to
change for it not to, is most of what that repo is about.  The second question there is
different from the first project's: a design loop does not need exact stresses, it needs the
*ranking* of candidates to be right, and it turns out a cheap biased solve already provides
that.

Doing only one of them would leave out either the mesh-native representation problem or the
differentiable-optimisation story — and, as it turned out, only the pair of them shows that
the same recipe gives a clean win in one setting and a negative result in the other.

## Shared method, and what is deliberately not shared

Both repos follow the same discipline, because it is the part that transfers:

- **the solver is verified before anything is learned from it** — `flow-surrogate`
  reproduces published cylinder drag to a few percent after the standard blockage
  correction; `stress-gnn` passes a constant-stress patch test to 1e-9 and an
  energy-balance check;
- **a non-neural baseline is always in the table** — a POD reduced-order model on
  seven parameters is hard to beat in-distribution, and a coarse FEM solve is already a
  usable answer.  A neural surrogate reported without them is not a result;
- **the metric is the engineering quantity**, not the loss — drag coefficient, peak von
  Mises stress — reported next to the field error, because the two disagree;
- **generalisation is measured off-distribution**, on splits built to be outside the
  training range, not only on a random test split;
- **cost is measured end to end**, including the meshing and the data generation, not just
  the forward pass.

What is not shared: the two repos have separate environments, separate dependencies and no
common package.  They are meant to be read independently; only the platform imports both.

## Run it

Three independent parts, each with its own virtual environment.  The research repos stand
alone; the platform installs both as libraries.

```bash
# the science
cd flow-surrogate && python3.12 -m venv .venv && ./.venv/bin/pip install -e .
./.venv/bin/python run_all.py --quick        # generate -> train -> optimise
cd ../stress-gnn  && python3.12 -m venv .venv && ./.venv/bin/pip install -e .
./.venv/bin/python run_all.py --quick        # mesh -> solve -> train -> rank

# the platform (two servers, one browser tab)
cd ../engine-copilot && python3.12 -m venv .venv
./.venv/bin/pip install -e ../flow-surrogate -e ../stress-gnn -e .

# terminal 1 — the backend must run from backend/: the package is imported from the
# source tree, not pip-installed, so any other cwd gives ModuleNotFoundError
cd engine-copilot/backend  && PYTHONPATH=. ../.venv/bin/uvicorn engine_copilot.main:app --reload --port 8000
# terminal 2
cd engine-copilot/frontend && pnpm install && pnpm dev      # http://localhost:5173

# the copilot, if you want it — an existing Claude Code session, never an API key
npm install -g @anthropic-ai/claude-code && claude login
```

Reading the results requires none of it to be re-run: every table and figure is committed.

## Layout

```
flow-surrogate/     LBM solver, surrogates, gradient-based shape optimisation
stress-gnn/         FEM solver, mesh graph networks, design-ranking study
engine-copilot/     FastAPI + React platform on top, with a Claude Agent SDK copilot
docs/               README assets — the mark, the banner, the screenshots, and the
                    scripts that generate them
```

Each research repo has its own `README.md` (what it does, the result table, how to run it),
`RESULTS.md` (generated tables), `figs/` (all committed), `tests/`, and a `run_all.py`:
[`flow-surrogate`](flow-surrogate/README.md), [`stress-gnn`](stress-gnn/README.md).  Only
`data/` and `runs/` are left out of the repos, being a few hundred megabytes of solver
output and checkpoints.

`engine-copilot` treats both as read-only libraries and turns them into an interactive
demo — see [`engine-copilot/README.md`](engine-copilot/README.md), and
[`KNOWLEDGE-GRAPH.md`](KNOWLEDGE-GRAPH.md) for the navigation map of the whole repo.

<sub>The mark is not decoration: it is a dot "C" drawn as a lattice of cells clipped to an
annulus, where a cell's radius follows the horizontal axis and its colour the vertical one
on the measure ramp — a discretised field, which is what both repos are about.  It is
generated by <a href="docs/make_logo.py"><code>docs/make_logo.py</code></a>, never hand-drawn.
The interface, and everything in <code>docs/</code>, follows the <em>Calque</em> charte:
paper, one orange used only as a measure scale, one ink-filled element per screen.</sub>
