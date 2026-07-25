# Background

Everything you need in order to read this repo, assuming no finite elements and no deep
learning. The `README.md` states *what was measured*; this file explains *what the
objects are*. It is self-contained: linear algebra, multivariable calculus and the idea
of a random variable are assumed, nothing else.

Contents:

1. [The problem a surrogate solves](#1-the-problem-a-surrogate-solves)
2. [Elasticity in one page](#2-elasticity-in-one-page)
3. [The finite element method](#3-the-finite-element-method)
4. [Why the peak stress is 35 % low and the compliance is not](#4-why-the-peak-stress-is-35--low-and-the-compliance-is-not)
5. [Supervised learning, in the terms used here](#5-supervised-learning-in-the-terms-used-here)
6. [From a mesh to a graph](#6-from-a-mesh-to-a-graph)
7. [Message-passing networks](#7-message-passing-networks)
8. [The Cartesian-grid baseline](#8-the-cartesian-grid-baseline)
9. [Correction models and multi-fidelity learning](#9-correction-models-and-multi-fidelity-learning)
10. [Max, averaging, and the interpolation floor](#10-max-averaging-and-the-interpolation-floor)
11. [Ranking is not accuracy](#11-ranking-is-not-accuracy)
12. [Generalisation and what "discretisation invariance" really claims](#12-generalisation-and-what-discretisation-invariance-really-claims)
13. [Reading the tables honestly](#13-reading-the-tables-honestly)
14. [Glossary](#14-glossary)
15. [References](#15-references)

---

## 1. The problem a surrogate solves

### 1.1 Three objects

| object | here | cost |
|---|---|---|
| a **design** $d$ | 2–4 holes $(x_j, y_j, r_j)$ + a load angle $\alpha$ | — |
| a **solver** $d \mapsto u_d$ | P1 finite elements on a triangulated plate | 0.83 s (fine), 0.04 s (coarse) |
| a **quantity of interest** $s(d)$ | $\max_\Omega \sigma_{vm}$, the peak von Mises stress | free once $u_d$ is known |

A **surrogate** is a cheap approximation fitted to solver runs. What is unusual in this
project — and what makes it different from its sister repo `flow-surrogate` — is that
the input is not a fixed-size image. The number of holes changes, so the domain changes,
so the mesh changes: node count, connectivity, everything. A convolutional network has
nowhere to stand unless the part is first resampled onto a background grid, which is
exactly the comparison the repo runs.

### 1.2 The specific bargain here

The surrogate is not replacing the solver outright. It is doing something narrower and
more useful: **predicting the fine-mesh answer from the coarse mesh.**

$$\underbrace{\text{coarse mesh, }\sim1\text{k nodes, }0.04\ \mathrm s}_{\text{input}} \ \longrightarrow\ \underbrace{\text{fine-mesh field, } \sim25\text{k nodes, }0.83\ \mathrm s}_{\text{target}}$$

That framing matters. A coarse solve is not *wrong* — it is 0.1 % accurate on the
compliance — it is wrong specifically on the peak stress, by 35 %, for reasons §4 makes
precise. The surrogate's job is to undo a known, systematic, structural discretisation
error. That is a much better-posed learning problem than "learn elasticity".

---

## 2. Elasticity in one page

### 2.1 Stress, strain, equilibrium

A loaded solid deforms. Let $u(x) \in \mathbb R^2$ be the **displacement** of the material
point at $x$. Two objects describe what happens locally:

- the **strain** $\varepsilon = \frac12(\nabla u + \nabla u^\top)$, the symmetric part of
  the displacement gradient — the local stretching and shearing, with rigid rotations
  removed by the symmetrisation;
- the **stress** $\sigma$, a symmetric $2\times2$ tensor whose meaning is: for any
  internal surface with unit normal $n$, the force per unit area transmitted across it is
  the **traction** $t = \sigma n$.

Static equilibrium of every infinitesimal element, with no body force, is

$$-\nabla\cdot\sigma = 0 \quad\text{in }\Omega, \qquad \sigma n = t \ \text{ on the loaded boundary}, \qquad u = 0 \ \text{ on the clamp}.$$

### 2.2 Hooke's law and plane stress

The constitutive law — the material's contribution — is linear here:

$$\sigma = \mathsf D\,\varepsilon, \qquad \mathsf D = \frac{E}{1-\nu^2}\begin{pmatrix}1&\nu&0\\ \nu&1&0\\ 0&0&\frac{1-\nu}{2}\end{pmatrix}, \qquad \nu = 0.3,$$

in Voigt notation ($\sigma = (\sigma_{xx}, \sigma_{yy}, \sigma_{xy})$). $E$ is Young's
modulus (stiffness), $\nu$ Poisson's ratio (how much the material contracts sideways when
stretched). $E = 1$ here: the problem is linear in $E$, so its value only sets a scale.

**Plane stress** is the 2-D reduction appropriate to a thin plate loaded in its own plane:
$\sigma_{zz} = \sigma_{xz} = \sigma_{yz} = 0$ (nothing pushes on the free faces), while
$\varepsilon_{zz} \ne 0$ (the plate is free to thin). The other reduction, **plane
strain**, applies to a long prismatic body and swaps which quantities vanish; the only
difference is the coefficients in $\mathsf D$.

Two consequences of **linearity**, both used in the repo:

- the map $t \mapsto u$ is linear, so a load at angle $\alpha$ is
  $u(\alpha) = \cos\alpha\,u_x + \sin\alpha\,u_y$: two solves suffice for all angles. The
  load angle is therefore *not* where the difficulty of the learning problem lives;
- superposition also means there is no history, no path dependence, no yielding. Real
  parts leave this regime as soon as they yield, and nothing here transfers to plasticity,
  contact or buckling.

### 2.3 Von Mises stress, and why that one

$\sigma$ has three independent components in 2-D; a designer wants **one number** that
says "how close is this to failing". For ductile metals the standard answer is the von
Mises equivalent stress,

$$\sigma_{vm} = \sqrt{\sigma_{xx}^2 - \sigma_{xx}\sigma_{yy} + \sigma_{yy}^2 + 3\sigma_{xy}^2},$$

which comes from the $J_2$ (distortion-energy) yield criterion: a ductile material yields
when the *shape-changing* part of the strain energy reaches a critical value, independent
of hydrostatic pressure. Design rules are stated as $\max_\Omega \sigma_{vm} \le \sigma_y / \text{SF}$.

Note what $s(d) = \max_\Omega \sigma_{vm}$ is, as a functional: a **pointwise maximum** of
a **nonlinear function** of a **derivative** of the solution, attained on a **curved
boundary**. Every one of those four words costs accuracy somewhere in this repo.

### 2.4 Stress concentration

A hole in a loaded plate does not merely remove material — it *redistributes* the load
around itself, and the redistribution overshoots. **Kirsch's solution** (1898) for a
circular hole of radius $a$ in an infinite plate under uniaxial tension $\sigma_\infty$
gives, on the hole boundary,

$$\sigma_{\theta\theta}(\theta) = \sigma_\infty\big(1 - 2\cos2\theta\big) \quad\Longrightarrow\quad \max = 3\,\sigma_\infty \ \text{ at } \theta = \pm\tfrac\pi2,$$

a stress concentration factor of exactly 3, independent of the hole size. This is why
the peak lives on a hole flank, why it is *sharp*, and why resolving it matters.

Two holes close together interact through the **ligament** between them and the factor
rises well above 3 — which is precisely the regime a lightweighting study walks into, and
precisely where a coarse mesh fails hardest.

### 2.5 The singular corners

At a corner where a clamped edge meets a free edge, the elasticity solution is genuinely
**singular**: locally $\sigma \sim r^{\lambda-1}$ with $\mathrm{Re}\,\lambda < 1$ (Williams,
1952), so $\sigma \to \infty$ as $r \to 0$. The "peak stress" there is not a physical
number: refining the mesh keeps raising it, and the converged value does not exist.

In a 12-design spot check, 4 designs had their global peak at such a corner rather than on
a hole. The surrogate learns the $h = 0.010$ reference faithfully, which is the right
behaviour for a benchmark defined against that reference and the wrong thing to quote as a
physical stress. Real practice removes the singularity — a fillet, a compliant pad, or a
self-equilibrated load — and this is listed as a limitation in the README rather than
quietly ignored.

---

## 3. The finite element method

### 3.1 The weak form

$-\nabla\cdot\sigma = 0$ demands two derivatives of $u$. Multiplying by a test function
$v$ that vanishes on the clamp and integrating by parts moves one derivative onto $v$:

$$\underbrace{\int_\Omega \varepsilon(v):\mathsf D:\varepsilon(u)\,\mathrm d\Omega}_{a(u,v)} \;=\; \underbrace{\int_{\Gamma_\ell} t\cdot v\,\mathrm ds}_{\ell(v)} \qquad \forall v \in V,$$

with $V = \{v \in H^1(\Omega)^2 : v|_{\Gamma_c} = 0\}$. Three gains, and they are the
whole reason FEM is formulated this way:

- **less regularity is needed.** $u$ need only have square-integrable *first* derivatives
  ($H^1$), not second. Piecewise-linear functions qualify; they would not satisfy the
  strong form;
- **the natural boundary conditions come for free.** The traction condition
  $\sigma n = t$ is not imposed anywhere — it *falls out* of the boundary term of the
  integration by parts. Only the clamp ($u = 0$) has to be enforced explicitly. That is
  the distinction between *natural* and *essential* boundary conditions;
- **$a(\cdot,\cdot)$ is symmetric and positive definite**, so the problem is equivalent to
  minimising the potential energy $\frac12 a(u,u) - \ell(u)$, and the discrete system is a
  symmetric positive-definite linear system.

### 3.2 Galerkin discretisation with P1 triangles

Replace $V$ by a finite-dimensional $V_h \subset V$: continuous functions that are
**linear on each triangle** of a mesh, determined by their nodal values. Then

- $u_h$ is piecewise linear, so $\varepsilon(u_h)$ and $\sigma_h$ are **constant on each
  element** — hence the name **constant-strain triangle (CST)**;
- writing $u_h = \sum_i U_i N_i$ in the nodal basis, the weak form becomes a linear system

$$K U = f, \qquad K = \sum_e A_e\,B_e^\top \mathsf D\,B_e, \qquad f_i = \int_{\Gamma_\ell}N_i\,t\,\mathrm ds,$$

with $B_e$ the (constant) strain–displacement matrix of element $e$ and $A_e$ its area.
$K$ is the **stiffness matrix**: sparse, symmetric, positive definite once the clamp is
applied, and assembled by adding each element's small contribution into the global
matrix. In `stressgnn/fem.py` this is vectorised over elements and solved directly.

The load vector is worth one line: a uniform traction $t$ on a boundary edge of length
$|e|$ contributes $\frac12|e|\,t$ to each of its two endpoints — the "consistent" nodal
load, i.e. the exact integral of $t$ against the linear basis functions, not a lumped
guess.

### 3.3 Galerkin orthogonality

The single most useful fact about the method. Since $V_h \subset V$, the exact solution
satisfies the weak form against every $v_h \in V_h$, and so does $u_h$; subtracting,

$$a(u - u_h,\ v_h) = 0 \qquad \forall v_h \in V_h.$$

**The error is $a$-orthogonal to the entire discrete space.** Equivalently $u_h$ is the
$a$-orthogonal projection of $u$ onto $V_h$, and therefore the best approximation in the
energy norm $\lVert v\rVert_a = a(v,v)^{1/2}$:

$$\lVert u - u_h\rVert_a = \min_{v_h \in V_h}\lVert u - v_h\rVert_a.$$

Combined with interpolation estimates, this gives the classical rates for P1 elements on a
smooth solution:

| quantity | rate |
|---|---|
| $\lVert u - u_h\rVert_{H^1}$ (energy) | $\mathcal O(h)$ |
| $\lVert u - u_h\rVert_{L^2}$ (displacement) | $\mathcal O(h^2)$ |
| $\lVert \sigma - \sigma_h\rVert_{L^2}$ (stress) | $\mathcal O(h)$ |
| $\max\lvert\sigma - \sigma_h\rvert$ near a concentration | worse, and not covered by these estimates |

### 3.4 Verification

Before fitting anything to a solver, check the solver. `tests/test_fem.py` runs the
standard battery:

- the **patch test**: a mesh of arbitrary triangles subjected to a uniform strain must
  reproduce it exactly. Passing to $10^{-9}$ means the element is correctly formulated
  and the assembly has no bugs; it is the standard admissibility test for an element;
- an **energy balance**: $f^\top u = 2 \times$ strain energy, to $10^{-9}$;
- linearity, zero load, clamped-node kinematics, element areas against the analytic
  material area, and monotone convergence of the peak stress under refinement.

Plus a mesh-convergence study (`scripts/01_validate_fem.py`) which produces the table in
§4. None of this is optional: every ML number in the repo is an error *relative to this
solver*, so the solver's own error has to be known and bounded first.

---

## 4. Why the peak stress is 35 % low and the compliance is not

The mesh-convergence table (12 designs, errors relative to $h = 0.008$):

| $h$ | nodes | peak von Mises err | compliance err |
|---|---|---|---|
| 0.060 | 787 | **−28.4 %** | −2.60 % |
| 0.045 | 1 350 | **−24.7 %** | −1.64 % |
| 0.030 | 2 897 | −17.3 % | −0.79 % |
| 0.020 | 6 318 | −11.6 % | −0.35 % |
| 0.014 | 12 808 | −7.1 % | −0.14 % |
| **0.010 (reference)** | 24 772 | −3.2 % | −0.04 % |

Two columns, two completely different behaviours, and the theory explains both.

### 4.1 The compliance is protected

The **compliance** $C = f^\top u$ (twice the strain energy) is a *global energy* quantity.
Galerkin orthogonality gives it directly:

$$C - C_h = a(u,u) - a(u_h,u_h) = a(u - u_h,\ u - u_h) = \lVert u - u_h\rVert_a^2 = \mathcal O(h^2).$$

Two facts fall out at once:

- it converges at **twice** the energy-norm rate — a "superconvergent" functional;
- the error is **always negative**: a Galerkin solution is stiffer than the truth, because
  restricting the minimisation of the potential energy to a subspace can only raise the
  minimum. Every compliance entry in the table is indeed negative, without exception.

### 4.2 The peak stress is not

Four separate mechanisms, all pushing the same way:

1. **a derivative.** $\sigma_h = \mathsf D B u_h$ differentiates the solution, losing one
   order: $\mathcal O(h)$ at best in $L^2$;
2. **piecewise constant.** A CST cannot represent a *gradient* of stress inside an
   element, and a stress concentration is exactly a region of large stress gradient. The
   element's single value is essentially an average over it;
3. **an $L^\infty$ functional.** The maximum has none of the error-cancellation that an
   integral enjoys. The superconvergence argument of §4.1 relies on the functional being
   the energy itself; nothing similar exists for $\max$;
4. **geometric error.** A curved hole boundary is meshed as a polygon whose chords are
   $\mathcal O(h)$ long, and it is *on that boundary* that the peak lives. The coarse mesh
   literally cuts the corner where the answer is. Panel (c) of the setup figure in the
   README is this sentence drawn to scale.

Over the 400-design training set the coarse solve reads the peak **35 % low** — worse
than the 12-design table, because the tail of tight-ligament designs is where coarse
meshes fail hardest, and that tail is exactly what a lightweighting study explores.

**This is what the surrogate is for.** Not "learn elasticity" but "learn the map from a
stiff, cheap, biased discretisation to the converged one" — a structured, systematic,
one-sided error, which is a far better-posed target than a general PDE.

---

## 5. Supervised learning, in the terms used here

### 5.1 The recipe

Given $N$ pairs (a coarse mesh, the fine-mesh answer on its nodes), pick a family
$\{N_\theta\}$ and minimise an average loss:

$$\theta^\star = \arg\min_\theta \frac1N\sum_i \ell\big(N_\theta(G_i),\ (y_i, s_i)\big).$$

What one actually cares about is performance on samples never seen, so the data is split
`train` (400) / `val` / `test` (120), and the test split is looked at once. Two further
splits deliberately break the i.i.d. assumption:

- **`res`**: the *same* test designs re-meshed at $h = 0.025$, finer than anything in
  training ($h \sim U[0.035, 0.060]$). This isolates one variable — the discretisation —
  from everything else;
- **`ood`**: 4–6 holes (training tops out at 4) and load angles from a different quadrant.

Reporting all three side by side is what turns "the model works" into a statement with a
domain of validity attached.

### 5.2 Networks in one page

A network is a composition of alternating linear and nonlinear maps,
$N_\theta = f_L\circ\cdots\circ f_1$ with $f_\ell(z) = \sigma(W_\ell z + b_\ell)$. Without
the nonlinearity $\sigma$ the whole composition collapses to one linear map. An **MLP**
(multilayer perceptron) is this with dense $W_\ell$; here every $\phi$ in §7 is a
two-layer MLP with **GELU** activations and a final **LayerNorm** (which rescales each
vector to zero mean and unit variance — it keeps the residual updates of §7.2 from
drifting in scale as blocks are stacked).

Fitting is by **stochastic gradient descent**, in the Adam variant: gradients come from
**backpropagation** (reverse-mode automatic differentiation — the chain rule applied
backwards through the computation, at a cost comparable to one forward pass), and
parameters move downhill one mini-batch at a time. One pass over the data is an **epoch**;
every model here gets 35, chosen so that ten models fit in an hour on a laptop. None is
trained to convergence, which is stated as a limitation.

### 5.3 The loss

$$\mathcal L(\theta) = \frac{1}{|V|}\sum_{i \in V}(\hat y_i - y_i)^2 \;+\; \lambda_{\text{peak}}(\hat s - s)^2 \;\Big[+\ \lambda_{\max}\big(\max_i \hat y_i - \max_i y_i\big)^2\Big],$$

with $\lambda_{\text{peak}} = 0.3$, $\lambda_{\max} = 0$ by default, targets standardised
by training-set statistics.

The **second term is the important one** and it is the same design decision as the drag
head in the sister repo: the quantity of interest gets its own supervised output rather
than being read off the predicted field. §10 explains why that is not optional here — it
is forced by a bound that no nodal-field model can beat.

The bracketed term is an ablation (`-maxloss`) that pushes the *field*'s maximum toward
the target's. It makes things **worse** (peak error 6.6 % → 10.1 %): it fights the field
MSE for capacity, and the head that actually predicts the peak gets nothing from it. It is
reported because a plausible-sounding idea that fails is information.

---

## 6. From a mesh to a graph

![Graph explained](figs/graph_explained.png)

*(a) the mesh is the graph — nodes, mesh edges, and the features carried by each;
(b) one message-passing block; (c) how far $L$ blocks actually reach, measured on a real
coarse mesh; (d) what the Cartesian baseline does instead. Regenerate with
`python scripts/06_explainer_figures.py`.*

### 6.1 Why not a CNN

A convolutional network needs data on a regular grid: fixed shape, fixed neighbour
offsets, the same everywhere. A finite-element mesh has none of that — node count and
connectivity change with every design, and node *ordering* is an artefact of the mesher.
Two options follow, and the repo runs both:

- **resample onto a grid** and use a CNN (§8);
- **treat the mesh as what it already is: a graph.**

### 6.2 The graph

$G = (V, E)$ with $V$ the coarse mesh nodes and $E$ the mesh edges, stored in both
orientations so that messages travel both ways. Features:

$$v_i^{(0)} = \big[\ \mathrm{onehot}(\mathrm{type}_i) \in \mathbb R^5,\ \cos\alpha,\ \sin\alpha,\ h_i/L,\ x_i\ \big(,\ u_i^{c},\ \sigma_i^{c}\big)\big],$$

$$e_{ij}^{(0)} = \Big[\ \frac{x_j - x_i}{L},\ \frac{\lVert x_j - x_i\rVert}{L}\ \Big], \qquad L = 0.05.$$

Each entry earns its place:

- **node type** (interior / clamped / loaded / hole / free) is how the boundary conditions
  of §2.1 reach the network. Without it the network cannot know which edge is held and
  which is pulled;
- **the load angle** as $(\cos\alpha, \sin\alpha)$, not $\alpha$: a raw angle has a
  spurious discontinuity at $\pm\pi$, and the pair is what the physics actually depends
  on;
- **$h_i$**, the mean incident edge length — the resolution cue. §12 is about what happens
  when the network has to rely on it outside the trained range;
- **$x_i$**, the absolute position. Ablating it costs a lot (6.6 % → 12.0 %) and it is
  worth understanding why: on a *fixed* rectangular domain with a fixed clamp, absolute
  position is genuine information ("am I near the clamped edge?"), not a shortcut. On a
  problem with no fixed frame it would be the opposite;
- **$u^c, \sigma^c$** — the coarse solution — only in the `-corr` variant, §9.

### 6.3 The symmetries this buys

Because edges carry **relative** positions only, and because every node is processed by
the *same* MLP:

- **permutation equivariance.** Renumber the nodes and the output is renumbered with them.
  A CNN gets this for free (there is no numbering); a naive "flatten the mesh into a
  vector" model would have to learn it from data, and could not;
- **the same weights run on any mesh size.** 700 nodes or 3 000: nothing in the parameter
  shapes depends on $|V|$;
- **translation invariance** of the interactions, since only $x_j - x_i$ enters the edge
  features. Absolute position enters separately, as a node feature, which is a deliberate
  and reversible choice — hence the ablation.

Cost is $\mathcal O(|E|)$ per block, i.e. linear in mesh size, with no dense matrix
anywhere.

---

## 7. Message-passing networks

### 7.1 The general form

Almost every graph neural network is an instance of

$$v_i^{(\ell+1)} = \phi_v\Big(v_i^{(\ell)},\ \bigoplus_{j \in \mathcal N(i)} \phi_e\big(v_i^{(\ell)}, v_j^{(\ell)}, e_{ij}\big)\Big),$$

where $\bigoplus$ is a **permutation-invariant** aggregation over neighbours (sum, mean or
max — it must not depend on the order in which neighbours are listed). One application
propagates information across one edge. Stacking $L$ of them is the graph analogue of
stacking convolutions, and the analogy is close: a graph convolution is a stencil on an
irregular neighbourhood.

### 7.2 What this repo uses: MeshGraphNet

**Encode–process–decode** (Battaglia et al., 2018; Pfaff et al., 2021):

$$e_{ij} \leftarrow e_{ij} + \phi_e\big([e_{ij}, v_i, v_j]\big),$$
$$v_i \leftarrow v_i + \phi_v\Big(\big[v_i,\ \textstyle\sum_{j\in\mathcal N(i)} e_{ji},\ g\big]\Big),$$
$$g \leftarrow g + \phi_g\big([g,\ \mathrm{mean}_i\,v_i]\big),$$

then $\hat y_i = \phi_d(v_i)$ and $\hat s = \phi_p([\mathrm{mean}_i v_i,\ g])$. Three
details worth noticing:

- **edges carry state.** Unlike simpler GNNs, $e_{ij}$ is itself updated and remembered
  from block to block. On a mesh this is natural: an edge is a piece of geometry, and its
  state can encode the local stress transfer along it;
- **every update is residual** ($z \leftarrow z + \phi(\cdot)$). As with ResNets, this
  makes deep stacks trainable: each block learns a *correction*, and the gradient has a
  short path back through the identity;
- **hidden width 64**, $L \in \{0, 2, 5, 10, 15\}$, 18k–456k parameters.

### 7.3 The receptive field, and why $L$ is swept and not tuned

Information moves exactly one edge per block, so after $L$ blocks a node has seen its
$L$-hop ball, of radius $\approx L\,h$. Panel (c) of the figure measures this on a real
coarse mesh ($h = 0.050$): $L = 5$ reaches 0.28, $L = 10$ reaches 0.55, $L = 15$ reaches
0.77 — against a plate **2.0** long.

So a plain 10-block network **structurally cannot** see the clamped edge from the loaded
edge. That is not a tuning issue, it is an architectural bound, and it is why the sweep is
reported rather than a single tuned number:

| $L$ | 0 (pointwise MLP) | 2 | 5 | 10 | 15 |
|---|---|---|---|---|---|
| peak error | 11.5 % | 11.0 % | 7.2 % | 6.0 % | 5.5 % |
| parameters | 18k | 76k | 164k | 310k | 456k |
| inference | 1.5 ms | 2.5 ms | 4.0 ms | 6.4 ms | 8.9 ms |

Monotone and saturating. Why does it saturate rather than keep improving to $L=40$? Two
reasons pull in the same direction: elasticity is elliptic but the *concentration* is
local — most of what sets the peak lives within a few hole radii — and very deep
message-passing stacks suffer **over-smoothing**, where repeated neighbourhood averaging
drives node states toward a common value and destroys the local contrast the prediction
needs.

### 7.4 The global latent

$g$ is a vector pooled from all nodes and broadcast back to all nodes in every block —
equivalently a **virtual node** connected to everything. It gives an $\mathcal O(1)$-hop
shortcut for exactly the long-range coupling $L$-hop message passing cannot afford, far
cheaper than stacking 40 blocks.

Measured: in-distribution it does **not** help (6.6 % against 6.0 % without) and it
triples inference time; off-distribution it helps a lot (58 % against 79 %). The honest
summary in the README — "worth having, not worth defaulting to" — is what a table like
that supports.

---

## 8. The Cartesian-grid baseline

The "why not just voxelise it?" answer, included to be argued with rather than dismissed.

1. **rasterise** the part onto a fixed $96\times192$ image: a signed distance field
   $\varphi$ (negative in material), squashed by $\tanh(\varphi/0.05)$, plus an occupancy
   channel and the two load-angle channels;
2. run a **U-Net** — an encoder–decoder CNN. The encoder applies $3\times3$ convolution
   blocks and halves the resolution repeatedly; the decoder mirrors it; **skip
   connections** join equal resolutions. A convolution is a *learned stencil*: compare
   $3\times3$ weights against the finite-difference Laplacian
   $\frac{1}{h^2}[[0,1,0],[1,-4,1],[0,1,0]]$ and the analogy is exact. Pooling is what buys
   reach — the receptive field grows to roughly 96 cells, half the image — for the same
   reason multigrid exists;
3. **sample the prediction back at the mesh nodes** by bilinear interpolation, so the loss
   is evaluated at exactly the same points as for the graph models.

It is given a fair fight: grid spacing $dx = 0.0104$ matches the fine element size, same
training data, same loss, same evaluation points. **And it wins** — 5.2 % peak error
against 5.5–6.0 %, at 2.8 ms against 6.4–19.8 ms.

Why, and when it would not:

- this geometry family flatters a grid. The outer boundary is a fixed rectangle aligned
  with the axes; the only topology change is circular holes; at $dx = 0.0104$ a hole is
  ~20 cells across, so the SDF resolves it well;
- the grid's cost scales with the **bounding-box volume**, the graph's with the **number
  of mesh nodes**. In 2-D on a rectangle that ratio favours the grid. In 3-D the
  equivalent grid is $512^3 = 1.3\times10^8$ cells, most of them empty, and the ordering
  inverts;
- a grid cannot place a sample point *on* a curved boundary, which is where the peak
  lives. Here that costs less than the graph's own approximations. On thin features,
  curved outer boundaries, or graded meshes it would cost more;
- and §12: the grid model is mesh-*blind*, which turns out to be an advantage on the
  resolution split and a catastrophe off-distribution.

Reporting this cleanly, in the same table, is the point. A repo that only showed the graph
network would have hidden its most useful result.

---

## 9. Correction models and multi-fidelity learning

The `-corr` variant appends the coarse displacement and coarse von Mises value to each
node's features. The network therefore sees $\sigma_i^{c}$ and only has to supply the
**discretisation error** $y_i - \sigma_i^{c}$ rather than the field itself.

This is **multi-fidelity modelling**, the same idea as Kennedy & O'Hagan's (2000)
autoregressive co-kriging $y_{\text{high}} = \rho\,y_{\text{low}} + \delta$, and as
"delta learning" in computational chemistry. It works for the usual reason: the residual
is smaller, smoother and more nearly stationary than the field, so less has to be learned.

Note the implementation is *not* hard-wired: nothing in the architecture forces
$\hat y_i \approx \sigma_i^c$; the residual structure is available to be learned, not
imposed. Hard-wiring it ($\hat y_i = \sigma_i^c + \Delta_\theta$) is the obvious variant
and is not tested here.

Measured: by far the best on the **field** (rel $L_2$ 0.042 against 0.126–0.160) and the
most robust to an unseen mesh size (7.6 % against 13.5 %), at the price of a coarse solve
you were paying anyway. On the **peak** it is not better (6.0 %), which is a good reminder
that the field metric and the QoI metric measure different things.

---

## 10. Max, averaging, and the interpolation floor

![Interpolation floor](figs/interpolation_floor.png)

*(a) schematic: element→node averaging and restriction to coarse nodes, neither of which
commutes with $\max$; (b) the same two losses measured over the 400 training designs.
Both distributions are entirely on one side of 1.*

### 10.1 The two inequalities

The learning target is the fine von Mises field interpolated onto the coarse nodes.
Between the true peak and *any* nodal value sit two operators.

**Element → node averaging.** P1 element stresses are piecewise constant and noisy, so
they are area-averaged onto nodes:

$$\sigma_i^{\text{node}} = \frac{\sum_{e \ni i} A_e\,\sigma_e}{\sum_{e\ni i} A_e} \quad\Longrightarrow\quad \max_i \sigma_i^{\text{node}} \le \max_e \sigma_e,$$

strictly, at a concentration, because the maximal element is averaged with smaller
neighbours. An average of numbers cannot exceed their maximum — this is the whole
argument, and it costs ~8 % of the peak at $h = 0.010$.

**Restriction to the coarse node set.** $\max_{i \in V_{\text{coarse}}}(\Pi\sigma)(x_i) \le \max_{x\in\Omega}\sigma(x)$:
a finite set of sample points cannot contain the maximum of a continuous field unless it
happens to land there. Another ~2 %.

### 10.2 The consequence

Even a **perfect** model — $\hat y \equiv y$ exactly — reading the peak off its predicted
nodal field is **10.7 % low**. That is the `interpolation floor` row in the tables: a
property of the representation, not of any model. The only way under it is to
over-predict somewhere, which is not a capability worth relying on.

Hence the **graph-level head** $\hat s = \phi_p([\mathrm{mean}_i v_i, g])$: a separate
output that regresses the fine-mesh peak directly, not through the field. The measurement
confirms the analysis exactly:

| | peak from the head | peak from the predicted field |
|---|---|---|
| grid U-Net | **5.2 %** | 10.3 % |
| GNN, 15 steps | **5.5 %** | 10.6 % |
| GNN + global + coarse | **6.0 %** | 11.4 % |
| the floor | — | 10.7 % |

Every model's field-read peak sits **at the floor**, exactly as predicted, while the heads
sit at half of it. Both columns are reported so that the difference is visible rather than
quietly chosen.

The general lesson generalises well beyond this repo: **if your quantity of interest is a
functional that does not commute with the operators in your pipeline, give it its own
output.** The sister project's drag head exists for the identical reason.

---

## 11. Ranking is not accuracy

![Ranking vs accuracy](figs/ranking_vs_accuracy.png)

*Two strategies with the same truth: one with a large systematic bias and little scatter,
one unbiased with more scatter. The first ranks designs better. Synthetic draws calibrated
to the measured bias and Spearman of each strategy.*

### 11.1 The decomposition

Write any strategy's score as

$$\hat s(d) = \psi\big(s(d)\big) + \varepsilon(d),$$

with $s$ the truth, $\psi$ a systematic distortion and $\varepsilon$ scatter.

**Spearman's rank correlation** is the Pearson correlation of the *ranks*, so it depends
only on the ordering of the scores. If $\psi$ is increasing, applying it changes no
ordering:

$$\rho_{\text{Spearman}}(\hat s, s) \ \text{is invariant under any increasing } \psi,$$

and is degraded only by $\varepsilon$. A strategy can be 35 % low everywhere and still
rank perfectly.

Which is exactly what happens. The coarse solve is almost pure $\psi$ — a stiff
discretisation under-predicts every design in much the same way — while the surrogate is
*trained* to remove that $\psi$ (its peak bias is +1.6 %) and pays for it in $\varepsilon$
(p95 error 18.8 %). Measured over 150 candidate designs:

| strategy | Spearman | regret of its top-1 | regret of a verified top-3 | cost/design |
|---|---|---|---|---|
| fine FEM | 1.000 | 0 | 0 | 0.83 s |
| coarse FEM | **+0.892** | 2.07 % | **0.00 %** | 0.04 s |
| GNN (corrected coarse) | +0.856 | **1.41 %** | 1.41 % | 0.11 s |

### 11.2 Regret is the number that matters

Rank correlation is a summary; what a design study actually loses is the quality of the
design it makes you keep:

$$R_k = \frac{s\big(\arg\min_{d \in S_k} s(d)\big) - \min_d s(d)}{\min_d s(d)}, \qquad S_k = \{k \text{ lowest-scoring designs}\}.$$

$R_1$ is "trust the score and build it". $R_3$ is "shortlist three and verify them with
the real solver" — the realistic workflow, and the one that makes a mediocre ranker
usable. With a three-design shortlist the coarse solve finds the true optimum exactly.

### 11.3 The conclusion, and its scope

If the question is **"which of these designs is best?"**, a cheap biased solve answers it
and the ML is unnecessary. If the question is **"does this part pass at 250 MPa?"**, the
coarse solve is 35 % low and useless, and the 5 % surrogate earns its place. Design
exploration and certification are different jobs, and only the second one needs an
absolute number.

Note also the duality with the sister project. There, a surrogate used as an *objective*
had to be verified by the solver because the optimiser walks into its error (the winner's
curse). Here, a surrogate used as a *ranker* may not be needed at all. Same recipe, two
different verdicts, and both come from asking what the number is going to be used for.

---

## 12. Generalisation and what "discretisation invariance" really claims

### 12.1 The result that surprised me most

Re-mesh the *same* test designs at $h = 0.025$, finer than anything in training:

| | test ($h \in [0.035, 0.060]$) | unseen $h = 0.025$ |
|---|---|---|
| grid U-Net | 5.2 % | **5.1 %** |
| GNN, 10 steps | 6.0 % | 13.5 % |
| GNN + global, fed the coarse solve | 6.0 % | **7.6 %** |

The **mesh-aware** model degrades and the **mesh-blind** one does not.

This is not a paradox once stated carefully. The architecture is discretisation
*agnostic* — the same weights run on any mesh, §6.3 — but the trained function still
depends on mesh statistics, because those statistics were part of the training inputs. The
network learned a relationship between edge lengths, node degrees, $h_i$, and the answer,
calibrated on $h \in [0.035, 0.060]$. Handed $h = 0.025$, it extrapolates. Meanwhile the
U-Net never saw a mesh at all — it saw a rasterised image on a fixed grid — so it cannot
be surprised by one.

**"GNNs are discretisation invariant" is a claim about the hypothesis class, not about a
trained model.** The same caveat applies verbatim to neural operators and their
"zero-shot super-resolution": the architecture permits it, the training distribution
decides whether it holds.

What recovers the robustness is feeding the coarse FEM solution in as a node feature
(13.5 % → 7.6 %). That input already carries the resolution information — a coarse solve
on a finer mesh *is* a better answer — so the network is no longer forced to infer
resolution from mesh statistics alone.

### 12.2 Off-distribution, everything loses to the solver

On designs with 5–6 holes and load angles from an untrained quadrant:

| | test | OOD |
|---|---|---|
| coarse FEM, no ML | 35.8 % | **35.5 %** |
| MLP per node (18k params) | 11.5 % | 44.5 % |
| GNN + global + coarse | 6.0 % | 51.0 % |
| grid U-Net | **5.2 %** | 88.1 % |

Two things to read off it:

- **the coarse solve is the only row that barely moves.** It has no training distribution
  to leave. That is the entire argument for keeping a solver in the loop rather than
  retiring it: it degrades gracefully because it is not fitted to anything;
- **the ordering inverts.** The *least* expressive model generalises best and the most
  accurate one worst. A flexible model fits the training distribution more tightly,
  including the parts of it that do not extend outside — the bias–variance trade-off,
  showing up as an extrapolation trade-off. This is a specific, measured instance of the
  general principle that there is no such thing as a free inductive bias.

A surrogate is trustworthy inside its training envelope and actively harmful outside it.
Knowing where the envelope ends is therefore part of the deliverable, not an afterthought.

---

## 13. Reading the tables honestly

- **no error bars.** Each configuration is trained once, 35 epochs. Gaps of ~0.5 % between
  neighbouring rows are within seed noise; the gap between 5 % and 35 % is not. Read the
  tables at the resolution they support;
- **not trained to convergence.** 35 epochs was chosen so ten models fit in an hour. The
  *ranking* between architectures is stable across the message-passing sweep, but the
  absolute numbers would move with a longer schedule;
- **the reference is a discretisation, not an experiment.** Everything is measured against
  $h = 0.010$ FEM. The FEM's own model error (linear elasticity, plane stress, perfect
  clamp) is out of scope, and at the singular corners of §2.5 the reference does not
  converge at all;
- **the geometry family is favourable to the grid baseline** — circular holes in an
  axis-aligned rectangle. Result 2 of the README is a statement about *this* family, and
  is labelled as such;
- **120 test designs** puts a standard error of order $\text{std}/\sqrt{120}$ on every
  mean; the p95 columns are order statistics and are noisier still;
- **the design study is a single load case** with 150 candidates at one removed-area
  fraction. The Spearman gap (+0.892 vs +0.856) is small enough that its *sign* should not
  be over-read; the structural argument of §11.1 is the durable part, not the third
  decimal.

---

## 14. Glossary

| term | meaning here |
|---|---|
| **surrogate / emulator** | cheap approximation of an expensive solver |
| **QoI** | quantity of interest; here the peak von Mises stress |
| **compliance** | $f^\top u$, twice the strain energy; a global stiffness measure |
| **plane stress** | 2-D reduction valid for a thin plate loaded in-plane |
| **von Mises stress** | scalar equivalent stress from the $J_2$ yield criterion |
| **stress concentration** | local amplification of stress near a hole or notch (factor 3 for a circular hole) |
| **ligament** | the material bridge between two holes, or between a hole and an edge |
| **weak form** | integrated-by-parts version of the PDE, needing one derivative fewer |
| **essential / natural BC** | imposed on the space (clamp) / falling out of the weak form (traction) |
| **Galerkin** | solve the weak form in a finite-dimensional subspace |
| **Galerkin orthogonality** | the error is energy-orthogonal to the discrete space |
| **P1 / CST** | piecewise-linear triangle; constant strain per element |
| **stiffness matrix $K$** | sparse SPD matrix assembled from element contributions |
| **patch test** | uniform strain must be reproduced exactly; the standard element check |
| **$h$** | target element size; "coarse" $\approx 0.045$, "fine" $= 0.010$ |
| **superconvergence** | a functional converging faster than the field it comes from |
| **interpolation floor** | error a *perfect* nodal model still makes on the peak (10.7 %) |
| **graph / node / edge features** | the mesh as $(V, E)$ with per-node and per-edge vectors |
| **message passing** | one round of edge update + neighbour aggregation + node update |
| **permutation equivariance** | renumber the nodes, the output renumbers with them |
| **$L$-hop ball / receptive field** | what one node can see after $L$ blocks |
| **over-smoothing** | loss of node contrast when too many blocks are stacked |
| **global latent / virtual node** | a state pooled from and broadcast to all nodes |
| **encode–process–decode** | the MeshGraphNet template |
| **residual update** | $z \leftarrow z + \phi(\cdot)$; makes deep stacks trainable |
| **LayerNorm / GELU** | per-vector normalisation; smooth nonlinearity |
| **U-Net** | encoder–decoder CNN with skip connections |
| **rasterise / SDF** | draw the part onto a fixed grid as a signed-distance image |
| **multi-fidelity / delta learning** | learn the correction from a cheap model to an expensive one |
| **Spearman correlation** | correlation of ranks; blind to any increasing distortion |
| **regret** | how much worse the design you keep is than the true best |
| **in-distribution / OOD** | test data drawn like training data / deliberately outside it |

---

## 15. References

**Elasticity and finite elements**
- Timoshenko & Goodier, *Theory of Elasticity* — Kirsch's solution, stress concentration factors.
- Williams (1952), *Stress singularities resulting from various boundary conditions in angular corners* — the corner singularity of §2.5.
- Zienkiewicz, Taylor & Zhu, *The Finite Element Method*, vol. 1 — CST elements, assembly, the patch test.
- Strang & Fix (1973), *An Analysis of the Finite Element Method* — Galerkin orthogonality and convergence rates.
- Brenner & Scott (2008), *The Mathematical Theory of Finite Element Methods* — the estimates of §3.3 in full.
- Ainsworth & Oden (2000), *A Posteriori Error Estimation in Finite Element Analysis* — why stress errors behave differently from energy errors.

**Graph networks and mesh-based learning**
- Gilmer et al. (2017), *Neural message passing for quantum chemistry* — the MPNN framework of §7.1.
- Battaglia et al. (2018), *Relational inductive biases, deep learning, and graph networks* — encode–process–decode, global latents.
- Pfaff, Fortunato, Sanchez-Gonzalez & Battaglia (2021), *Learning mesh-based simulation with graph networks* (MeshGraphNet), ICLR — the architecture used here.
- Sanchez-Gonzalez et al. (2020), *Learning to simulate complex physics with graph networks*, ICML.
- Li et al. (2018), *Deeper insights into graph convolutional networks* — over-smoothing.
- Baque, Remelli, Fleuret & Fua (2018), *Geodesic convolutional shape optimization*, ICML — surrogates on CAD surfaces for design.

**CNN baselines and operator learning**
- Ronneberger, Fischer & Brox (2015), *U-Net: convolutional networks for biomedical image segmentation*.
- Guo, Li & Iorio (2016), *Convolutional neural networks for steady flow approximation*, KDD.
- Kovachki et al. (2023), *Neural operator: learning maps between function spaces*, JMLR — the discretisation-invariance claims discussed in §12.1.

**Multi-fidelity and design**
- Kennedy & O'Hagan (2000), *Predicting the output from a complex computer code when fast approximations are available*, Biometrika.
- Peherstorfer, Willcox & Gunzburger (2018), *Survey of multifidelity methods in UQ, inference and optimization*, SIAM Review.
- Forrester, Sóbester & Keane (2008), *Engineering Design via Surrogate Modelling* — surrogate-based design loops, shortlisting and verification.
- Jones, Schonlau & Welch (1998), *Efficient global optimization of expensive black-box functions* — the principled version of the sampling loop.
