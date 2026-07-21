# Background

Everything you need in order to read this repo, assuming no CFD and no deep learning.
The `README.md` states *what was measured*; this file explains *what the objects are*.
It is self-contained: linear algebra, multivariable calculus and the idea of a random
variable are assumed, nothing else.

Contents:

1. [The problem a surrogate solves](#1-the-problem-a-surrogate-solves)
2. [Supervised learning, in the terms used here](#2-supervised-learning-in-the-terms-used-here)
3. [Three error measures that disagree](#3-three-error-measures-that-disagree)
4. [Models 1–2: POD + regression](#4-models-12-pod--regression-the-reduced-order-models)
5. [What a neural network is, concretely](#5-what-a-neural-network-is-concretely)
6. [Model 3: the U-Net](#6-model-3-the-u-net)
7. [Model 4: the physics-informed U-Net](#7-model-4-the-physics-informed-u-net)
8. [Model 5: the Fourier Neural Operator](#8-model-5-the-fourier-neural-operator)
9. [Which model, when](#9-which-model-when)
10. [Differentiable geometry and shape optimisation](#10-differentiable-geometry-and-shape-optimisation)
11. [Optimising against your own error](#11-optimising-against-your-own-error)
12. [Reading the tables honestly](#12-reading-the-tables-honestly)
13. [Glossary](#13-glossary)
14. [References](#14-references)

---

## 1. The problem a surrogate solves

### 1.1 Three objects

Everything in this repo is an instance of the same three-object template.

| object | here | cost |
|---|---|---|
| a **design** $d$ | 7 shape parameters + Reynolds number | — |
| a **solver** $d \mapsto u_d$ | the lattice-Boltzmann code, returns the flow field | 1.74 s |
| a **quantity of interest** $J = Q(u_d)$ | the drag coefficient $C_d$ | free, once $u_d$ is known |

The solver is a map from a finite-dimensional design vector to an infinite-dimensional
object (a field), discretised here as a $3 \times 96 \times 192$ tensor. The engineer
does not want the field; they want $J$. But $J$ is only obtainable *through* the field,
and the field costs 1.74 s.

A **surrogate** (equivalently: emulator, metamodel, response surface, reduced-order
model, "AI solver") is any cheap approximation $\hat u_\theta \approx u_d$ or
$\hat J_\theta \approx J(d)$ fitted to a finite set of solver runs. The bargain is
always the same: pay a large one-off cost (here, 640 solver runs $\approx$ 19 min, plus
minutes of training) to make each *subsequent* evaluation ~$10^3$ times cheaper.

That bargain only pays off if you are going to ask the question many times. Optimisation,
uncertainty quantification, real-time design feedback and inverse problems all ask it
thousands of times; a single validation run does not, and for a single run you should
just use the solver.

### 1.2 Why not just fit $d \mapsto J$ directly?

You can, and the POD baselines almost do. Two reasons the models here predict the whole
field:

- **the field is 55 296 numbers of supervision per sample instead of 1.** With 640
  samples, a scalar regression sees 640 numbers total; a field regression sees $3.5\times10^7$.
  Predicting the field is a form of massive data augmentation — it forces the model to
  learn *why* the drag is what it is;
- **the field is what a downstream user asks for next.** Peak velocity, wake length,
  separation point, pressure distribution: all are functionals of the same field, and a
  field surrogate answers all of them without refitting.

The cost is that field accuracy and $J$ accuracy are different objectives, which is
§3 and one of the recurring themes of the repo.

### 1.3 Where the difficulty is

Not in the dimension of the design space — it is 7-dimensional, which is small. The
difficulty is that:

- the map $d \mapsto u_d$ is **nonlinear** and its output is high-dimensional;
- the geometry $\Omega_b(d)$ **moves**, so the same lattice cell is fluid for one design
  and solid for another. The map is not even continuous, cell by cell, as a function of
  $d$;
- $J$ is a **boundary integral of a derivative** of $u$: the worst kind of functional to
  approximate, because errors in $\nabla u$ near the wall matter far more than errors in
  $u$ in the far field, and an $L^2$ loss does not know that.

---

## 2. Supervised learning, in the terms used here

### 2.1 The recipe

Given $N$ pairs $(X_i, Y_i)$ drawn from some distribution — here $X_i$ is a
3-channel image encoding the geometry and $Re$, $Y_i$ is the field and the drag — pick a
family of functions $\{N_\theta\}$ indexed by parameters $\theta \in \mathbb{R}^P$, and
choose $\theta$ by minimising an average loss:

$$\theta^\star = \arg\min_\theta\ \underbrace{\frac{1}{N}\sum_{i=1}^N \ell\big(N_\theta(X_i), Y_i\big)}_{\text{empirical risk } \hat R(\theta)}.$$

The object one actually cares about is the **risk** $R(\theta) = \mathbb E\,\ell(N_\theta(X), Y)$
on samples never seen. The gap $R - \hat R$ is the generalisation gap, and the entire
discipline of the field is about not fooling yourself about it. Practically:

- **train / validation / test.** Fit on `train` (640), pick hyperparameters and stopping
  point on `val` (64), report on `test` (128), which is touched once. Any number obtained
  by looking at `test` and then changing the model is contaminated;
- **out-of-distribution.** `test` is drawn from the same distribution as `train`, so it
  measures interpolation, not extrapolation. The `ood` split (64 samples: higher $Re$,
  larger deformations) deliberately breaks that assumption, and every model roughly
  doubles its error on it. That is the honest measurement of what a surrogate is worth
  outside its envelope.

### 2.2 Capacity, over- and underfitting

A family $\{N_\theta\}$ that is too small cannot represent the true map (**underfitting**:
train and test errors both high). One that is too large can memorise the training set
(**overfitting**: train error low, test error high). The classical control is
**regularisation** — a penalty $\lambda\lVert\theta\rVert^2$, early stopping, or simply
fewer parameters.

Two remarks specific to this problem:

- the ROMs here are regularised explicitly (ridge parameter $\alpha$), the networks
  essentially only by early stopping at 40 epochs and by the smoothness of the data;
- with 640 samples and ~1.8 M parameters, the U-Net has vastly more parameters than
  data points and still does not overfit catastrophically. This is normal for
  overparametrised networks trained by gradient descent and is not a contradiction:
  parameter count is a poor proxy for capacity in this regime.

### 2.3 What "training" does

Minimisation is by **stochastic gradient descent**, in practice the Adam variant. The
dataset is cut into *batches*; for each batch, $\nabla_\theta \ell$ is computed by
**backpropagation** (the chain rule, applied backwards through the computational graph,
at a cost comparable to one forward evaluation) and $\theta$ is nudged downhill:

$$\theta \leftarrow \theta - \eta\,\widehat{m}/(\sqrt{\widehat v} + \epsilon),$$

with $\widehat m, \widehat v$ running estimates of the first and second moments of the
gradient. One pass over the dataset is an **epoch**; the models here get 40. The
learning rate $\eta$ is the single most important knob; Adam's per-coordinate scaling is
what makes a single $\eta$ work across layers with very different gradient magnitudes.

Nothing here is convex, so "the" minimum does not exist — different seeds give different
$\theta^\star$ with similar risk. That is why the comparison between architectures is
made at equal budget (40 epochs, same loss, same data) rather than by tuning each to
death.

### 2.4 Normalisation

The fields are standardised channel-wise over fluid cells of the training split,
$\tilde Y = (Y - \mu)/\sigma$, and the drag likewise. This is not cosmetic: an MSE loss
on raw $u, v, p$ would be dominated by whichever channel has the largest variance, and
gradient descent on badly scaled inputs is slow. Every reported number is de-normalised
back to physical units before the metric is computed — otherwise the numbers would
silently change with the normalisation statistics.

---

## 3. Three error measures that disagree

### 3.1 Relative $L_2$ on the field

For a predicted field $\hat Y$ against the solver's $Y$, restricted to the fluid mask $M$:

$$\text{rel } L_2 = \frac{\lVert M(\hat Y - Y)\rVert_2}{\lVert MY\rVert_2}, \qquad \lVert f \rVert_2 = \Big(\sum_{\text{cells}} f^2\Big)^{1/2}.$$

This is a *global average*. It is dominated by wherever the field is large and by
whatever occupies the most cells — here, the free stream and the wake. It is almost
blind to a thin boundary layer a few cells thick, which is exactly where drag comes from.

Reported per channel, because $u$, $v$, $p$ have very different difficulty: the
$u$-errors sit at 0.02, the $v$-errors at 0.13. $v$ is small in magnitude almost
everywhere, so the *relative* error has a small denominator; that is a property of the
normalisation, not a sign that transverse velocity is badly predicted.

### 3.2 MAPE on the drag

$$\text{MAPE} = \frac{1}{N}\sum_i \left|\frac{\hat C_{d,i} - C_{d,i}}{C_{d,i}}\right|,$$

plus the **worst case** $\max_i |\cdot|$, because a design loop is exposed to the tail,
not to the mean. The gap between the two columns is the interesting one: POD + kernel
ridge is at 3.78 % mean and 13.8 % worst case; the U-Net at 1.03 % and 4.2 %.

Why can a model be excellent in $L^2$ and mediocre on $C_d$? Because

$$C_d = \frac{1}{\frac12\rho U_\infty^2 D_{eq}}\oint_{\partial\Omega_b}\big(-p\,I + 2\mu S\big)\,n\,\mathrm ds$$

is a **boundary** integral of a **derivative**. Formally, $Q$ is not a bounded functional
on $L^2$: two fields can be arbitrarily close in $L^2(\Omega)$ and have arbitrarily
different traces of $\nabla u$ on $\partial\Omega_b$. Boundedness needs a stronger norm
(an $H^1$ or $H^{1/2}$ trace estimate), i.e. control of the derivative, which the
training loss never asks for. Reporting both numbers is the only way to see this.

### 3.3 The continuity residual

$$r_{\text{div}} = \big\langle|\nabla_h \cdot \hat u|\big\rangle_{\Omega_h}\cdot \frac{L}{U_\infty}, \qquad L = 20 \text{ cells}.$$

A physics-consistency check rather than an accuracy measure: an incompressible flow
satisfies $\nabla\cdot u = 0$ exactly, so any nonzero value is a statement that the
prediction is not a flow field. Dimensionless by construction, so it can be compared
across resolutions.

It has a floor: the LBM fields themselves score **0.006**, because LBM is weakly
compressible and the divergence is evaluated by central differences on a finite grid.
0.025 (U-Net) is therefore 4x the reference, and 0.109 (FNO) is 18x. Reading these
against zero would be wrong.

---

## 4. Models 1–2: POD + regression (the reduced-order models)

![POD explained](figs/pod_explained.png)

*(a) the mean snapshot and the first three POD modes of $u$, computed from the 640
training fields; (b) the singular-value spectrum — the first 64 modes carry 96.2 % of
the energy; (c) the rank-$r$ projection error of the test fields against what the two
fitted ROMs actually score. Regenerate with `python scripts/06_explainer_figures.py`.*

### 4.1 The idea

Two observations, and the whole method follows.

**Observation 1: the solutions are not scattered all over $\mathbb{R}^{55296}$.** They
are the flow past 640 mildly different bodies at mildly different Reynolds numbers. They
look like each other. If they lie close to a low-dimensional *linear* subspace, storing
the subspace plus a handful of coefficients is enough.

**Observation 2: the best such subspace is computable in closed form**, by the SVD.

### 4.2 The SVD and Eckart–Young

Stack the snapshots as rows of $Y \in \mathbb R^{N\times M}$ ($N = 640$ samples,
$M = 55296$ cells $\times$ channels), subtract the mean $\bar Y$, and take the singular
value decomposition

$$Y - \mathbf 1\bar Y^\top = U\Sigma V^\top, \qquad \Sigma = \mathrm{diag}(\sigma_1 \ge \sigma_2 \ge \dots \ge 0).$$

The rows $\psi_j$ of $V^\top$ are the **POD modes** (a.k.a. principal components, proper
orthogonal modes, empirical eigenfunctions, Karhunen–Loève basis — all the same object).
The **Eckart–Young–Mirsky theorem** says the rank-$r$ truncation is optimal: for every
rank-$r$ matrix $B$,

$$\lVert (Y - \mathbf 1\bar Y^\top) - B\rVert_F \ \ge\ \Big(\sum_{j>r}\sigma_j^2\Big)^{1/2},$$

with equality for the truncated SVD. **No** linear subspace of dimension $r$ reconstructs
these snapshots better. That is a strong statement and it is what makes POD the honest
baseline: any reduced-order model built on a linear subspace is at best this good.

The **energy** captured is $\sum_{j\le r}\sigma_j^2 / \sum_j \sigma_j^2$. Here:

| modes $r$ | 1 | 2 | 4 | 8 | 16 | 32 | **64** | 128 |
|---|---|---|---|---|---|---|---|---|
| energy | 33.2 % | 48.9 % | 65.6 % | 81.2 % | 89.6 % | 93.5 % | **96.2 %** | 98.1 % |

`n_modes = 64` is chosen there — the point where the spectrum flattens and the marginal
mode buys ~0.1 % of energy.

Panel (a) is worth looking at: mode 1 is a wake-strength mode (how far the deficit
extends), mode 2 a wake-width mode, mode 3 a boundary-layer/near-field mode. They are
not physical modes in any dynamical sense — they are the directions of maximal variance
of *this dataset*, and they would change if the sampling changed.

### 4.3 Offline / online, and where the 20 000x comes from

- **offline (once):** compute $\bar Y$, the modes $\psi_j$, and the training coefficients
  $\alpha_{ij} = \langle Y_i - \bar Y, \psi_j\rangle$. One SVD of a $640\times 55296$
  matrix — seconds.
- **online (per design):** predict $\hat\alpha(d) \in \mathbb R^{64}$, then
  $\hat Y = \bar Y + \sum_j \hat\alpha_j \psi_j$. The expensive part is a
  $64 \times 55296$ matrix–vector product, which is why it clocks 0.09 ms/sample batched
  — 19 800x faster than the solver, and 25x faster than the U-Net.

Note that the reported "parameter count" of the ROM (3.6 M) is the stored modes, not
fitted degrees of freedom. It is the fair analogue of a network's weights in memory, and
deliberately not the same thing as model capacity.

### 4.4 The regression $d \mapsto \alpha$

POD compresses the *output*. Something still has to map the design to the coefficients.
Two choices are compared, and they are the two ends of a spectrum.

**Ridge regression** (`pod-linear`). Fit $\hat\alpha = W d + b$ by

$$\min_W\ \sum_i \lVert W d_i + b - \alpha_i\rVert^2 + \lambda\lVert W\rVert_F^2 \quad\Longrightarrow\quad W = \alpha^\top D (D^\top D + \lambda I)^{-1},$$

a closed-form linear solve. The penalty $\lambda$ is what makes it well-posed when $D^\top D$
is ill-conditioned, and it trades variance for bias in the usual way. This model asserts
that the POD coefficients depend *linearly* on the shape parameters and $Re$. They do
not — hence 0.144 field error and 7.1 % drag error, the weakest model in the table.

**Kernel ridge regression** (`pod-rbf`). Replace $d$ by an implicit feature map
$\phi(d)$ living in a reproducing-kernel Hilbert space and fit a linear model *there*.
The representer theorem says the solution is a combination of kernels centred on the
training points:

$$\hat\alpha(d) = \sum_{i=1}^N \beta_i\,k(d, d_i), \qquad \beta = (K + \lambda I)^{-1}\alpha, \qquad K_{ij} = k(d_i, d_j),$$

with the Gaussian (RBF) kernel $k(d,d') = \exp(-\gamma\lVert d - d'\rVert^2)$. Two ways
to read this:

- **as interpolation.** The prediction is a weighted average of the training
  coefficients, weighted by proximity in design space, with $\gamma$ setting "how close
  is close" and $\lambda$ smoothing;
- **as a Gaussian process.** $(K+\lambda I)^{-1}$ is exactly the posterior mean of GP
  regression with covariance $k$ and noise $\lambda$ — i.e. kernel ridge *is* kriging,
  the workhorse of design-of-experiments since the 1990s.

This is a genuinely strong baseline, and the reason is dimensional: the design space is
7-dimensional and smooth, which is precisely the regime where kernel methods shine.
0.107 field error and 3.78 % drag error, fitted in under a second.

### 4.5 What limits it — three separate ceilings

Panel (c) of the figure separates the first two.

1. **the basis.** Projecting the *test* fields onto the rank-64 training subspace — the
   best any regression on this basis could do — gives rel $L_2 = \mathbf{0.043}$. The
   fitted ROM scores 0.107. The two contributions are orthogonal (the projection residual
   is orthogonal to the subspace, the regression error lies inside it), so they add in
   quadrature: the regression's own share is $\sqrt{0.107^2 - 0.043^2} = 0.098$, i.e.
   almost all of it. **The basis is not the bottleneck; the map $d\mapsto\alpha$ is.**
   Doubling the number of modes would buy nothing;
2. **the sample count.** The kernel ridge model saturates at ~300 solver runs (0.108 at
   $N=320$, 0.107 at $N=640$): it has extracted what a linear subspace over 7 smooth
   parameters has to give. The U-Net crosses it there and keeps improving. This crossover
   is the practical decision rule in the README;
3. **the input.** The regressor consumes the *Fourier coefficients*. Change the CAD
   template — add a mode, switch to splines, use a mesh — and the model is worthless.
   The image-based models never see the parametrisation, only a rasterised geometry.

There is a fourth ceiling that this problem does not hit but which is why ROMs are not
the end of the story in general: the **Kolmogorov $n$-width**. Define

$$d_n(\mathcal M) = \inf_{\dim V_n = n}\ \sup_{u \in \mathcal M}\ \inf_{v \in V_n}\lVert u - v\rVert,$$

the best achievable error of *any* $n$-dimensional linear subspace on the solution
manifold $\mathcal M$. For elliptic and diffusion-dominated problems $d_n$ decays
exponentially and ROMs are spectacular. For advection-dominated problems — a sharp front
or a shock that *moves* with the parameter — $d_n$ decays like $n^{-1/2}$, and no linear
method is viable, because translating a discontinuity is not a low-rank operation. Here
the bodies are all near the same position at moderate $Re$, so the spectrum decays fast
(96 % in 64 modes) and POD works. Move the body across the channel and it would not.

---

## 5. What a neural network is, concretely

Before the U-Net and the FNO, the minimum needed to read them.

### 5.1 Layers

A network is a composition of simple maps, alternating linear and nonlinear:

$$N_\theta = f_L \circ \cdots \circ f_1, \qquad f_\ell(z) = \sigma\big(W_\ell z + b_\ell\big).$$

Without $\sigma$, the composition collapses to a single linear map. The nonlinearity is
the entire point, and the specific choice matters less than its existence; this repo uses
**GELU**, $\sigma(x) = x\,\Phi(x)$ with $\Phi$ the Gaussian CDF — a smooth ReLU, and its
smoothness matters here because §10 differentiates through the network.

An **MLP** (multilayer perceptron) is the version where $W_\ell$ is a dense matrix. It is
used here only for the small heads.

### 5.2 Convolution

For image-shaped data, a dense $W$ is both enormous and wrong: it would learn a separate
weight for every (input pixel, output pixel) pair, ignoring the fact that physics is
local and the same everywhere. A **convolution** replaces it with a small kernel applied
identically at every position:

$$(K * z)(x) = \sum_{\lVert\delta\rVert_\infty \le 1} K_\delta\, z(x + \delta) \qquad (3\times3 \text{ kernel}).$$

Compare with a finite-difference Laplacian,

$$\nabla_h^2 z = \frac{1}{h^2}\begin{pmatrix}0&1&0\\1&-4&1\\0&1&0\end{pmatrix} * z,$$

and the analogy is exact: **a convolution layer is a stencil whose coefficients are
learned rather than derived.** A network of them is a learned, nonlinear,
multi-resolution stencil scheme. This is why CNNs are a natural hypothesis class for
PDE fields, and it also tells you their two structural properties:

- **translation equivariance** — shift the input, the output shifts. Correct for the
  physics away from the walls, wrong near them, which is why the FNO explicitly appends
  coordinate channels;
- **locality** — one layer sees 3 cells. Which is a problem, and §6.2 is how it is fixed.

Each layer carries $C_{\text{out}}$ independent kernels ("channels"), so a layer with
$C_{\text{in}} = C_{\text{out}} = 64$ holds $3\times3\times64\times64 \approx 37$k
weights regardless of image size.

**GroupNorm** rescales activations within groups of channels to zero mean and unit
variance before the nonlinearity. It stabilises optimisation and is used instead of
BatchNorm because it does not couple samples within a batch — relevant when the batch is
small and when a single sample is pushed through at optimisation time.

---

## 6. Model 3: the U-Net

![Architectures](figs/architectures.png)

*The three families side by side. (a) POD + regression; (b) the U-Net; (c) one FNO layer.
Regenerate with `python scripts/06_explainer_figures.py --quick`.*

### 6.1 The shape of the network

The U-Net (Ronneberger et al., 2015, originally for biomedical image segmentation) is
an **encoder–decoder with skip connections**:

- the **encoder** applies two conv blocks, halves the resolution by max-pooling, doubles
  the channel count, and repeats. Here $96\times192\times32 \to 48\times96\times64 \to 24\times48\times128 \to 12\times24\times256$;
- the **decoder** mirrors it, upsampling by bilinear interpolation;
- **skip connections** concatenate the encoder feature map at each resolution onto the
  matching decoder one.

1.8 M parameters, 2.6 ms per sample.

### 6.2 Why the pyramid — the receptive field argument

The **receptive field** of an output cell is the set of input cells that can influence it.
For a stack of $3\times3$ convolutions it grows by 2 cells per layer, which is hopeless:
reaching across a 192-cell channel would take ~96 layers.

Pooling changes the arithmetic. Tracking the receptive field $r$ and the accumulated
stride $j$ through the network ($r \leftarrow r + (k-1)j$ per conv, $j \leftarrow 2j$ per
pool):

| after | $r$ (cells) | $j$ |
|---|---|---|
| first conv block | 5 | 1 |
| level 1 block | 14 | 2 |
| level 2 block | 32 | 4 |
| bottleneck | **68** | 8 |
| output (through the decoder) | **≈ 96** | 1 |

So the pyramid buys a 96-cell reach — half the channel, the full height — out of
$3\times3$ stencils. This is not a detail: incompressible flow is **elliptic** in the
pressure. $\nabla^2 p = -\nabla\cdot((u\cdot\nabla)u)$ means the pressure at any point
depends on the entire domain instantaneously, and a purely local model cannot represent
that. The same insight underlies **multigrid** solvers, which fix the identical problem
(local relaxation cannot propagate long-wavelength information) with the identical
device: a hierarchy of resolutions.

The drag head sidesteps the question entirely: $\hat C_d = \mathrm{MLP}(\langle z\rangle_x)$
reads a *global average* over the bottleneck, so it sees everything by construction.

### 6.3 Why skip connections

The bottleneck is $12\times24$: the fine detail of the boundary layer cannot survive
there. Skips give the decoder direct access to the high-resolution encoder features, so
the deep path only has to carry the *global* information (where the wake goes, how strong
it is) and the shallow path carries the *local* detail (where the wall is). Empirically
they also make the optimisation dramatically easier — they give gradients a short path
back to early layers.

### 6.4 The loss

$$\mathcal L(\theta) = \big\lVert M \odot(\hat Y - Y)\big\rVert^2_{\text{MSE}} + \lambda_{cd}\big(\hat C_d - C_d\big)^2, \qquad \lambda_{cd} = 0.2.$$

Two deliberate choices:

- the **mask** $M$: the loss is evaluated on fluid cells only. Inside the body there is
  no flow, and asking the network to fit zeros there wastes capacity and biases the
  boundary;
- the **explicit drag term**. Without it the model is fitted to the field and merely
  hopes the drag follows. Since §3.2 says it does not follow, the quantity of interest
  gets its own supervised head. This is the single most important design decision in the
  repo and it appears again, identically, in the sister project's peak-stress head.

### 6.5 What "geometry-agnostic" means

The U-Net's input is $[\tanh(\varphi/L_s),\ \mathbf 1_{\{\varphi<0\}},\ Re/Re_s]$ — a
rasterised **level set**, a mask and a scalar. It never sees $\mathbf c$. Consequently it
would run unchanged on an aerofoil, an ellipse, or a shape drawn by hand, as long as it
can be rasterised on the same lattice; whereas the POD model would have to be refitted
from scratch. Whether it would be *accurate* on those is a different question — that is
the `ood` split's job, and the answer is "degraded but ranked the same".

The level set is
$\varphi(x) = \lVert x - x_c\rVert - r(\theta(x))$: negative inside, zero on the boundary,
positive outside. It is a *radial* signed distance, not a Euclidean one; it is exact on
$\partial\Omega_b$ and monotone across it, which is what the network needs. The $\tanh$
squash keeps far-field cells at a small but nonzero gradient with respect to the shape,
which is what §10 needs.

---

## 7. Model 4: the physics-informed U-Net

### 7.1 Soft physics constraints

The general idea (Raissi et al., 2019, "physics-informed neural networks") is to add the
PDE residual to the loss, so the network is penalised for producing fields that are not
solutions. In its pure form, PINNs use *only* the residual and no data. What is done here
is the hybrid version, and the weaker claim: data-driven fitting with a physics term as a
**regulariser**.

$$\mathcal L \mathrel{+}= \lambda_{\text{phys}}\left(\frac{L}{U_\infty}\right)^2 \big\langle(\nabla_h\cdot\hat u)^2\big\rangle_{\Omega_h}, \qquad \lambda_{\text{phys}} = 0.02.$$

Only the continuity equation is used, not the momentum equations. Three reasons:
$\nabla\cdot u = 0$ needs one derivative rather than two (cheap, and much less noisy on a
finite grid); it involves no unknown constants; and it is a *hard* constraint of the
physics — any admissible field satisfies it exactly, so any violation is unambiguously an
error.

### 7.2 The dimensionless weight

Note $(L/U_\infty)^2$. Without it, $\lambda_{\text{phys}}$ would carry units and its
meaning would change with the normalisation statistics. The residual is evaluated on the
**de-normalised** prediction, so the term measures a physical divergence and
$\lambda_{\text{phys}} = 0.02$ means the same thing next month.

### 7.3 Soft is not hard, and the measurement says so

A penalty is not a constraint. Minimising $\mathcal L_{\text{data}} + \lambda \mathcal L_{\text{phys}}$
gives a solution where the two gradients balance, not one where the residual vanishes.
Enforcing $\nabla\cdot u = 0$ *exactly* would need a different construction — predicting a
streamfunction $\psi$ and setting $u = \nabla^\perp\psi$ makes it identically true by
construction, at the cost of assuming 2-D and needing a Helmholtz decomposition to get
$p$.

The measured effect: drag error 1.15 % $\to$ **1.03 %**, divergence residual 0.027 $\to$
0.025 against the solver's own 0.006. So at this weight the term helps as a mild
regulariser and does *not* enforce physics. The README says so; that is the correct
reading, and the alternative reading ("physics-informed learning improves the surrogate")
would be overselling a 0.002 change in the residual.

The honest experiment that is *not* run here would be a sweep in $\lambda_{\text{phys}}$
to find where the residual starts moving and where the field error starts degrading. It
is the obvious next measurement.

---

## 8. Model 5: the Fourier Neural Operator

### 8.1 Operator learning

The framing is one level more abstract than "image to image". A CNN learns a map between
*tensors of a fixed shape*. A **neural operator** aims at a map between *function spaces*,

$$\mathcal G: a \mapsto u, \qquad a \in \mathcal A(\Omega),\ u \in \mathcal U(\Omega),$$

here from a geometry field to a velocity field, such that the same weights apply at any
discretisation. If that is achieved, one can train on a $96\times192$ grid and evaluate
on $192\times384$ — "zero-shot super-resolution" — which no CNN can do meaningfully,
because a $3\times3$ stencil means a different physical length at a different resolution.

The two standard constructions are **DeepONet** (Lu et al., 2021: a branch net encoding
$a$, a trunk net encoding the query point, combined by an inner product) and the **FNO**
(Li et al., 2021), which is what is implemented here.

### 8.2 The kernel integral layer

Generalise a convolution to a continuous integral operator,

$$(\mathcal K z)(x) = \int_\Omega \kappa(x, y)\, z(y)\,\mathrm dy.$$

If the kernel is translation-invariant, $\kappa(x,y) = \kappa(x-y)$, the **convolution
theorem** turns this into a pointwise product in Fourier space:

$$\mathcal K z = \mathcal F^{-1}\big[\widehat\kappa \cdot \mathcal F z\big].$$

The FNO's move is to **never parametrise $\kappa$ at all** and instead learn
$\widehat\kappa$ directly — one complex matrix $R_\phi[k] \in \mathbb C^{C\times C}$ per
retained wavenumber $k$. Truncating to $|k_1|, |k_2| \le 16$ keeps the parameter count
finite. One layer is

$$z^{(\ell+1)} = \sigma\Big(W z^{(\ell)} + \mathcal F^{-1}\big[R_\phi\cdot\mathcal F z^{(\ell)}\big]\Big),$$

with $W$ a $1\times1$ convolution — a local, pointwise path that carries what the
truncated spectral path throws away. Four such layers, width 32, 2.1 M parameters.

Note what this buys: a **global** receptive field in a single layer, for
$\mathcal O(n\log n)$ instead of $\mathcal O(n^2)$. Where the U-Net needs a pyramid to
reach across the domain, the FNO reaches across it by construction. For smooth,
periodic-ish problems (Darcy flow, Burgers, homogeneous turbulence) this is an excellent
inductive bias, and the FNO's reputation comes from exactly those benchmarks.

### 8.3 Why it underperforms here

![Fourier truncation](figs/fourier_truncation.png)

*(a) the occupancy channel of a real test sample; (b) the same after keeping only the
$16\times16$ modes the FNO retains; (c) a horizontal cut, with a 32-mode reconstruction
added — the overshoot narrows but does not shrink, which is Gibbs' phenomenon;
(d) radial energy spectra: the velocity field decays 3 decades by $|k|=45$, the geometry
barely one.*

Fourier bases are global and smooth. A geometry channel is a **step function** across
$\partial\Omega_b$, and its Fourier coefficients decay like $1/|k|$ — panel (d). Three
consequences, all visible in the figure:

- **Gibbs' phenomenon.** Truncating a Fourier series at a jump discontinuity produces an
  overshoot of ~9 % of the jump that does *not* decay as more modes are added; it only
  gets narrower. Panel (c), 16 versus 32 modes, is that statement measured;
- **the ringing is not local.** Panel (b): the error from a boundary a few cells wide is
  spread over the whole channel, because every retained mode has global support. The
  U-Net's mistakes stay near the body; the FNO's do not, and the field-comparison figure
  in the README shows this directly;
- **the boundary moves.** Each sample has its discontinuity in a different place, so the
  network cannot learn a fixed correction for it.

The result is the interesting one: **the FNO matches the U-Net on field error (0.080 vs
0.081) and loses on physics (divergence residual 0.109 vs 0.027, 18x the solver's floor)
and on drag (1.94 % vs 1.15 %).** A global-average metric hides the ringing; a derivative
functional does not. This is §3.1 and §3.2 again in a different costume.

The general lesson is not "FNO is bad". It is that **operator learning in a fixed
Fourier basis assumes the geometry is a smooth coefficient field, and CAD geometry is
not**. Variants exist that address exactly this — Geo-FNO learns a deformation to a
reference domain, and factorised/local versions restore locality — and would be the
right next thing to try.

---

## 9. Which model, when

| | POD + ridge | POD + kernel ridge | U-Net | U-Net + phys | FNO |
|---|---|---|---|---|---|
| what it is | linear subspace + linear map | linear subspace + kriging | learned multiscale stencil | idem + divergence penalty | learned spectral multiplier |
| input | 7 CAD params + $Re$ | idem | rasterised geometry | idem | idem + coordinates |
| field rel $L_2$ | 0.144 | 0.107 | 0.081 | **0.078** | 0.080 |
| drag MAPE | 7.08 % | 3.78 % | 1.15 % | **1.03 %** | 1.94 % |
| div. residual (LBM: 0.006) | 0.024 | 0.026 | 0.027 | 0.025 | 0.109 |
| train | seconds | seconds | 4 min | 4 min | 8 min |
| inference vs solver | 17 600x | 19 800x | 709x | 713x | 343x |
| needs a fixed parametrisation | yes | yes | no | no | no |
| differentiable in the design | via the regression | via the regression | **yes, end to end** | yes | yes |
| best below ~300 samples | | **yes** | | | |

The practical decision rule the repo argues for:

- **fewer than ~300 solver runs available** → build the POD + kernel ridge ROM. It is
  free, it is 25x faster at inference, and at $N=80$ it is *better* (0.139 vs 0.184);
- **past a few hundred, and you need the tail** → a network. Its advantage shows up in
  worst-case drag error long before it shows up in the mean;
- **the geometry family will change** → a network, necessarily; the ROM is welded to its
  parametrisation;
- **you want gradients through the surrogate** → a network on a fixed grid, which is
  §10 and the reason this project is set up on a lattice at all.

---

## 10. Differentiable geometry and shape optimisation

### 10.1 The classical picture

Shape optimisation asks $\min_{\mathbf c} J(\mathbf c)$ where evaluating $J$ requires a
PDE solve. Finite differences would need one solve per design variable per iteration
(here $6\times250\times16 = 24000$ solves, ~11 hours of LBM). The classical fix is the
**adjoint method** (Pironneau; Jameson, 1988): solve the linearised PDE backwards once to
get the whole gradient for the price of roughly one extra solve. It is the foundation of
industrial aerodynamic design, and it is mathematically the same object as
backpropagation — the reverse mode of automatic differentiation, applied to a PDE instead
of a network.

What this repo does is the surrogate analogue: replace the PDE by the network, and the
adjoint solve by a backward pass costing ~4 ms.

### 10.2 Making the whole chain differentiable

$$\mathbf c \longrightarrow r_0(\mathbf c) \longrightarrow \varphi_{\mathbf c} \longrightarrow X(\mathbf c) \longrightarrow N_\theta \longrightarrow \hat C_d.$$

Each arrow must be differentiable. The first three are analytic (§1.3 of the README);
the fourth is autodiff. One link fails: the **occupancy channel**
$\mathbf 1_{\{\varphi<0\}}$ is a step function of $\varphi$, so its derivative is zero
almost everywhere, and backpropagation returns nothing through it.

The fix is to replace the indicator by a smoothed Heaviside at optimisation time,

$$\mathbf 1_{\{\varphi<0\}} \longrightarrow \sigma(-\varphi/\epsilon), \qquad \frac{\partial}{\partial\varphi}\sigma(-\varphi/\epsilon) = -\frac1\epsilon\,\sigma(1-\sigma) \xrightarrow[\epsilon\to0]{} -\delta(\varphi),$$

with $\epsilon = 0.7$ cell. Numerically the two images differ only within ~1.5 cells of
the boundary (checked in `tests/test_pipeline.py`), so the network — trained on the hard
mask — is not being fed something it has not seen. But the derivative is now a **smeared
surface delta concentrated on $\partial\Omega_b$**, which is exactly the right object:
composed with $\partial\varphi/\partial\mathbf c$, which on the boundary is the normal
perturbation field, it is a discrete **shape derivative**.

This is worth stating in the classical language. The Hadamard structure theorem says the
derivative of a domain functional under a perturbation field $V$ depends only on the
normal component $V\cdot n$ on the boundary:

$$\mathrm dJ(\Omega)[V] = \int_{\partial\Omega} g\,(V\cdot n)\,\mathrm ds$$

for some scalar $g$ determined by the state and adjoint solutions. The smoothed indicator
is what makes the discrete pipeline reproduce that structure instead of returning zero.

### 10.3 Constraints

Three kinds appear, handled three different ways — the choice matters more than it looks:

- **the area constraint** $A(r_0, \mathbf c) = A_0$ is **eliminated in closed form**.
  Since $A = \pi r_0^2(1 + \frac12\lVert\mathbf c\rVert^2)$, solving for $r_0$ turns a
  7-parameter constrained problem into a 6-parameter unconstrained one. No penalty weight
  to tune, no infeasible iterate, exact to machine precision at every step. Always prefer
  this when the constraint is invertible;
- **the box constraint** $\lVert\mathbf c\rVert_\infty \le 0.16$ is handled by
  **projection**: take the Adam step, then clip. This is projected gradient descent, and
  it is exact for a box. Its purpose is not physical — it keeps the optimiser inside the
  envelope the surrogate was trained on;
- what is *not* used is a **penalty** $J + \mu\,g(\mathbf c)^2$, which is the default
  reflex and the worst of the three: it introduces a weight to tune, satisfies the
  constraint only approximately, and stiffens the problem as $\mu$ grows.

Optimisation is 16 random starts in parallel (non-convex problem; local minima are real)
for 250 iterations of projected Adam. 4 000 surrogate evaluations, 17 s.

### 10.4 The answer, and the bound

The optimum is $a_2 = 0.160$ with every other coefficient within 0.02 of zero, i.e.
$r(\theta) \approx r_0(1 + 0.16\cos2\theta)$: pure elongation, which is the textbook way
to reduce pressure drag at constant area. Physically expected, which is the reassuring
part.

The bound is **active**, and this must be reported: the surrogate's unconstrained optimum
lies outside the training box, so what stops the optimiser is the box, not the physics.
Widening it means widening the training distribution first — otherwise §11 gets worse.

---

## 11. Optimising against your own error

### 11.1 The winner's curse

Write the surrogate as truth plus error, $\hat J(d) = J(d) + e(d)$, and suppose $e$ is
*unbiased* over the design distribution, $\mathbb E[e] = 0$. Now minimise $\hat J$. The
minimiser is attracted to designs where $J$ is small **and** to designs where $e$ happens
to be negative, and it cannot tell the two apart. Hence

$$\mathbb E\big[e(d^\star)\ \big|\ d^\star = \arg\min\hat J\big] < \mathbb E[e(d)] = 0.$$

This is a selection effect, not a modelling failure — the same phenomenon as the winner's
curse in auctions, regression to the mean after selecting on a noisy score, and the
optimism of cross-validated model selection. It is worse when the error is large, when the
design space is big, and when the objective is flat near its optimum (many designs of
nearly equal $J$, so the selection is driven almost entirely by $e$).

Two things it does **not** say:

- it does not say the design is bad. The bias is in the *predicted value*. Here the
  solver confirms a genuine 18.6 % drag reduction;
- it does not go away with an unbiased model. Unbiasedness is an average over the design
  distribution, and $\arg\min\hat J$ is precisely not a typical draw from it.

Measured here: the surrogate predicted $C_d = 1.598$ for its own optimum; the solver says
1.436. **11.3 % optimistic.** Which is why the design is re-solved rather than trusted —
the one non-negotiable step in surrogate-based optimisation.

### 11.2 Active learning

The optimiser has just told you where it is trusting the surrogate, which is the cheapest
possible acquisition criterion. Take the 32 designs it proposed, label them with the
solver, fold them into the training set, warm-start from $\theta^\star$, fine-tune.

Result: the gap drops **11.3 % $\to$ 6.7 %** and the optimum barely moves (1.436 $\to$
1.453 in true $C_d$). The correct summary is that active learning bought **reliability,
not a better design** — an important distinction, and one that a less careful writeup
would blur into "active learning improved the optimisation by 40 %".

The principled version of this loop is **Bayesian optimisation** (EGO; Jones, Schonlau &
Welch, 1998): keep a probabilistic surrogate with error bars, and sample the next design
where the *expected improvement*

$$\mathrm{EI}(d) = \mathbb E\big[\max(J_{\text{best}} - \hat J(d),\ 0)\big]$$

is largest, which balances exploiting a low prediction against exploring a high variance.
A deterministic network gives no $\sigma(d)$, so the loop here is the crude version:
sample where the optimiser went. Ensembles, MC dropout or a last-layer Laplace
approximation would supply the missing variance and are the natural extension.

---

## 12. Reading the tables honestly

Some caveats that apply to every number in this repo:

- **no error bars on the model comparison.** Each architecture is trained once. The
  differences between the U-Net variants (1.03 % vs 1.15 %) are of the same order as
  seed-to-seed variability; the differences between families (1.0 % vs 3.8 % vs 7.1 %)
  are not. Read the table at the resolution it supports;
- **equal budget, not equal tuning.** 40 epochs each, one loss, one optimiser. A
  hyperparameter search would move every number, probably not the ranking;
- **the surrogates are fitted to the solver, not to reality.** The LBM code is verified
  to within 2 % of published cylinder drag on a refined grid and sits 4–6 % low on the
  production grid (§2.5 of the README). Every accuracy claim is relative to *that*;
- **$N = 128$ test samples** means the MAPE has a standard error of roughly
  $\text{std}/\sqrt{128}$; worst-case columns are single order statistics and are noisier
  still;
- **the OOD split is one particular direction of extrapolation** (higher $Re$, larger
  deformations). It is evidence about extrapolation, not a general characterisation of it.

---

## 13. Glossary

| term | meaning here |
|---|---|
| **surrogate / emulator / metamodel** | cheap approximation of an expensive solver |
| **ROM** | reduced-order model; here POD + regression |
| **POD / PCA / KL basis** | the same orthogonal basis, from the SVD of the snapshot matrix |
| **snapshot** | one solver output, flattened to a vector |
| **mode** | one basis vector $\psi_j$ |
| **Eckart–Young** | the truncated SVD is the optimal low-rank approximation |
| **Kolmogorov $n$-width** | best possible error of *any* $n$-dimensional linear subspace |
| **ridge / kernel ridge / kriging** | linear regression with an $L^2$ penalty; its kernelised version; the same as a GP posterior mean |
| **RKHS** | the function space implied by a kernel |
| **MLP** | dense feed-forward network |
| **CNN** | network built from convolutions = learned stencils |
| **channel** | one feature map in a CNN layer |
| **receptive field** | input cells that can influence one output cell |
| **U-Net** | encoder–decoder CNN with skip connections |
| **skip connection** | direct link from encoder to decoder at equal resolution |
| **GELU / GroupNorm** | smooth nonlinearity; per-group activation normalisation |
| **epoch / batch** | one pass over the data; one gradient step's worth of it |
| **Adam** | SGD variant with per-coordinate step scaling |
| **backpropagation** | reverse-mode automatic differentiation |
| **neural operator** | model of a map between function spaces, discretisation-independent in principle |
| **FNO** | neural operator that learns a multiplier in truncated Fourier space |
| **Gibbs phenomenon** | non-vanishing overshoot when truncating a Fourier series at a jump |
| **PINN** | network trained on a PDE residual; here only the soft-constraint idea is borrowed |
| **level set** | scalar field whose zero set is the boundary |
| **adjoint method** | one backward solve giving a full design gradient |
| **shape derivative** | derivative of a functional w.r.t. a boundary normal perturbation |
| **projected gradient** | gradient step followed by projection onto the feasible set |
| **winner's curse** | selection bias making the optimum's prediction optimistic |
| **active learning** | choosing the next samples to label using the current model |
| **Bayesian optimisation / EGO / EI** | the principled version of that loop, using a probabilistic surrogate |
| **in-distribution / OOD** | test data drawn like training data; test data deliberately outside it |

---

## 14. References

**Reduced-order modelling**
- Lumley (1967), *The structure of inhomogeneous turbulent flows* — POD in fluids.
- Sirovich (1987), *Turbulence and the dynamics of coherent structures* — the snapshot method.
- Eckart & Young (1936); Mirsky (1960) — optimality of the truncated SVD.
- Benner, Gugercin & Willcox (2015), *A survey of projection-based model reduction methods*, SIAM Review.
- Ohlberger & Rave (2016), *Reduced basis methods: success, limitations and future challenges* — the $n$-width obstruction for transport-dominated problems.

**Kernel methods and design of experiments**
- Kimeldorf & Wahba (1971) — representer theorem.
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, ch. 2.
- Jones, Schonlau & Welch (1998), *Efficient global optimization of expensive black-box functions* — EGO / expected improvement.

**Networks and operator learning**
- Ronneberger, Fischer & Brox (2015), *U-Net: convolutional networks for biomedical image segmentation*.
- Kingma & Ba (2015), *Adam*; Wu & He (2018), *Group normalization*; Hendrycks & Gimpel (2016), *GELU*.
- Guo, Li & Iorio (2016), *Convolutional neural networks for steady flow approximation* — the original CNN flow surrogate.
- Raissi, Perdikaris & Karniadakis (2019), *Physics-informed neural networks*, JCP.
- Li et al. (2021), *Fourier Neural Operator for parametric partial differential equations*, ICLR.
- Lu, Jin, Pang, Zhang & Karniadakis (2021), *Learning nonlinear operators via DeepONet*, Nature Machine Intelligence.
- Li et al. (2023), *Fourier neural operator with learned deformations for PDEs on general geometries* (Geo-FNO) — the direct answer to §8.3.
- Kovachki et al. (2023), *Neural operator: learning maps between function spaces*, JMLR — the framework and its discretisation-invariance claims.

**Shape optimisation**
- Pironneau (1974), *On optimum design in fluid mechanics*, JFM.
- Jameson (1988), *Aerodynamic design via control theory* — the adjoint method in aerodynamics.
- Allaire, Jouve & Toader (2004), *Structural optimization using sensitivity analysis and a level-set method*, JCP.
- Delfour & Zolésio (2011), *Shapes and Geometries* — the Hadamard structure theorem.

**Lattice Boltzmann** (for the solver itself)
- Krüger et al. (2017), *The Lattice Boltzmann Method: Principles and Practice*.
- Chen & Doolen (1998), *Lattice Boltzmann method for fluid flows*, Annu. Rev. Fluid Mech.
