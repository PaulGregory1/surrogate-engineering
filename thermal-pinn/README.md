# thermal-pinn

**Under construction — milestones M0 to M2 are done.  No method has been compared to
another yet; this file becomes the headline at M6, and every number in it will come from
`scripts/03_report.py`, not from a keyboard.**

## The question

A brake disc is stopped from 160 km/h in four seconds.  The friction surface reaches
several hundred degrees, and no sensor can be put there.  The only measurement is a
thermocouple buried 2 mm under the surface, sampled at 100 Hz, with ±0.5 K of noise.

What was the temperature of the surface?

This is not the question the other two repos in this monorepo ask.  `flow-surrogate` and
`stress-gnn` both ask *can a learned model replace an expensive solver*, and both answer
"yes, inside a narrow envelope".  Here the solver is not being replaced:

> The network is asked to solve a problem the classical solver cannot pose.

Recovering a surface flux from an interior measurement is a Cauchy problem for the heat
equation — ill posed in the sense of Hadamard, and the classical theory for it (Beck,
Tikhonov, Alifanov's adjoint) is excellent.  The thesis this repo exists to test, by
measurement rather than by assertion, is narrower:

> The regularised classical method is as good as anything **as long as it is handed the
> right model**.  A PINN matches it there, and pulls ahead once one of the unknowns is a
> **function** rather than a handful of scalars — because it needs neither an a-priori
> parametrisation nor a fresh adjoint derivation.

So the repo is built as three acts, which differ only in the constitutive law:

| act | law | unknown | what it measures |
|---|---|---|---|
| 1 | constant `k`, `c` | `q(t)` | the fair fight — superposition holds, so Beck and Tikhonov are exactly applicable |
| 2 | smooth `k(T)`, `c(T)`, radiation | `q(t)` | what it costs to *ignore* the nonlinearity, and what it costs to have to *know* it |
| 3 | law 2 plus a −12 % knee in `k` at 450 °C | `q(t)` **and** `k(T)` | the classical route has to guess the shape of `k(T)`; the PINN is asked to discover it |

Two things are settled before any method runs, because they bound what *any* method can
do: marching the measurement back to the surface diverges, and a component at frequency
`f` measured at depth `d` is attenuated by `exp(-d sqrt(w/2a))`, which puts the cutoff at
2 mm in cast iron near 7 Hz.  The 8 Hz ABS ripple in the true flux is therefore already
destroyed before any processing.  No method is scored on it; it is reported separately as
information lost.  Both are measured in `figs/illposed.png`, not assumed.

## State

| milestone | status |
|---|---|
| M0 skeleton | done |
| M1 truth solver + verification | done — `results/solver_validation.json` |
| M2 the ill-posedness, demonstrated | done — `results/illposedness.json`, `figs/illposed.png` |
| M3 classical baselines (Beck, Tikhonov, Alifanov) | done — `results/baseline_validation.json` |
| M4 the PINN, act 1 | done — `results/act1.json`, `figs/act1_linear.png` |
| M5 acts 2 and 3 | not started |
| M6 studies, tables, writing | not started |

## What is here today

```
thermalpinn/material.py   the three constitutive laws, their derivatives, Kirchhoff
thermalpinn/scenario.py   geometry, the braking event, the true flux, the sensor
thermalpinn/solver.py     Crank-Nicolson finite volume + Newton, the truth generator
thermalpinn/verify.py     manufactured solutions, the erfc reference, the energy balance
thermalpinn/naive.py      space marching, the depth attenuation, the recoverable band
thermalpinn/classical.py  the inverse model + its adjoint; Beck, Tikhonov, Alifanov
thermalpinn/metrics.py    the recoverable-band scores, and what is deliberately not scored
thermalpinn/pinn.py       the reduced variables, the two networks, the residual, the training
thermalpinn/plots.py      figures
scripts/01_validate_solver.py
scripts/02_run_methods.py
scripts/03_act1.py
scripts/06_explainer_figures.py
```

The solver is verified three independent ways, and the numbers are regenerated rather
than quoted:

- **order of accuracy** — manufactured solutions on all three laws, driving the full
  nonlinear machinery including radiation and the flux boundary condition.  Observed
  order 2.00 in space and 1.98–2.07 in time.
- **physics** — the closed-form constant-flux half-space solution, matched to better than
  0.02 % while the thermal wave is still far from the mid-plane.
- **conservation** — stored enthalpy against boundary influx over the whole 20 s event,
  closed to 4e-15 relative on the linear law.

The truth is generated at `nx = 801, dt = 1e-4`; every inverse method will be given
`nx = 201, dt = 1e-3`.  Sharing a discretisation with the truth is the inverse crime, and
results obtained that way measure the grid rather than the method.

## What M2 establishes, and one thing it contradicts

Both numbers below are in `results/illposedness.json`; `figs/illposed.png` is the same
thing in three panels.

**Marching in space diverges — but not at the rate this repo samples at.**  The march is
run twice at every sampling rate, on noise-free data and on the same data plus the
thermocouple's real 0.5 K.  Error on the surface peak, at 20 marching steps:

| sampling rate | clean data | + 0.5 K of noise |
|---|---|---|
| 100 Hz (this repo's sensor) | −0.008 % | **+6.0 %** |
| 200 Hz | −0.002 % | +46 % |
| 1 kHz | +0.001 % | +16 400 % (1.1 × 10⁵ K) |
| 5 kHz | +0.004 % | 5 × 10¹¹ K |

The clean column is the control, and it matters: the marcher is *correct*, so what the
noisy column shows is the problem and not the code.  What it also shows is that the plan's
expectation — naive inversion simply diverges — is wrong at 100 Hz, where the march is
merely 6 % high on the peak rather than divergent.  The reason is worth more than the
original claim: a 100 Hz sample rate band-limits the data below the frequencies where the
amplification is catastrophic, so **the sampling rate is acting as an accidental
regulariser**.  Sample the same thermocouple faster, which is strictly more information,
and the same code returns 10⁵ K.  There is no rate at which the two columns agree, and
choosing a rate is not the same as having a regularisation parameter.  The headline table
at M6 will therefore report the naive row at 100 Hz as a number, not as *diverges*.

**Part of the flux history was destroyed in the measurement.**  The depth attenuation is
measured on the solver itself — one forward run carrying 26 tones at once, a second with
the tones removed, and the tones separated by least squares over a window they are
orthogonal on — then compared with the closed form `exp(-d sqrt(pi f / alpha))`:

| sensor depth | measured cutoff | `alpha ln(1/eta)^2 / pi d^2` | ratio |
|---|---|---|---|
| 1 mm | 26.46 Hz | 26.50 Hz | 0.999 |
| 2 mm | 6.62 Hz | 6.62 Hz | 0.999 |
| 4 mm | 1.84 Hz | 1.66 Hz | 1.112 |

The 4 mm row is the closed form's own limit rather than a measurement error: the harmonic
at that cutoff penetrates 1.7 mm and only 1.0 mm of disc remains behind the probe, so the
adiabatic mid-plane reflects part of the wave back and the real disc attenuates *less* than
a half space would.  The criterion for which depths the closed form is allowed to be
scored on is declared in the script before the numbers are read, and 4 mm is excluded from
the verdict for that reason — it passes ±20 % anyway.

At 2 mm, then: 8 % of the 8 Hz ABS ripple reaches the sensor, and it sits outside the
recoverable band.  Every later metric on `q(t)` is computed below 6.6 Hz, for the truth and
the estimate alike.

## What M3 establishes: the baselines are real, and they are good

Numbers from `results/baseline_validation.json`.  All three methods estimate the same object
— a flux staircase on the 100 Hz measurement grid — from the same coarse model
(`nx = 201, dt = 1e-3`), which tracks the truth at the sensor to **0.017 K**: the floor no
inverse method can beat, thirty times below the measurement noise.

| check | result |
|---|---|
| inverse crime, zero noise (< 1 %) | Beck 0.094 %, Tikhonov 0.005 %, Alifanov 0.041 % |
| adjoint gradient vs central differences (< 1e-4) | law A **3e-12**, laws B and C **3e-8** |
| the three within a factor 2 on real noise | spread **1.24×** |

The gradient check is the one that cannot be waived, since every act 2 and act 3 result is a
descent of it.  It passes by four orders of magnitude because the adjoint is **discrete** —
it differentiates the scheme that is actually solved, transposed step by step, rather than
the PDE the scheme approximates.  On the linear law the model is exactly linear in the flux,
so the central difference is exact and only round-off separates them; on laws B and C the
limit is the finite difference's own truncation, not the adjoint.

On the showcase event with 0.5 K of noise, all three read a surface they cannot see to
**within 0.14 %** of its 600 K rise (Beck −0.016 %, Tikhonov −0.112 %, Alifanov −0.135 %),
at 3.2–4.0 % RMSE on the flux in band. **Act 1 is not going to be a fight the PINN wins** —
that was always the expectation, and the bar is now measured rather than assumed.

Three things had to be got right for that to be true, and each was found by measuring:

- **Beck was being handicapped by its flux step.** Estimating one flux value per measurement
  gives a dimensionless step `alpha dt / x_s^2` of 0.039, five times finer than the
  classical guidance, and Beck came out three times worse than Tikhonov — which would have
  failed the factor-2 check and made the comparison a straw man. Holding the flux over five
  measurements (0.196, mid-range) puts it level with Tikhonov. Beck's unregularised limit
  does not exist in practice: after 10 ms the sensor at 2 mm has barely heard of the flux
  (step response 1.6e-9 K per W/m²), so the recursion amplifies round-off by ~10⁹ per step
  and diverges to 10²⁵ % on noise-free data.
- **The band metric was contaminated by its own filter.** Band-limiting a *finite* record
  cannot suppress out-of-band content near its ends, and 40–60 % of every method's reported
  error was coming from 15 % of the samples. Scores are now taken on the interior, excluding
  one filter transient at each end, with the full-record number reported beside them. This
  also corrected a finding: the L-curve's flux looked twenty times worse than the
  discrepancy principle's, and most of that was the boundary.
- **The L-curve does not work on this problem.** There is no corner to find — the penalty
  axis spans five decades while the residual spans under two — so it settles several decades
  low and returns a flux 1.8× worse in band and far rougher. Reported rather than dropped:
  the honest reading is that the classical route needs a tuning parameter, and here the usual
  way of setting it *without* knowing the noise level does measurably worse than knowing σ.

## What M4 establishes: the PINN loses act 1, and by how much

Numbers from `results/act1.json`; the figure is `figs/act1_linear.png`. Law A, the showcase
stop, the same 0.5 K record all three baselines were given, five seeds, and a residual weight
chosen on a different scenario entirely.

| method | band RMSE on `q` | of the flux peak | error on `T_surf` max |
|---|---|---|---|
| Beck | 3.97 % | 0.97 % | **−0.016 %** |
| Tikhonov | 3.19 % | 0.78 % | −0.112 % |
| Alifanov | 3.26 % | 0.80 % | −0.135 % |
| PINN, 5 seeds | 9.18 % ± 1.07 | 2.25 % | **+0.18 % ± 0.08** |

**On the question the repo was built to ask, it is a draw.** The PINN reads a surface nobody
can instrument to 0.18 % of its 600 K rise — about 1.1 K on a peak of 640 °C — against 0.016
to 0.135 % for the three classical methods. The spread across five seeds, ±0.08 %, is smaller
than the spread between the baselines themselves.

**On the flux it is a loss, by a factor of about 2.5**, and the milestone's gate is failed
rather than argued with: the criterion was a factor 1.5 of the *best* baseline on the surface
peak, and the median seed is 13.5× Beck. Against the worst baseline it is 1.57×, which is
still outside. The plan said to write the gap down as measured if it came out this way, so:
the PINN does not win act 1, and act 1 is where the classical route is strongest.

**The number that says how much to trust the first row.** The PINN gives two answers for the
surface — the network's own `T(0, t)`, and what its flux produces when pushed through the same
finite-volume model the baselines are scored with. They disagree by 4.6 %. That gap is not
noise, it is the trained network failing to satisfy the physics it was scored on, and it means
the +0.18 % above is partly the network's smoothness rather than a correct inversion. A
converged PINN would have those two agree. This one does not, and reporting only the flattering
half would be the easiest dishonesty available in this repo.

Two further readings, both from the same file:

- **The regularisation is real and it has an optimum.** The residual weight was swept over
  four decades on the tuning scenario: 91.4 % in band at ×1, 32.6 % at ×10, 13.9 % at ×100,
  **8.6 % at ×1000**, then back up to 25.5 % at ×10⁴ where the sensor fit degrades to 1.58 K
  against 0.5 K of noise. That is the same U-shape Tikhonov's `lambda` has, found the same
  way, and it needs the truth to find — which is why it is found on a scenario that is
  neither the showcase nor any case the M6 table will average over.
- **The physics term is what does the inverting.** With `lambda_r = 0` the same network fits
  the thermocouple *better* than any other setting — 0.45 K, below the noise — and gets the
  surface peak wrong by 41 %, 16.7× worse in band. Fitting the sensor and inverting for the
  surface are different problems, and that row is the cleanest demonstration of it here.

## What it cost to get there

Three things had to be corrected before any of it meant anything, and all three were found by
measuring rather than by reading. Each is written up where it lives in the code, and each has
a test; none of them is a hyperparameter that happened to work.

- **The residual, unweighted, is not a well-defined objective on this problem.** Switch a
  flux on at t = 0 and the surface obeys the heat kernel, `T_t ~ t^(-1/2)`. Square that and
  average it over collocation points drawn uniformly in time and the integral is one of
  `1/t`: logarithmically divergent, so the mean squared residual has no limit and simply
  climbs with the number of points. Worse than the divergence sounds, a logarithm weights
  every *decade* equally, so a thousandth of the points carried as much of the objective as
  the last decade did. Weighting the residual by `sqrt(t)` cancels it exactly and cannot
  change which fields solve the equation.
- **Adaptive gradient-norm balancing of the loss weights diverges here.** The textbook scheme
  re-measures the weights as training proceeds; the better the residual is satisfied the
  smaller its gradient, the rule reads a small gradient as *weight this more*, and it feeds
  back. Left running it took the residual weight to four million times its initial value and
  arrived at the trivial solution — uniform `T0`, zero flux, exactly zero residual, and a
  sensor error of 517 K that by then weighed nothing. The balance is now measured once and
  frozen, and the weight is swept explicitly.
- **The obvious hard initial condition is the wrong one.** Writing `T = T0 + t N(x, t)`
  satisfies `T(x, 0) = T0` identically, which is why it is the standard choice — but it also
  asserts that the surface leaves `T0` *linearly* in time, and a switched-on flux makes it
  leave as `sqrt(t)`. For `t N` to produce that, `N` would have to diverge at the origin, and
  a tanh network cannot, so the flux it can express is pinned near zero at t = 0 and has to
  climb to meet the truth. `T = T0 + sqrt(t) N(x, t)` is the same statement as the residual's
  own weight, applied to the ansatz instead of to the norm, and the two cancel. Run as an
  ablation at the chosen weight, everything else identical: **8.6 % in band against 30.7 %**,
  and the disagreement between the PINN's two answers for the surface goes from 15.3 % to
  3.3 %. It is the single most expensive mistake in the file, it is invisible until the flux
  is plotted, and it is the standard choice.

The third is why the act-1 numbers above are the second set produced: it changes the answer,
so every training behind the first set was redone rather than patched.

There is a fourth, smaller, in the same spirit. The plan called for 20 000 Adam steps then
2 000 of L-BFGS. Measured, that is close to backwards: past about 4 000 steps Adam moved the
residual from 65 to 57, while the next 2 000 L-BFGS iterations took it from 57 to 1.0 at
1.2× the cost per iteration. The split is now 4 000 / 12 000. `float64` was tried and buys
about 5 % for twice the time, so `float32` stands as the plan assumed; Fourier bandwidth and
network width were swept and are not the lever.

## What is left

- **M5, the milestone that decides whether the repo has a result.** Act 2 measures what
  frozen properties cost; act 3 asks the PINN to discover `k(T)`'s knee against three
  parametric families the classical route has to guess between. `pinn.py` is written for it —
  the residual is in conservative form precisely so that `k_hat` can become a third network
  without anyone re-deriving a chain rule — but `k_psi` itself is not written yet.
- **The act-1 disagreement is a loose end worth pulling on before M5.** 4.6 % between the
  network's own surface and its flux pushed through the model is the honest bound on the
  first table above, and act 3's headline is a curve read off the same field. Whatever
  closes that gap is worth more to M5 than anything else on this list.
- **M6**: the studies (noise, depth, sensor count, PINN ablations, cost), `RESULTS.md`,
  `BACKGROUND.md`, the eight figures, and `run_all.py --force` regenerating everything.
- **Outside this directory, once M6 lands**: the root `README.md` three-column table,
  `KNOWLEDGE-GRAPH.md` (new node, and its baseline advanced), `CLAUDE.md` (the third `.venv`).

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python run_all.py --quick
./.venv/bin/python -m pytest tests -q
```

Always `./.venv/bin/python` from inside `thermal-pinn/`, never a shared interpreter — the
three repos in this monorepo have separate environments on purpose.
