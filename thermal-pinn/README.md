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
| M3 classical baselines (Beck, Tikhonov, Alifanov) | not started |
| M4 the PINN, act 1 | not started |
| M5 acts 2 and 3 | not started |
| M6 studies, tables, writing | not started |

## What is here today

```
thermalpinn/material.py   the three constitutive laws, their derivatives, Kirchhoff
thermalpinn/scenario.py   geometry, the braking event, the true flux, the sensor
thermalpinn/solver.py     Crank-Nicolson finite volume + Newton, the truth generator
thermalpinn/verify.py     manufactured solutions, the erfc reference, the energy balance
thermalpinn/naive.py      space marching, the depth attenuation, the recoverable band
thermalpinn/plots.py      figures
scripts/01_validate_solver.py
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

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python run_all.py --quick
./.venv/bin/python -m pytest tests -q
```

Always `./.venv/bin/python` from inside `thermal-pinn/`, never a shared interpreter — the
three repos in this monorepo have separate environments on purpose.
