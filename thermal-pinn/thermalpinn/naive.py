"""Why every method after this one has to regularise, and what none of them can recover.

Two facts bound the whole repo, and both are established here before any inverse method
is written -- otherwise there is no way to tell a regularisation that works from one that
merely looks tidy.

*Marching towards the surface is explosive.*  Reconstructing the field between the sensor
and the friction surface is a Cauchy problem for the heat equation: the data sit on a line
inside the domain and the solution is wanted to one side of it.  `space_march` does it in
the most direct way possible -- rearrange the heat equation into a recursion in x and step
towards x = 0.  On noise-free data the march is essentially exact, which matters: the
divergence that follows is a property of the *problem*, not a bug in the marcher.

The size of the divergence, though, turns entirely on how finely the march is asked to
resolve time.  With the thermocouple's real 0.5 K it is 6 % high on the surface peak at
the 100 Hz this repo samples at, and 10^5 K at 1 kHz -- so at 100 Hz the sampling rate is
already acting as an accidental regulariser, and sampling faster, which is strictly more
information, makes the answer worse.  More data, worse answer: that is what ill posed
means, stated in kelvin, and it is why picking a sampling rate is not a substitute for
having a regularisation parameter.

*Part of the signal is gone before any method runs.*  A harmonic component of the surface
temperature at frequency f reaches depth d attenuated by exp(-d sqrt(pi f / alpha)).
Inverting means multiplying by the reciprocal, so demanding an amplification no larger
than 1/eta caps the recoverable band at

    f_c = alpha ln(1/eta)^2 / (pi d^2)

which is about 7 Hz at 2 mm in cast iron and below 1 Hz at 6 mm.  The 8 Hz ABS ripple in
the true flux is therefore already destroyed in the measurement.  `transfer_function` and
`cutoff_frequency` are the closed forms; `measured_attenuation` measures the same thing on
the actual solver, so the bound the repo scores against is verified rather than asserted.

Everything here is for the linear law: the naive march has no defensible nonlinear
extension, and the attenuation bound is a statement about the linearised operator, which
is the strongest form the argument needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .material import LAW_A, Material
from .scenario import Scenario
from .solver import TRUTH_GRID, Grid, Solution, forward

#: Amplification budget the recoverable band is defined by: a component attenuated to
#: less than this fraction of its surface amplitude cannot be lifted back without
#: multiplying the measurement noise by more than 1/eta.
ETA = 0.1


# ======================================================================================
# The bandwidth the depth of the sensor imposes
# ======================================================================================


def transfer_function(f: np.ndarray | float, depth: float, alpha: float) -> np.ndarray:
    """Complex ratio T(depth) / T(surface) for a harmonic at frequency `f`.

    A harmonic at angular frequency w propagates into a half space as exp(-sigma x) with
    sigma = sqrt(i w / alpha) = (1 + i) sqrt(w / 2 alpha), so amplitude and phase both
    decay by beta = d sqrt(pi f / alpha).

    The surface boundary condition fixes the amplitude at x = 0 but drops out of the
    ratio, which is why this is a property of the material and the depth alone: no choice
    of h, emissivity or flux history can widen the band.  The half-space assumption is the
    one approximation, and it is good exactly where it matters -- at the cutoff the
    thermal wave penetrates well under a millimetre, far short of the mid-plane.
    """
    beta = np.sqrt(np.pi * np.abs(np.asarray(f, dtype=float)) / alpha) * depth
    return np.exp(-beta) * np.exp(-1j * beta)


def attenuation(f: np.ndarray | float, depth: float, alpha: float) -> np.ndarray:
    """Amplitude damping between the surface and `depth`, in (0, 1]."""
    return np.abs(transfer_function(f, depth, alpha))


def cutoff_frequency(depth: float, alpha: float, eta: float = ETA) -> float:
    """Top of the recoverable band, Hz: where the damping reaches `eta`.

    Metrics on q(t) are computed inside this band for every method, truth included.
    Scoring anyone on what the measurement destroyed would rank noise amplification.
    """
    return alpha * np.log(1.0 / eta) ** 2 / (np.pi * depth**2)


def _crossing(freqs: np.ndarray, values: np.ndarray, level: float) -> float:
    """Frequency at which `values` falls through `level`, interpolated log-log."""
    below = np.nonzero(np.asarray(values) < level)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")  # the sweep does not bracket the crossing
    j = below[0]
    lf, lv = np.log(freqs[j - 1 : j + 1]), np.log(np.asarray(values)[j - 1 : j + 1])
    return float(np.exp(lf[0] + (np.log(level) - lv[0]) * (lf[1] - lf[0]) / (lv[1] - lv[0])))


#: Tone numbers of the projection window used by `measured_attenuation`.  Integer
#: multiples of 1/window, so the tones are exactly orthogonal over it, and clustered
#: around the cutoffs of 1, 2 and 4 mm (26.5, 6.6 and 1.7 Hz) so each is bracketed
#: tightly enough that the interpolated crossing is not the thing being measured.
TONES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)


def measured_attenuation(
    *,
    material: Material = LAW_A,
    grid: Grid = TRUTH_GRID,
    depths: tuple[float, ...] = (1.0e-3, 2.0e-3, 4.0e-3),
    settle: float = 4.0,
    window: float = 4.0,
    tones: tuple[int, ...] = TONES,
    q_bias: float = 1.0e6,
    q_tone: float = 1.0e5,
    eta: float = ETA,
) -> dict:
    """Measure the depth attenuation on the real solver, on the real geometry.

    One forward run carries every frequency at once: the applied flux is a bias plus a
    sum of tones at m / `window` Hz, and a second run with the bias alone is subtracted.
    Law A is exactly linear, so superposition makes that difference the pure response to
    the tones, and the tones separate exactly because they are orthogonal over the window.
    Fifteen frequencies for the price of one time integration, with no leakage.

    The instantaneous flux dips negative when the tones align.  That is meaningless as a
    braking history and irrelevant here: this run measures a linear operator, not a
    physical event.

    `settle` discards the switch-on transient; a constant and a linear trend are fitted
    alongside the tones to absorb what remains of the slab's slow convective drift, whose
    time constant (rho c L / h, about 280 s) is far longer than the window.

    Returns the measured amplitude ratios and phase lags against the closed form, and the
    cutoff frequency read off each -- the comparison milestone M2 turns on.
    """
    scenario = Scenario(t_end=settle + window)
    probes = (0.0, *depths)
    freqs = np.array([m / window for m in tones], dtype=float)

    def q_tones(t: float) -> float:
        return q_bias + q_tone * float(np.sum(np.sin(2.0 * np.pi * freqs * t)))

    kw = dict(grid=grid, probes=probes, t_end=scenario.t_end, n_snapshots=2)
    tonal = forward(scenario, material, q_tones, **kw)
    base = forward(scenario, material, q_bias, **kw)
    response = tonal.T_probe - base.T_probe

    inside = tonal.t >= settle
    t_win = tonal.t[inside]
    design = [np.ones_like(t_win), t_win - t_win.mean()]
    for f in freqs:
        design += [np.sin(2.0 * np.pi * f * t_win), np.cos(2.0 * np.pi * f * t_win)]
    design = np.column_stack(design)

    amp = np.empty((len(probes), freqs.size), dtype=complex)
    for j in range(len(probes)):
        coef = np.linalg.lstsq(design, response[inside, j], rcond=None)[0]
        amp[j] = coef[2::2] + 1j * coef[3::2]

    rows, per_depth = [], []
    for j, d in enumerate(depths, start=1):
        ratio = np.abs(amp[j]) / np.abs(amp[0])
        # Phase lag of the deep probe behind the surface, unwrapped into a growing lag:
        # the closed form predicts beta radians, which passes 2 pi well inside the sweep.
        lag = np.unwrap(np.angle(amp[0]) - np.angle(amp[j]))
        exact = attenuation(freqs, d, material.alpha0)
        beta = np.sqrt(np.pi * freqs / material.alpha0) * d
        for i, f in enumerate(freqs):
            rows.append(
                {
                    "depth": d,
                    "f": float(f),
                    "amplitude_surface_K": float(abs(amp[0, i])),
                    "measured": float(ratio[i]),
                    "analytic": float(exact[i]),
                    "phase_lag_measured": float(lag[i]),
                    "phase_lag_analytic": float(beta[i]),
                }
            )
        f_meas = _crossing(freqs, ratio, eta)
        f_exact = cutoff_frequency(d, material.alpha0, eta)
        per_depth.append(
            {
                "depth": d,
                "f_c_measured": f_meas,
                "f_c_analytic": f_exact,
                "ratio": f_meas / f_exact,
                # Where the half-space form is allowed to be wrong: the wave has to die
                # before it feels the adiabatic mid-plane.  One penetration depth is
                # sqrt(alpha / pi f_c); at 4 mm that leaves barely a millimetre of slab
                # behind the probe, the reflection lifts the measured amplitude, and the
                # measured cutoff comes out above the closed form rather than below.
                "penetration_at_f_c_m": float(np.sqrt(material.alpha0 / (np.pi * f_exact))),
                "slab_behind_probe_m": scenario.L - d,
            }
        )

    return {
        "law": material.name,
        "grid": {"nx": grid.nx, "dt": grid.dt},
        "eta": eta,
        "settle": settle,
        "window": window,
        "frequencies": freqs.tolist(),
        "q_bias": q_bias,
        "q_tone": q_tone,
        "rows": rows,
        "per_depth": per_depth,
    }


# ======================================================================================
# Marching in space, with nothing holding it back
# ======================================================================================


@dataclass
class Marched:
    """What a space march returns.  `T[:, 0]` is the reconstructed surface trace."""

    t: np.ndarray  # (nt,)
    x: np.ndarray  # (n_march + 1,) from 0 to x_s
    T: np.ndarray  # (nt, n_march + 1)
    q_applied: np.ndarray  # (nt,) flux implied at the surface, W/m^2

    @property
    def T_surface(self) -> np.ndarray:
        return self.T[:, 0]


def _back_region(t: np.ndarray, T_measured: np.ndarray, scenario: Scenario, material: Material, dx: float) -> np.ndarray:
    """Solve the *well posed* half, [x_s, L], and return the trace one dx behind the sensor.

    Behind the sensor the problem is an ordinary initial-boundary-value problem -- the
    measurement is a Dirichlet condition, the mid-plane is adiabatic -- so it is solved
    the ordinary way: Crank-Nicolson, second order, mirrored ghost node at the mid-plane.
    Noise is *smoothed* going this direction, which is exactly the point: the difficulty
    is entirely in the other one.  The march needs two adjacent layers to start, and this
    supplies the second.
    """
    dt = float(t[1] - t[0])
    n = int(round((scenario.L - scenario.x_s) / dx))
    if n < 2:
        raise ValueError(f"back region needs at least two cells, got {n}")
    r = material.alpha0 * dt / dx**2

    # Unknowns are nodes 1..n, measured from x_s; node n sits on the mid-plane.
    ab = np.zeros((3, n))
    ab[0, 1:] = -0.5 * r  # upper
    ab[1, :] = 1.0 + r  # main
    ab[2, :-1] = -0.5 * r  # lower
    ab[2, n - 2] = -r  # mirrored ghost node: the interior neighbour counts twice

    # Node 0 carries the measurement and never enters the solve; it appears only as the
    # known term in the j = 1 row, at both time levels.
    T = np.full(n + 1, float(scenario.T0))
    out = np.empty(t.size)
    out[0] = T[1]
    for k in range(1, t.size):
        rhs = (1.0 - r) * T[1:]
        rhs[:-1] += 0.5 * r * T[2:]
        rhs[1:] += 0.5 * r * T[1:-1]
        rhs[-1] += 0.5 * r * T[n - 1]  # the mirror again, on the old level
        rhs[0] += 0.5 * r * (T_measured[k - 1] + T_measured[k])
        T[1:] = solve_banded((1, 1), ab, rhs)
        out[k] = T[1]
    return out


def space_march(
    t: np.ndarray,
    T_measured: np.ndarray,
    scenario: Scenario,
    material: Material = LAW_A,
    *,
    n_march: int = 20,
) -> Marched:
    """March the heat equation from the sensor to the friction surface, unregularised.

    Read as a recursion in x rather than in t, the second-order stencil for
    T_xx = T_t / alpha gives

        T_{j-1} = 2 T_j - T_{j+1} + (dx^2 / alpha) dT_j/dt

    so two adjacent layers determine the next one closer to the surface.  Every step
    differentiates the layer it was handed, and differentiation multiplies a component at
    frequency f by 2 pi f: after marching a distance d the amplification is
    exp(d sqrt(pi f / alpha)), which is `transfer_function` run backwards.  Nothing is
    wrong with the algebra -- that factor is the exact inverse of a real physical
    attenuation.  It is simply unbounded in f, and measurement noise is not band limited.

    `t` must be uniformly spaced, and its spacing is what sets the damage: the march can
    only amplify what the sampling rate lets it see.  The linear law only; there is no
    honest nonlinear version of this and it is not needed to make the point.
    """
    t = np.asarray(t, dtype=float)
    if not material.is_linear:
        raise ValueError(f"space marching assumes constant properties; law {material.name} is nonlinear")
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, rtol=1e-9, atol=0.0):
        raise ValueError("space marching needs a uniform time grid")

    dx = scenario.x_s / n_march
    T = np.empty((t.size, n_march + 1))
    T[:, n_march] = T_measured
    behind = _back_region(t, T_measured, scenario, material, dx)

    ahead = behind  # the layer one dx *deeper* than the one being stepped from
    for j in range(n_march, 0, -1):
        T[:, j - 1] = 2.0 * T[:, j] - ahead + dx**2 / material.alpha0 * np.gradient(T[:, j], dt)
        ahead = T[:, j]

    # Second-order one-sided gradient at the surface, then the boundary condition read
    # backwards for the applied flux.  Radiation is off on the linear law.
    dTdx = (-3.0 * T[:, 0] + 4.0 * T[:, 1] - T[:, 2]) / (2.0 * dx)
    q = -material.k0 * dTdx + scenario.h * (T[:, 0] - scenario.T_inf)
    return Marched(t=t, x=np.arange(n_march + 1) * dx, T=T, q_applied=q)


def march_divergence(
    scenario: Scenario | None = None,
    material: Material = LAW_A,
    *,
    grid: Grid = TRUTH_GRID,
    rates: tuple[float, ...] = (100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0),
    n_marches: tuple[int, ...] = (10, 20, 40),
    sigma: float | None = None,
    seed: int = 0,
    truth: Solution | None = None,
) -> dict:
    """Run the march on the real braking event, clean and noisy, at several sample rates.

    The sweep is the argument.  On noise-free data the march reconstructs the surface peak
    to a few hundredths of a percent, so the implementation is sound.  With the same 0.5 K
    per sample the thermocouple actually has, it is off by tens of kelvin at 100 Hz and by
    more than 10^5 K at 1 kHz: sampling faster hands the march more of the band where its
    amplification is enormous, and there is no rate at which the two agree.  A sampling
    rate is not a regularisation parameter, and picking one is not a substitute for having
    one.

    The last rate is the limit of the argument rather than part of it.  At 10 kHz the clean
    column breaks too, because the data are then the truth solver's own output at its own
    step and its O(dt^2) discretisation error is all the march needs.  Nothing is exact
    enough to be marched at that resolution.

    `n_marches` varies the spatial step of the march itself, to show that the divergence
    is not an artefact of one discretisation choice: the amplification is set by the
    distance covered, not by how finely it is cut up.
    """
    scenario = Scenario() if scenario is None else scenario
    sigma = scenario.sigma_noise if sigma is None else sigma
    truth = forward(scenario, material, scenario.q_true, grid=grid) if truth is None else truth
    T_surface_max = float(truth.T_surface.max())
    rise = T_surface_max - scenario.T0

    rows = []
    for f_s in rates:
        dt = 1.0 / f_s
        t = np.arange(int(round(scenario.t_end * f_s)) + 1, dtype=float) * dt
        surface = np.interp(t, truth.t, truth.T_surface)
        clean = np.interp(t, truth.t, truth.at(scenario.x_s))
        noisy = clean + np.random.default_rng(seed).normal(0.0, sigma, t.shape)
        for n_march in n_marches:
            row: dict = {"f_sample": f_s, "n_march": n_march, "n_samples": int(t.size), "sigma_K": sigma}
            for tag, data in (("clean", clean), ("noisy", noisy)):
                marched = space_march(t, data, scenario, material, n_march=n_march)
                peak = float(marched.T_surface.max())
                row[tag] = {
                    "excursion_K": float(np.max(np.abs(marched.T_surface - scenario.T0))),
                    "max_abs_error_K": float(np.max(np.abs(marched.T_surface - surface))),
                    "T_surface_max_error_pct": float(100.0 * (peak - T_surface_max) / rise),
                }
            rows.append(row)

    return {
        "law": material.name,
        "grid": {"nx": grid.nx, "dt": grid.dt},
        "n_marches": list(n_marches),
        "x_s": scenario.x_s,
        "sigma_K": sigma,
        "seed": seed,
        "T_surface_max": T_surface_max,
        "T_sensor_max": float(truth.at(scenario.x_s).max()),
        "rows": rows,
    }


def march_traces(
    scenario: Scenario | None = None,
    material: Material = LAW_A,
    *,
    grid: Grid = TRUTH_GRID,
    rates: tuple[float, ...] = (100.0, 1000.0),
    n_march: int = 20,
    seed: int = 0,
    truth: Solution | None = None,
) -> dict:
    """The same marches again, keeping the traces themselves for `figs/illposed.png`.

    Noise is drawn from `seed` per rate exactly as in `march_divergence`, so the traces
    drawn are the ones the table scored rather than a second, luckier draw.
    """
    scenario = Scenario() if scenario is None else scenario
    truth = forward(scenario, material, scenario.q_true, grid=grid) if truth is None else truth
    ts = scenario.sensor_times()
    clean = np.interp(ts, truth.t, truth.at(scenario.x_s))
    noisy = clean + np.random.default_rng(seed).normal(0.0, scenario.sigma_noise, ts.shape)

    marches = {}
    for f_s in rates:
        t = np.arange(int(round(scenario.t_end * f_s)) + 1, dtype=float) / f_s
        data = np.interp(t, truth.t, truth.at(scenario.x_s))
        data = data + np.random.default_rng(seed).normal(0.0, scenario.sigma_noise, t.shape)
        marches[f_s] = space_march(t, data, scenario, material, n_march=n_march)

    return {
        "t_truth": truth.t,
        "T_surface": truth.T_surface,
        "T_sensor": truth.at(scenario.x_s),
        "t_sensor": ts,
        "sensor_clean": clean,
        "sensor_noisy": noisy,
        "marches": marches,
    }
