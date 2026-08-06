"""The braking event: geometry, sensor, and the true surface heat flux.

Half a 10 mm grey-cast-iron disc, braked on both faces, so only x in [0, L] with
L = 5 mm is modelled: x = 0 is the friction surface, x = L the adiabatic mid-plane.
A thermocouple sits at `x_s` = 2 mm and is sampled at 100 Hz with Gaussian noise.

The true flux is built from three components, chosen for what each one does to the
inverse problem rather than for realism alone:

    ramp    constant deceleration means q is proportional to speed, so a straight ramp
            down over the braking time.  Every method recovers this.
    dip     a 0.3 s release of the pedal in the middle of the braking.  Marginally
            recoverable -- this is the component that separates the regularisations.
    ABS     an 8 Hz ripple.  Physically unrecoverable at 2 mm depth (see naive.py's
            cutoff frequency), and the repo says so instead of scoring anyone on it.

The event runs 4 s of braking followed by cooling out to t_end = 20 s.  The cooling
phase is not padding: it is what makes h and epsilon identifiable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .material import to_kelvin


@dataclass(frozen=True)
class Scenario:
    """One braking event, plus the measurement chain that observes it."""

    name: str = "showcase"

    # geometry
    L: float = 5.0e-3  # m, half thickness
    x_s: float = 2.0e-3  # m, thermocouple depth

    # event
    t_end: float = 20.0  # s, braking + cooling
    t_brake: float = 4.0  # s, braking phase only
    T0: float = to_kelvin(40.0)  # initial disc temperature
    T_inf: float = to_kelvin(25.0)  # ambient

    # surface losses
    h: float = 60.0  # W/m^2/K
    eps: float = 0.80  # emissivity (used only when the law has radiation on)

    # flux profile
    q_peak: float = 5.0e6  # W/m^2 at t = 0
    dip_t: float = 2.0  # s, centre of the pedal release
    dip_width: float = 0.30  # s, full width
    dip_depth: float = 0.60  # fraction of the local ramp value removed at the centre
    abs_amp: float = 0.15  # relative amplitude of the ABS ripple
    abs_freq: float = 8.0  # Hz

    # measurement
    sigma_noise: float = 0.5  # K
    f_sample: float = 100.0  # Hz
    noise_seed: int = 0

    # ----------------------------------------------------------------------------------

    def q_ramp(self, t: np.ndarray) -> np.ndarray:
        """The recoverable backbone: q proportional to vehicle speed, then zero."""
        t = np.asarray(t, dtype=float)
        return self.q_peak * np.clip(1.0 - t / self.t_brake, 0.0, None)

    def q_dip(self, t: np.ndarray) -> np.ndarray:
        """Pedal release: a compactly supported raised cosine, negative."""
        t = np.asarray(t, dtype=float)
        half = 0.5 * self.dip_width
        inside = np.abs(t - self.dip_t) <= half
        bump = np.where(inside, 0.5 * (1.0 + np.cos(np.pi * (t - self.dip_t) / half)), 0.0)
        return -self.dip_depth * float(self.q_ramp(self.dip_t)) * bump

    def q_true(self, t: np.ndarray) -> np.ndarray:
        """The flux the inverse methods are asked to recover, W/m^2."""
        t = np.asarray(t, dtype=float)
        ripple = 1.0 + self.abs_amp * np.sin(2.0 * np.pi * self.abs_freq * t)
        q = self.q_ramp(t) * ripple + self.q_dip(t)
        return np.clip(q, 0.0, None)  # no heat is pumped back into the pad

    def sensor_times(self) -> np.ndarray:
        """Sampling instants of the thermocouple, s."""
        n = int(round(self.t_end * self.f_sample))
        return np.arange(n + 1, dtype=float) / self.f_sample

    def with_seed(self, seed: int) -> "Scenario":
        return replace(self, noise_seed=seed)


#: The one scenario the figures are drawn from.
SHOWCASE = Scenario()

#: The only scenario a tunable is allowed to be set on.
#:
#: Every weight, bandwidth and schedule of the PINN is chosen here and then frozen, because
#: a loss weight picked on `SHOWCASE` would be a hyperparameter fitted to the figure it is
#: about to draw, and one picked on a member of `sample_scenarios` would be fitted to the
#: table (see the plan, 8.3).  It is a deliberately *different* stop -- longer, hotter at
#: the start, weaker peak flux, an earlier and shallower pedal release -- so that a setting
#: which only works on the showcase does not survive being moved here.
TUNING = Scenario(
    name="tuning",
    t_brake=4.6,
    T0=to_kelvin(55.0),
    q_peak=4.2e6,
    dip_t=1.4,
    dip_width=0.35,
    dip_depth=0.50,
    abs_amp=0.12,
    abs_freq=9.0,
    noise_seed=20250806,
)


def sample_scenarios(rng: np.random.Generator, n: int = 24) -> list[Scenario]:
    """The population the tables average over.

    Variation is in the *event*, not in the material or the sensor: peak flux, braking
    duration, where and how deep the pedal release falls, ABS amplitude and frequency,
    and the disc temperature at the start of the stop.  The cost here is per inversion,
    not per sample, so 24 is the right order -- not 400.
    """
    out: list[Scenario] = []
    for i in range(n):
        t_brake = float(rng.uniform(3.0, 5.5))
        out.append(
            Scenario(
                name=f"case{i:02d}",
                t_brake=t_brake,
                T0=to_kelvin(float(rng.uniform(30.0, 70.0))),
                q_peak=float(rng.uniform(3.5e6, 6.5e6)),
                dip_t=float(rng.uniform(0.30, 0.70)) * t_brake,
                dip_width=float(rng.uniform(0.20, 0.40)),
                dip_depth=float(rng.uniform(0.40, 0.75)),
                abs_amp=float(rng.uniform(0.08, 0.20)),
                abs_freq=float(rng.uniform(6.0, 10.0)),
                noise_seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return out
