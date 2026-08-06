"""Grey cast iron, in three deliberately different constitutive laws.

The three laws are the three acts of the repo, and they differ only in how hard they
make the inverse problem:

    A  linear          k, c constant, no radiation.  Superposition holds, so Duhamel /
                       Beck / Tikhonov are exactly applicable.  This is the fair fight.
    B  smooth          k, c affine in temperature, radiation on.  Superposition is gone;
                       the classical route survives by freezing properties at a
                       reference temperature, and the bias that costs is what act 2
                       measures.
    C  knee            law B plus a localised -12 % step in k around 450 degC.  The step
                       belongs to no parametric family an engineer would guess, which is
                       precisely the assumption the adjoint route needs.

Everything is in SI and in kelvin.  Kelvin rather than celsius because the radiation
term needs absolute temperature, and one unit system throughout is cheaper than
remembering where the conversion lives.

Property fits are referenced to 20 degC (`T_REF`): `k(T_REF) == k0` and `c(T_REF) == c0`
for all three laws, which is also the normalisation the PINN's learned `k_psi` is held to.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: 0 degC in kelvin.
KELVIN: float = 273.15
#: Reference temperature of every property fit: 20 degC.
T_REF: float = 293.15
#: Stefan-Boltzmann constant, W/m^2/K^4.
SIGMA_SB: float = 5.670374419e-8


def to_kelvin(t_celsius: float | np.ndarray) -> float | np.ndarray:
    return t_celsius + KELVIN


def to_celsius(t_kelvin: float | np.ndarray) -> float | np.ndarray:
    return t_kelvin - KELVIN


@dataclass(frozen=True)
class Material:
    """One constitutive law.

    `k_slope` and `c_slope` are *relative* slopes, per kelvin, measured from `T_REF`:

        k(T) = k0 (1 + k_slope (T - T_REF)) (1 - knee_drop s(T))
        c(T) = c0 (1 + c_slope (T - T_REF))

    with `s` a tanh step from 0 to 1 centred on `knee_T`, going from ~5 % to ~95 % of
    its rise across `knee_width`.  `knee_drop = 0` removes the knee entirely.

    `radiation` is a property of the law rather than of the surface because it is what
    distinguishes act 1 from acts 2 and 3; the emissivity itself lives on the
    `Scenario`, and the solver multiplies the two.
    """

    name: str
    rho: float = 7200.0  # kg/m^3
    k0: float = 52.0  # W/m/K at T_REF
    c0: float = 460.0  # J/kg/K at T_REF
    k_slope: float = 0.0  # 1/K
    c_slope: float = 0.0  # 1/K
    knee_drop: float = 0.0  # dimensionless, relative drop across the knee
    knee_T: float = 723.15  # 450 degC
    knee_width: float = 60.0  # K
    radiation: bool = False

    # ----------------------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------------------

    def _knee(self, T: np.ndarray) -> np.ndarray:
        """Smooth 0 -> 1 step across the knee."""
        if self.knee_drop == 0.0:
            return np.zeros_like(np.asarray(T, dtype=float))
        return 0.5 * (1.0 + np.tanh((np.asarray(T, dtype=float) - self.knee_T) / (0.5 * self.knee_width)))

    def _dknee(self, T: np.ndarray) -> np.ndarray:
        if self.knee_drop == 0.0:
            return np.zeros_like(np.asarray(T, dtype=float))
        a = 0.5 * self.knee_width
        return 0.5 / (a * np.cosh((np.asarray(T, dtype=float) - self.knee_T) / a) ** 2)

    def k(self, T: np.ndarray) -> np.ndarray:
        """Thermal conductivity, W/m/K."""
        dT = np.asarray(T, dtype=float) - T_REF
        return self.k0 * (1.0 + self.k_slope * dT) * (1.0 - self.knee_drop * self._knee(T))

    def dk_dT(self, T: np.ndarray) -> np.ndarray:
        """dk/dT, W/m/K^2 -- needed by the Newton solve, not just by curiosity."""
        dT = np.asarray(T, dtype=float) - T_REF
        lin, step = 1.0 + self.k_slope * dT, 1.0 - self.knee_drop * self._knee(T)
        return self.k0 * (self.k_slope * step - lin * self.knee_drop * self._dknee(T))

    def c(self, T: np.ndarray) -> np.ndarray:
        """Specific heat, J/kg/K."""
        return self.c0 * (1.0 + self.c_slope * (np.asarray(T, dtype=float) - T_REF))

    def enthalpy(self, T: np.ndarray) -> np.ndarray:
        """Specific enthalpy relative to `T_REF`, J/kg -- the exact integral of `c`.

        The solver advances *this*, not `rho c dT/dt`, which is what makes the discrete
        energy balance close to round-off even when `c` depends on temperature.
        """
        dT = np.asarray(T, dtype=float) - T_REF
        return self.c0 * (dT + 0.5 * self.c_slope * dT**2)

    @property
    def alpha0(self) -> float:
        """Thermal diffusivity at `T_REF`, m^2/s -- sets the Fourier number and f_c."""
        return self.k0 / (self.rho * self.c0)

    @property
    def is_linear(self) -> bool:
        return self.k_slope == 0.0 and self.c_slope == 0.0 and self.knee_drop == 0.0

    # ----------------------------------------------------------------------------------
    # Kirchhoff transform
    # ----------------------------------------------------------------------------------
    #
    # u(T) = (1/k0) int_{T_REF}^{T} k(T') dT' turns div(k(T) grad T) into k0 div(grad u),
    # i.e. it linearises the *operator*.  It is the textbook way around k(T), and
    # BACKGROUND.md uses it to explain why the classical route still loses here: it does
    # nothing for rho c(T) dT/dt in a transient, and it does not commute with the
    # radiation boundary condition, which is nonlinear in T and not in u.

    def kirchhoff(self, T: np.ndarray) -> np.ndarray:
        """u(T), in kelvin (normalised by k0, so u == T - T_REF when k is constant)."""
        if self.knee_drop == 0.0:
            dT = np.asarray(T, dtype=float) - T_REF
            return dT + 0.5 * self.k_slope * dT**2
        grid, table = _kirchhoff_table(self)
        return np.interp(np.asarray(T, dtype=float), grid, table)

    def kirchhoff_inverse(self, u: np.ndarray) -> np.ndarray:
        """T(u).  Well defined because k > 0 makes u strictly increasing in T."""
        u = np.asarray(u, dtype=float)
        if self.knee_drop == 0.0:
            if self.k_slope == 0.0:
                return u + T_REF
            disc = 1.0 + 2.0 * self.k_slope * u
            return T_REF + (np.sqrt(np.maximum(disc, 0.0)) - 1.0) / self.k_slope
        grid, table = _kirchhoff_table(self)
        return np.interp(u, table, grid)


@lru_cache(maxsize=8)
def _kirchhoff_table(material: Material) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative integral of k/k0 on a fine temperature table.

    Only laws with a knee need this: the tanh has no elementary antiderivative once it
    is multiplied by the affine part.  The table is fine enough (0.02 K) that both the
    quadrature and the linear interpolation are far below every tolerance in the repo;
    `tests/test_material.py` checks the round trip and the derivative.
    """
    grid = np.arange(T_REF - 300.0, T_REF + 1700.0 + 1e-9, 0.02)
    k = material.k(grid) / material.k0
    table = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(grid))])
    return grid, table - np.interp(T_REF, grid, table)


#: Act 1: constant properties, no radiation.  Superposition holds.
LAW_A = Material(name="A")

#: Act 2: smooth temperature dependence and radiation.  Superposition is gone.
LAW_B = Material(name="B", k_slope=-4.5e-4, c_slope=7.0e-4, radiation=True)

#: Act 3: law B plus a -12 % knee in k at 450 degC -- the function the classical route
#: has to guess the shape of, and the one the PINN is asked to discover.
LAW_C = Material(name="C", k_slope=-4.5e-4, c_slope=7.0e-4, knee_drop=0.12, radiation=True)

LAWS: dict[str, Material] = {"A": LAW_A, "B": LAW_B, "C": LAW_C}
