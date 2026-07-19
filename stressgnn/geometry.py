"""Parametric family of perforated brackets.

A 2 x 1 plate with 2-4 circular holes, clamped on the left edge and loaded on the
right.  The design vector is (x, y, r) per hole plus the load angle -- so, unlike the
flow project, the *topology* changes from sample to sample and the mesh has a
different number of nodes every time.  That is the whole reason this project uses a
graph network instead of a CNN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WIDTH: float = 2.0
HEIGHT: float = 1.0

MIN_R: float = 0.08
MAX_R: float = 0.18
MARGIN: float = 0.07  # minimum ligament to the outer boundary
GAP: float = 0.09  # minimum ligament between two holes


@dataclass(frozen=True)
class Design:
    """One geometry + load case."""

    holes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (n, 3): x, y, r
    load_angle: float = -np.pi / 2  # traction direction on the right edge
    load_magnitude: float = 1.0

    @property
    def n_holes(self) -> int:
        return len(self.holes)

    @property
    def traction(self) -> np.ndarray:
        return self.load_magnitude * np.array([np.cos(self.load_angle), np.sin(self.load_angle)])

    def removed_area_fraction(self) -> float:
        return float(np.pi * (self.holes[:, 2] ** 2).sum() / (WIDTH * HEIGHT))

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.holes.ravel(), [self.load_angle]])


def is_valid(holes: np.ndarray) -> bool:
    """Holes must stay inside the plate and keep a minimum ligament between them."""
    if len(holes) == 0:
        return True
    x, y, r = holes[:, 0], holes[:, 1], holes[:, 2]
    if np.any(x - r < MARGIN) or np.any(x + r > WIDTH - MARGIN):
        return False
    if np.any(y - r < MARGIN) or np.any(y + r > HEIGHT - MARGIN):
        return False
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] + GAP:
                return False
    return True


def sample_design(
    rng: np.random.Generator,
    n_holes_choices: tuple[int, ...] = (2, 3, 4),
    angle_range: tuple[float, float] = (-np.pi / 2, 0.0),
    r_range: tuple[float, float] = (MIN_R, MAX_R),
) -> Design:
    """Rejection-sample a valid design."""
    n = int(rng.choice(n_holes_choices))
    for _ in range(2000):
        r = rng.uniform(*r_range, size=n)
        x = rng.uniform(MARGIN + r, WIDTH - MARGIN - r)
        y = rng.uniform(MARGIN + r, HEIGHT - MARGIN - r)
        holes = np.stack([x, y, r], axis=1)
        if is_valid(holes):
            return Design(holes=holes, load_angle=float(rng.uniform(*angle_range)))
    raise RuntimeError("could not sample a valid design; loosen the constraints")


def sample_designs(rng: np.random.Generator, n: int, **kwargs) -> list[Design]:
    return [sample_design(rng, **kwargs) for _ in range(n)]
