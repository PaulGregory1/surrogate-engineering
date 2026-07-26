"""Dataset generation, loading and normalisation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .geometry import Domain, N_SHAPE_PARAMS, equivalent_diameter, level_set, sample_shapes, solid_mask
from .lbm import LBMSolver, SolverConfig

#: Length scale used to squash the level set into [-1, 1] for the network input.
SDF_SCALE: float = 16.0
#: Reynolds number used to normalise the conditioning channel.
RE_SCALE: float = 100.0


@dataclass
class GenSpec:
    """One dataset split to generate."""

    name: str
    n: int
    seed: int
    re_range: tuple[float, float] = (30.0, 110.0)
    coeff_range: float = 0.16
    r0_range: tuple[float, float] = (8.5, 11.0)


def generate_split(
    spec: GenSpec,
    domain: Domain,
    solver_cfg: SolverConfig,
    out_dir: Path,
    batch: int = 64,
    device: str = "auto",
    compile: bool = True,
) -> Path:
    """Run the solver over a batch of random geometries and store the results."""
    rng = np.random.default_rng(spec.seed)
    params = sample_shapes(rng, spec.n, r0_range=spec.r0_range, coeff_range=spec.coeff_range)
    re = rng.uniform(*spec.re_range, size=spec.n)
    d_eq = np.array([equivalent_diameter(p) for p in params])
    masks = np.stack([solid_mask(p, domain) for p in params])

    solver = LBMSolver(domain, solver_cfg, device=device, compile=compile)
    fields = {k: [] for k in ("u", "v", "p")}
    scalars = {k: [] for k in ("cd", "cl", "cd_drift")}

    t0 = time.perf_counter()
    n_batches = int(np.ceil(spec.n / batch))
    for bi in range(n_batches):
        sl = slice(bi * batch, min((bi + 1) * batch, spec.n))
        # Keep every batch the same size: a changing batch dimension would force
        # torch.compile to re-trace the step function.
        idx = np.arange(sl.start, sl.stop)
        if len(idx) < batch:
            idx = np.concatenate([idx, np.full(batch - len(idx), idx[-1])])
        out = solver.run(masks[idx], re[idx], d_eq[idx])
        keep = slice(0, sl.stop - sl.start)
        for k in fields:
            fields[k].append(out[k][keep])
        for k in scalars:
            scalars[k].append(out[k][keep])
        done = sl.stop
        el = time.perf_counter() - t0
        print(
            f"  [{spec.name}] {done}/{spec.n}  {el:.0f}s elapsed  "
            f"{el / done:.2f} s/sample  eta {el / done * (spec.n - done) / 60:.1f} min",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.npz"
    np.savez_compressed(
        path,
        params=params.astype(np.float32),
        re=re.astype(np.float32),
        d_eq=d_eq.astype(np.float32),
        **{k: np.concatenate(v).astype(np.float32) for k, v in fields.items()},
        **{k: np.concatenate(v).astype(np.float32) for k, v in scalars.items()},
    )
    meta = {
        "spec": asdict(spec),
        "domain": asdict(domain),
        "solver": {k: v for k, v in asdict(solver_cfg).items() if k != "dtype"},
        "seconds_per_sample": (time.perf_counter() - t0) / spec.n,
        "device": str(solver.device),
    }
    (out_dir / f"{spec.name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [{spec.name}] wrote {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
    return path


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def input_channels(params: np.ndarray, re: np.ndarray, domain: Domain) -> np.ndarray:
    """Geometry + operating point, as an image the networks can consume.

    Channels: squashed level set, solid mask, Reynolds number.  The level set is
    passed through a tanh rather than clipped so that far-field cells keep a (small)
    gradient w.r.t. the shape -- that matters for the shape optimiser.
    """
    phi = np.stack([level_set(p, domain) for p in params])
    return np.stack(
        [
            np.tanh(phi / SDF_SCALE),
            (phi < 0).astype(np.float32),
            np.broadcast_to((re / RE_SCALE)[:, None, None], phi.shape),
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class Normalizer:
    """Per-channel standardisation, computed over fluid cells of the training split."""

    field_mean: np.ndarray
    field_std: np.ndarray
    cd_mean: float
    cd_std: float

    @classmethod
    def fit(cls, fields: np.ndarray, mask: np.ndarray, cd: np.ndarray) -> "Normalizer":
        m = mask[:, 0].astype(bool)
        mean = np.array([fields[:, c][m].mean() for c in range(fields.shape[1])])
        std = np.array([fields[:, c][m].std() for c in range(fields.shape[1])])
        return cls(mean, std, float(cd.mean()), float(cd.std()))

    def encode_fields(self, x: np.ndarray) -> np.ndarray:
        return (x - self.field_mean[None, :, None, None]) / self.field_std[None, :, None, None]

    def decode_fields(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.field_mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.as_tensor(self.field_std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return x * std + mean

    def encode_cd(self, x: np.ndarray) -> np.ndarray:
        return (x - self.cd_mean) / self.cd_std

    def decode_cd(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cd_std + self.cd_mean

    def to_json(self) -> dict:
        return {
            "field_mean": self.field_mean.tolist(),
            "field_std": self.field_std.tolist(),
            "cd_mean": self.cd_mean,
            "cd_std": self.cd_std,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Normalizer":
        return cls(np.array(d["field_mean"]), np.array(d["field_std"]), d["cd_mean"], d["cd_std"])


class FlowDataset(torch.utils.data.Dataset):
    """Tensors for one split.  Everything fits in memory (~50 MB per 256 samples)."""

    def __init__(
        self,
        path: Path | None,
        domain: Domain,
        normalizer: Normalizer | None = None,
        limit: int | None = None,
        arrays: dict[str, np.ndarray] | None = None,
    ):
        raw = arrays if arrays is not None else np.load(path)
        n = len(raw["re"]) if limit is None else min(limit, len(raw["re"]))
        self.params = raw["params"][:n]
        self.re = raw["re"][:n]
        self.d_eq = raw["d_eq"][:n]
        self.cd_raw = raw["cd"][:n]
        self.drift = raw["cd_drift"][:n]
        self.fields_raw = np.stack([raw["u"][:n], raw["v"][:n], raw["p"][:n]], axis=1)
        self.inputs_raw = input_channels(self.params, self.re, domain)
        self.mask = (self.inputs_raw[:, 1:2] < 0.5).astype(np.float32)  # 1 in the fluid

        self.normalizer = normalizer or Normalizer.fit(self.fields_raw, self.mask, self.cd_raw)
        fields = self.normalizer.encode_fields(self.fields_raw) * self.mask
        self.x = torch.from_numpy(self.inputs_raw)
        self.y = torch.from_numpy(fields.astype(np.float32))
        self.m = torch.from_numpy(self.mask)
        self.cd = torch.from_numpy(self.normalizer.encode_cd(self.cd_raw).astype(np.float32))

    def __len__(self) -> int:
        return len(self.re)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i], self.m[i], self.cd[i]

    def raw_arrays(self) -> dict[str, np.ndarray]:
        return {
            "params": self.params,
            "re": self.re,
            "d_eq": self.d_eq,
            "cd": self.cd_raw,
            "cd_drift": self.drift,
            "u": self.fields_raw[:, 0],
            "v": self.fields_raw[:, 1],
            "p": self.fields_raw[:, 2],
        }

    def augmented(self, extra: dict[str, np.ndarray], domain: Domain) -> "FlowDataset":
        """New dataset with `extra` samples appended, reusing this normalizer.

        Used by the active-learning loop: the designs the optimiser proposed, once
        labelled by the solver, are folded back into the training set.
        """
        base = self.raw_arrays()
        merged = {k: np.concatenate([base[k], extra[k]]) for k in base}
        return FlowDataset(None, domain, normalizer=self.normalizer, arrays=merged)


def load_splits(
    data_dir: Path, domain: Domain, limit_train: int | None = None
) -> tuple[FlowDataset, FlowDataset, FlowDataset, Normalizer]:
    train = FlowDataset(data_dir / "train.npz", domain, limit=limit_train)
    norm = train.normalizer
    val = FlowDataset(data_dir / "val.npz", domain, normalizer=norm)
    test = FlowDataset(data_dir / "test.npz", domain, normalizer=norm)
    return train, val, test, norm
