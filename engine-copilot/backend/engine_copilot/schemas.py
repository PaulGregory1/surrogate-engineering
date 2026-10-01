from typing import Literal

from pydantic import BaseModel, Field, field_validator

FlowModelKey = Literal["unet-phys", "unet", "fno"]
StressModelKey = Literal["gnn-mp10-glob", "gnn-mp10-glob-corr", "gnn-mp10"]


class FlowPredictRequest(BaseModel):
    coeffs: list[float] = Field(..., min_length=6, max_length=6, description="a2,b2,a3,b3,a4,b4")
    reynolds: float = Field(80.0, ge=10.0, le=200.0)
    model: FlowModelKey = "unet-phys"
    target_area: float | None = None


class FlowFields(BaseModel):
    u: list[list[float]]
    v: list[list[float]]
    p: list[list[float]]


class FlowPredictResponse(BaseModel):
    cd_surrogate: float
    cd_error_pct_expected: float
    fields: FlowFields
    mask: list[list[int]]
    domain: dict[str, float]
    boundary_xy: list[list[float]]
    target_area: float
    latency_ms: float
    model: FlowModelKey


class FlowVerifyRequest(BaseModel):
    coeffs: list[float] = Field(..., min_length=6, max_length=6)
    reynolds: float = Field(80.0, ge=10.0, le=200.0)
    target_area: float | None = None
    steps: int = Field(9000, ge=1000, le=20000)


class FlowVerifyResponse(BaseModel):
    cd_lbm: float
    cd_surrogate: float
    gap_pct: float
    fields: FlowFields
    wall_time_s: float


class FlowOptimizeRequest(BaseModel):
    # `coeffs` is the shape currently on screen. It seeds start #0 so the run actually
    # descends from the user's design; the other n_starts-1 starts stay random, which is
    # what keeps the search global (see flowsurrogate.optimize.optimise). Omit it and the
    # run is purely multi-start, as it was before.
    coeffs: list[float] | None = Field(None, min_length=6, max_length=6, description="a2,b2,a3,b3,a4,b4")
    reynolds: float = Field(80.0, ge=10.0, le=200.0)
    n_starts: int = Field(16, ge=1, le=64)
    iters: int = Field(250, ge=10, le=2000)
    model: FlowModelKey = "unet-phys"
    seed: int = 0


class FlowSimulateRequest(BaseModel):
    coeffs: list[float] = Field(..., min_length=6, max_length=6)
    reynolds: float = Field(80.0, ge=10.0, le=200.0)
    target_area: float | None = None
    n_frames: int = Field(60, ge=5, le=200)
    steps: int = Field(6000, ge=200, le=20000)


class StressPredictRequest(BaseModel):
    holes: list[list[float]] = Field(..., min_length=2, max_length=4, description="(x, y, r) per hole")
    load_angle_deg: float = Field(-90.0, ge=-180.0, le=180.0)
    model: StressModelKey = "gnn-mp10-glob"

    @field_validator("holes")
    @classmethod
    def _holes_are_triples(cls, v: list[list[float]]) -> list[list[float]]:
        if any(len(h) != 3 for h in v):
            raise ValueError("each hole must be [x, y, r]")
        return v


class StressField(BaseModel):
    vm: list[list[float | None]]
    extent: list[float]


class StressMesh(BaseModel):
    nodes: list[list[float]]
    triangles: list[list[int]]
    vm_nodal: list[float]


class StressPredictResponse(BaseModel):
    peak_vm_surrogate: float
    peak_error_pct_expected: float
    field: StressField
    mesh: StressMesh
    peak_location: list[float]
    vm_coarse_fem: list[list[float | None]]
    peak_vm_coarse_fem: float
    latency_ms: float
    model: StressModelKey


class StressOptimizeRequest(BaseModel):
    holes: list[list[float]] = Field(..., min_length=2, max_length=4)
    load_angle_deg: float = Field(-90.0, ge=-180.0, le=180.0)
    model: StressModelKey = "gnn-mp10-glob"
    n_iters: int = Field(30, ge=5, le=150)
    seed: int = 0

    @field_validator("holes")
    @classmethod
    def _holes_are_triples(cls, v: list[list[float]]) -> list[list[float]]:
        if any(len(h) != 3 for h in v):
            raise ValueError("each hole must be [x, y, r]")
        return v


class ReportRequest(BaseModel):
    project: Literal["flow", "stress"]
    # Whatever the matching /optimize stream ended with, plus anything the caller wants
    # to enrich it with (a `verify` block, the initial Cd). Deliberately unvalidated:
    # report.py treats every field as optional, so a partial context degrades to "n/a"
    # rows instead of a 422 in the middle of a chat turn.
    context: dict = Field(default_factory=dict)


class ReportResponse(BaseModel):
    markdown: str
