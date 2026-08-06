from typing import Literal

from pydantic import BaseModel, Field

FlowModelKey = Literal["unet-phys", "unet", "fno"]


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


class ReportRequest(BaseModel):
    project: Literal["flow"]
    # Whatever the matching /optimize stream ended with, plus anything the caller wants
    # to enrich it with (a `verify` block, the initial Cd). Deliberately unvalidated:
    # report.py treats every field as optional, so a partial context degrades to "n/a"
    # rows instead of a 422 in the middle of a chat turn.
    context: dict = Field(default_factory=dict)


class ReportResponse(BaseModel):
    markdown: str
