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
    domain: dict[str, int]
    boundary_xy: list[list[float]]
    target_area: float
    latency_ms: float
    model: FlowModelKey
