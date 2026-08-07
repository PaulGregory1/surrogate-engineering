import asyncio
import json
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from pydantic import BaseModel

from .agent import agent_session
from .flow_service import flow_service
from .lbm_frames import simulate_streaming
from .report import render_report
from .schemas import (
    FlowOptimizeRequest,
    FlowPredictRequest,
    FlowPredictResponse,
    FlowSimulateRequest,
    FlowVerifyRequest,
    FlowVerifyResponse,
    ReportRequest,
    ReportResponse,
)


def _warm_lbm_solver() -> None:
    # torch.compile's trace happens on the first call to `_step_fn`; pay that cost
    # once at startup (a handful of steps is enough to trigger it) rather than on
    # whichever browser tab first clicks "Simulate".
    coeffs = [0.0] * 6
    list(
        simulate_streaming(
            flow_service.domain, flow_service.device, coeffs, flow_service.default_target_area, 80.0, n_frames=2, steps=20
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The service is already loaded at import time (module-level singleton); this just
    # warms first-call JIT/MPS overhead so the first real request from the UI isn't
    # the slow one.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, flow_service.warmup)
    await loop.run_in_executor(None, _warm_lbm_solver)
    yield


app = FastAPI(title="Engine-Copilot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        # Proxy check for M0/M1 — full `claude auth status` wiring lands in M5.
        "claude_cli_installed": shutil.which("claude") is not None,
    }


@app.post("/api/flow/predict", response_model=FlowPredictResponse)
def flow_predict(req: FlowPredictRequest) -> dict:
    return flow_service.predict(
        coeffs=req.coeffs,
        reynolds=req.reynolds,
        model_key=req.model,
        target_area=req.target_area,
    )


@app.post("/api/flow/optimize")
async def flow_optimize(req: FlowOptimizeRequest):
    async def event_stream():
        loop = asyncio.get_event_loop()
        gen = flow_service.optimise_streaming(
            req.reynolds, req.n_starts, req.iters, req.model, req.seed, init_coeffs=req.coeffs
        )
        while True:
            item = await loop.run_in_executor(None, lambda: next(gen, None))
            if item is None:
                break
            event = "done" if item.get("done") else "iteration"
            yield {"event": event, "data": json.dumps(item)}

    return EventSourceResponse(event_stream())


@app.post("/api/flow/verify", response_model=FlowVerifyResponse)
async def flow_verify(req: FlowVerifyRequest) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: flow_service.verify(req.coeffs, req.reynolds, req.target_area, req.steps),
    )


@app.post("/api/flow/simulate")
async def flow_simulate(req: FlowSimulateRequest):
    target_area = req.target_area or flow_service.default_target_area

    async def event_stream():
        loop = asyncio.get_event_loop()
        gen = simulate_streaming(
            flow_service.domain,
            flow_service.device,
            req.coeffs,
            target_area,
            req.reynolds,
            req.n_frames,
            req.steps,
        )
        while True:
            item = await loop.run_in_executor(None, lambda: next(gen, None))
            if item is None:
                break
            event = "done" if item.get("done") else "frame"
            yield {"event": event, "data": json.dumps(item)}

    return EventSourceResponse(event_stream())


@app.post("/api/report/generate", response_model=ReportResponse)
def report_generate(req: ReportRequest) -> dict:
    # Synchronous and cheap: rendering is pure string formatting (see report.py on why
    # this does not call Claude), so there is nothing to offload to a thread.
    try:
        return {"markdown": render_report(req.project, req.context)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ChatRequest(BaseModel):
    message: str


@app.post("/api/agent/chat")
async def agent_chat(req: ChatRequest):
    async def event_stream():
        async for event in agent_session.chat(req.message):
            yield {"event": event["type"], "data": json.dumps(event, default=str)}

    return EventSourceResponse(event_stream())


@app.post("/api/agent/reset")
def agent_reset() -> dict:
    """Drop the resumed session behind *Clear the conversation*.

    Clearing the transcript client-side is not enough: the SDK conversation is resumed
    by id on the server, so without this the agent still answers out of the turns the
    user just deleted. Nothing to offload to a thread -- it assigns one attribute.
    """
    agent_session.reset()
    return {"ok": True}
