import asyncio
import json
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from pydantic import BaseModel

from .agent import agent_session
from .flow_service import flow_service
from .schemas import FlowPredictRequest, FlowPredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # flow_service is already loaded at import time (module-level singleton);
    # this just warms the model's first-call JIT/MPS overhead so the first
    # real request from the UI isn't the slow one.
    await asyncio.get_event_loop().run_in_executor(None, flow_service.warmup)
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
async def flow_optimize(reynolds: float = 80.0, n_starts: int = 8, iters: int = 120, model: str = "unet-phys", seed: int = 0):
    async def event_stream():
        loop = asyncio.get_event_loop()
        gen = flow_service.optimise_streaming(reynolds, n_starts, iters, model, seed)
        while True:
            item = await loop.run_in_executor(None, lambda: next(gen, None))
            if item is None:
                break
            event = "done" if item.get("done") else "iteration"
            yield {"event": event, "data": json.dumps(item)}

    return EventSourceResponse(event_stream())


class ChatRequest(BaseModel):
    message: str


@app.post("/api/agent/chat")
async def agent_chat(req: ChatRequest):
    async def event_stream():
        async for event in agent_session.chat(req.message):
            yield {"event": event["type"], "data": json.dumps(event, default=str)}

    return EventSourceResponse(event_stream())
