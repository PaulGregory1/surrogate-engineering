# Engine-Copilot

Local web platform exposing the `flow-surrogate` and `stress-gnn` surrogates behind
an interactive UI, plus a Claude-orchestrated chat that calls the same tools. Built
following `../PLATFORM_PLAN.md`.

## Status

Work in progress, built milestone by milestone (see `PLATFORM_PLAN.md §6`). Check
`git log` for what's landed so far — this README is updated as milestones complete.

## Setup

```bash
cd engine-copilot
python3.12 -m venv .venv
./.venv/bin/pip install -e ../flow-surrogate -e ../stress-gnn fastapi "uvicorn[standard]" pydantic numpy torch sse-starlette httpx pytest

cd frontend
pnpm install
```

No API key, no `.env` secret required for the model endpoints. The Chat tab (M5)
will need `claude login` once — see `PLATFORM_PLAN.md §7.5`.

## Run

```bash
# backend
cd engine-copilot/backend && PYTHONPATH=. ../.venv/bin/uvicorn engine_copilot.main:app --reload --port 8000

# frontend
cd engine-copilot/frontend && pnpm dev
```

Open `http://localhost:5173`.

## Tests

```bash
cd engine-copilot/backend && PYTHONPATH=. ../.venv/bin/python -m pytest tests/ -q
```
