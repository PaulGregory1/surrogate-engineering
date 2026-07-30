# Engine-Copilot

Local web platform exposing the `flow-surrogate` and `stress-gnn` surrogates behind
an interactive UI, plus a Claude-orchestrated chat that calls the same tools. Built
following `../PLATFORM_PLAN.md`.

## Status

Landed: **M0** (scaffolding), **M1** (flow prediction, real U-Net surrogate), most of
**M3** (optimizer backend — SSE endpoint built, frontend panel not wired yet), and
**M5** (Claude Agent SDK chat, real tool-calling, verified live). Not yet started:
**M2** (LBM animation), **M4** (Stress Lab), **M6** (report/polish). See `git log`
for the exact history and `PLATFORM_PLAN.md §6` for what each milestone covers.

Known limitation: the chat agent resumes context via the SDK's `resume` session id
rather than a fully server-managed multi-session model — fine for one user, would
need per-browser-tab session handling for multiple concurrent users.

## Setup

```bash
cd engine-copilot
python3.12 -m venv .venv
./.venv/bin/pip install -e ../flow-surrogate -e ../stress-gnn -e .

cd frontend
pnpm install

# Chat tab only — no API key, ever:
npm install -g @anthropic-ai/claude-code
claude login
```

No API key, no `.env` secret anywhere in this project. The Chat tab runs on the
**Claude Agent SDK**, authenticated via the local `claude login` session (your
Claude Code subscription) — see `PLATFORM_PLAN.md §5, §7.5`.

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
