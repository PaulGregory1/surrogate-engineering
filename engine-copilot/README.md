# Engine-Copilot

Local web platform exposing the `flow-surrogate` and `stress-gnn` surrogates behind
an interactive UI, plus a Claude-orchestrated chat that calls the same tools. Built
following `../PLATFORM_PLAN.md`.

## Status

Landed: **M0** (scaffolding), **M1** (flow prediction, real U-Net surrogate), **M2**
(client-side RK4 streamlines + real LBM frame streaming with a live Von Kármán wake),
**M3** (optimizer panel with live convergence plot + shape morph, verify-with-solver
against the real LBM), **M4** (Stress Lab — hole editor, GNN peak-stress prediction
with a mesh-native surrogate, coarse-FEM comparison, randomised-local-search hole
optimizer), and **M5** (Claude Agent SDK chat, real tool-calling, verified live). Not
yet started: **M6** (report export, demo GIF, final polish). See `git log` for the
exact history and `PLATFORM_PLAN.md §6` for what each milestone covers.

Known limitations:
- The chat agent resumes context via the SDK's `resume` session id rather than a
  fully server-managed multi-session model — fine for one user, would need
  per-browser-tab session handling for multiple concurrent users.
- All torch/MPS model calls (flow predict/optimize/verify, LBM frame streaming,
  stress predict) are serialised behind one process-wide lock
  (`engine_copilot/device_lock.py`) — a live full-stack run surfaced a real crash
  (`cd.sum().backward()` losing its grad_fn) under concurrent request traffic on
  Apple's MPS backend; the lock trades a little throughput for correctness on a
  single-GPU laptop. gmsh mesh building (Stress Lab) is similarly serialised and its
  one-time `gmsh.initialize()` is forced onto the main thread at import time, since
  it registers a SIGINT handler that only CPython's main thread is allowed to touch.

## Setup

```bash
cd engine-copilot
python3.12 -m venv .venv
./.venv/bin/pip install -e ../flow-surrogate -e ../stress-gnn -e .

cd frontend
pnpm install

# Copilot sidebar only — no API key, ever:
npm install -g @anthropic-ai/claude-code
claude login
```

No API key, no `.env` secret anywhere in this project. The copilot runs on the
**Claude Agent SDK**, authenticated via the local `claude login` session (your
Claude Code subscription) — see `PLATFORM_PLAN.md §5, §7.5`.

The copilot is a sidebar on every page (⌘/Ctrl-J), not a separate tab: it receives
the design currently on screen, so you can ask about it without restating the numbers.

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
