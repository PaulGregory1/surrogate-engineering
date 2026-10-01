# CLAUDE.md

## Code map (read this first)

Before exploring, read `KNOWLEDGE-GRAPH.md`: it's this repo's **knowledge graph** (role of
every module + dependencies, route table, key flows, milestone status, and an inverse index
"task → files"). Use it to go straight to the right files **instead of re-scanning the whole
project**.

If you change the repo's structure, apply the *Update protocol* at the end of that graph
(git diff since the baseline → re-read only what moved → advance the baseline).

## Quick pointers

- Three independent parts, each with its **own** `.venv`: `flow-surrogate/` and
  `stress-gnn/` (ML research repos, read-only libraries from the platform's point of view)
  and `engine-copilot/` (FastAPI + React platform on top). Always run `./.venv/bin/python`
  from inside the relevant directory, never a shared interpreter.
- **`main` carries the Flow Lab and the Stress Lab.** The one remaining lab lives on its
  own branch — `thermal-pinn` — and none of its files exist here. Don't re-add it to `main`
  without being asked; build on its branch instead.
- `stress-gnn/data/` and `stress-gnn/runs/` are gitignored but **must be present** for the
  backend to import: `StressService.__init__` loads `data/train.npz` for the normalisation
  stats at import time. Regenerate with `stress-gnn/run_all.py`, or copy them from the
  `stress-lab` worktree.
- `stress-gnn/` has **no `.venv` of its own on this branch yet** — `engine-copilot/.venv`
  carries `stressgnn` editable-installed, so both the platform and `stress-gnn/tests` run
  from there. Create one (`python3.12 -m venv .venv && ./.venv/bin/pip install -e .`) only
  if you need its `run_all.py` standalone.
- `engine-copilot` backend: `cd engine-copilot/backend && ../.venv/bin/python -m uvicorn
  engine_copilot.main:app --reload --port 8000`. It **must** run from `backend/`: the package is
  not pip-installed into the venv, it's imported from the source tree, so any other cwd gives
  `ModuleNotFoundError: No module named 'engine_copilot'`. Frontend:
  `cd engine-copilot/frontend && pnpm dev` (Vite on :5173, proxies `/api` to the backend).
- No API key anywhere: the Claude Agent SDK chat (`engine-copilot/backend/engine_copilot/agent.py`)
  drives the local `claude` CLI via `claude login`, not `ANTHROPIC_API_KEY`.
- Nothing here is committed automatically — the user commits.
