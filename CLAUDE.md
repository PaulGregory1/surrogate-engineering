# CLAUDE.md

## Code map (read this first)

Before exploring, read `KNOWLEDGE-GRAPH.md`: it's this repo's **knowledge graph** (role of
every module + dependencies, route table, key flows, milestone status, and an inverse index
"task → files"). Use it to go straight to the right files **instead of re-scanning the whole
project**.

If you change the repo's structure, apply the *Update protocol* at the end of that graph
(git diff since the baseline → re-read only what moved → advance the baseline).

## Quick pointers

- Two independent parts, each with its **own** `.venv`: `flow-surrogate/` (ML research repo,
  a read-only library from the platform's point of view) and `engine-copilot/` (FastAPI +
  React platform on top). Always run `./.venv/bin/python` from inside the relevant
  directory, never a shared interpreter.
- **`main` carries the Flow Lab only.** The other labs live on their own branches —
  `stress-lab`, `bridge-lab`, `thermal-pinn` — and none of their files exist here. Don't
  re-add one to `main` without being asked; build on its branch instead.
- `engine-copilot` backend: `cd engine-copilot/backend && ../.venv/bin/python -m uvicorn
  engine_copilot.main:app --reload --port 8000`. It **must** run from `backend/`: the package is
  not pip-installed into the venv, it's imported from the source tree, so any other cwd gives
  `ModuleNotFoundError: No module named 'engine_copilot'`. Frontend:
  `cd engine-copilot/frontend && pnpm dev` (Vite on :5173, proxies `/api` to the backend).
- No API key anywhere: the Claude Agent SDK chat (`engine-copilot/backend/engine_copilot/agent.py`)
  drives the local `claude` CLI via `claude login`, not `ANTHROPIC_API_KEY`.
- Nothing here is committed automatically — the user commits.
