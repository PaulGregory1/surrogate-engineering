# CLAUDE.md

## Code map (read this first)

Before exploring, read `KNOWLEDGE-GRAPH.md`: it's this repo's **knowledge graph** (role of
every module + dependencies, route table, key flows, milestone status, and an inverse index
"task → files"). Use it to go straight to the right files **instead of re-scanning the whole
project**.

If you change the repo's structure, apply the *Update protocol* at the end of that graph
(git diff since the baseline → re-read only what moved → advance the baseline).

## Quick pointers

- Three independent parts, each with its **own** `.venv`: `flow-surrogate/`, `stress-gnn/`
  (ML research repos, read-only libraries from the platform's point of view), `engine-copilot/`
  (FastAPI + React platform on top). Always run `./.venv/bin/python` from inside the relevant
  directory, never a shared interpreter.
- `engine-copilot` backend: `cd engine-copilot && uvicorn engine_copilot.main:app --reload`. Frontend:
  `cd engine-copilot/frontend && pnpm dev`.
- No API key anywhere: the Claude Agent SDK chat (`engine-copilot/backend/engine_copilot/agent.py`)
  drives the local `claude` CLI via `claude login`, not `ANTHROPIC_API_KEY`.
- Nothing here is committed automatically — the user commits.
