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
optimizer), **M5** (Claude Agent SDK chat, all 8 planned tools, verified live) and
**M6** (Markdown report export from both labs). See `git log` for the exact history
and `PLATFORM_PLAN.md §6` for what each milestone covers.

Outstanding: the **demo GIF** called for in M6 is not in the repo — recording it needs
a headless browser driver and `ffmpeg`, neither of which is installed here, and adding
Playwright to the frontend's dependency tree for one animation is a call for the repo
owner to make. To produce it: `pnpm add -D playwright && npx playwright install
chromium`, drive `http://localhost:5173` through a shape change → Optimize → Verify,
capture frames, and assemble with `ffmpeg -i frame%04d.png docs/demo.gif`.

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

## Architecture

`flow-surrogate/` and `stress-gnn/` are treated as read-only libraries: nothing in the
backend reimplements their physics, it renders inputs, calls the model and reshapes
outputs for JSON. The service layer is the only thing that touches torch, and the agent
tools call those same service functions in-process — a tool is never an HTTP round trip
back into this app.

```mermaid
flowchart TB
    subgraph browser["Browser — React + Vite"]
        FL["Flow Lab<br/>shape editor · fields · optimizer"]
        SL["Stress Lab<br/>hole editor · von Mises · optimizer"]
        CS["Copilot sidebar<br/>⌘/Ctrl-J"]
    end

    subgraph api["FastAPI — engine_copilot.main"]
        R1["/api/flow/*<br/>predict · optimize (SSE) · verify · simulate (SSE)"]
        R2["/api/stress/*<br/>predict · optimize (SSE)"]
        R3["/api/report/generate"]
        R4["/api/agent/chat (SSE)"]
    end

    subgraph svc["Service layer — the only torch caller"]
        FS["flow_service.py"]
        LB["lbm_frames.py"]
        SS["stress_service.py"]
        RP["report.py<br/>deterministic template"]
        LK(["device_lock.py<br/>one process-wide MPS lock"])
    end

    subgraph agent["Claude Agent SDK"]
        AG["agent.py<br/>in-process MCP server"]
        TL["tools.py<br/>8 tools"]
        CLI["local claude CLI<br/>auth via claude login"]
    end

    subgraph libs["Research repos — read-only"]
        FSU["flowsurrogate<br/>U-Net · LBM solver"]
        SGN["stressgnn<br/>GNN · gmsh · FEM"]
    end

    FL --> R1
    SL --> R2
    FL --> R3
    SL --> R3
    CS --> R4

    R1 --> FS
    R1 --> LB
    R2 --> SS
    R3 --> RP
    R4 --> AG
    AG --> TL
    AG -.spawns.-> CLI
    TL --> FS
    TL --> SS
    TL --> RP

    FS --> LK
    LB --> LK
    SS --> LK
    LK --> FSU
    LK --> SGN
```

Two things that look odd until you know why. Every torch/MPS call and every gmsh mesh
build funnels through one process-wide lock — concurrent traffic on Apple's MPS backend
produced a real crash, and a single-GPU laptop has no throughput to lose anyway. And
`/api/report/generate` renders from a fixed template instead of calling Claude: the
`render_report` tool would otherwise re-enter the agent that called it.

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

## Reports

Both labs export a Markdown report once an optimisation has run — **Export report**, next
to *Verify with solver* in the Flow Lab and *Apply this design* in the Stress Lab. The
copilot can produce the same document on request ("write me a report"), because the chat
tool and the button call the same renderer.

The report is a fixed template, not a generated text: every number in it comes straight
from the solver or the surrogate. A flow report run before **Verify with solver** says in
as many words that nothing confirmed the surrogate — an un-verified optimum is reported
as exactly that.

## Tests

```bash
cd engine-copilot && ./.venv/bin/python -m pytest -q
```

`pyproject.toml` puts `backend/` on pytest's path, so no `PYTHONPATH` is needed. The
suite hits the real singletons — no mocking — so it loads the models, meshes a real
plate with gmsh, and takes ~55 s.
The two slowest tools, `flow_verify` and `flow_simulate`, are not covered there: both
run the real LBM for tens of seconds, and their service-layer counterparts already are.
