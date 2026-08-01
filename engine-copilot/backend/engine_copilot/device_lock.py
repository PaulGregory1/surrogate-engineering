"""One process-wide lock around torch/MPS work.

FastAPI dispatches each request (and each SSE generator step) to its own worker
thread via `run_in_executor`, so predict/optimize/simulate calls can genuinely
overlap in time — a debounced slider predict firing while an optimize stream is
mid-iteration, two browser tabs open, etc. `torch.no_grad()`'s grad-mode guard is
thread-local (verified directly against this torch build), so that in isolation is
not the issue, but Apple's MPS backend is documented as unsafe for concurrent
multi-threaded kernel submission from a single process. A live full-stack run hit
exactly this: `cd.sum().backward()` failed with "does not require grad and does not
have a grad_fn" only under real concurrent traffic, never in an isolated repro —
consistent with a corrupted op recorded on the shared MPS command queue rather than
a logic error in the optimisation loop itself. Serialising all MPS-touching model
calls behind one lock removes the concurrent-submission path entirely.
"""

from __future__ import annotations

import threading

MPS_LOCK = threading.Lock()
