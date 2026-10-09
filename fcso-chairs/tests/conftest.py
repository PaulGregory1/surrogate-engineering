import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pytest  # noqa: E402

from fcso import prior as P  # noqa: E402


@pytest.fixture(scope="session")
def prior():
    """AE only, on MPS (or CPU), parameters frozen."""
    return P.Prior(load_dm=False)


@pytest.fixture(scope="session")
def prior64():
    """AE only, on the CPU in float64, for gradcheck / finite differences."""
    pr = P.Prior(device="cpu", load_dm=False)
    pr.ae.double()
    return pr


@pytest.fixture(scope="session")
def opt_ids():
    return P.read_ids(os.path.join(ROOT, "splits", "opt.txt"))


@pytest.fixture(scope="session")
def record():
    """Collects numbers from the tests and writes them to runs/m4/test_results.json at the end."""
    out = {}
    yield out
    if out:
        os.makedirs(os.path.join(ROOT, "runs", "m4"), exist_ok=True)
        path = os.path.join(ROOT, "runs", "m4", "test_results.json")
        old = {}
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
        old.update(out)
        with open(path, "w") as f:
            json.dump(old, f, indent=1)
