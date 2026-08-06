from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FLOW_SURROGATE_DIR = ROOT / "flow-surrogate"

FLOW_RUNS_DIR = FLOW_SURROGATE_DIR / "runs"
FLOW_DATA_DIR = FLOW_SURROGATE_DIR / "data"
FLOW_RESULTS = FLOW_SURROGATE_DIR / "results" / "metrics.json"

# Field downscale factor for JSON transport (192x96 -> 96x48, etc.)
FIELD_DOWNSCALE = 2

# Model quality numbers, hand-copied from RESULTS.md so the UI/agent never
# fabricates an error figure. Keep these in sync if the models are retrained.
FLOW_MODEL_QUALITY = {
    "unet-phys": {"cd_mape_pct": 1.03, "cd_worst_pct": 4.2, "speedup_vs_lbm": 700},
    "unet": {"cd_mape_pct": 1.15, "cd_worst_pct": 3.9, "speedup_vs_lbm": 772},
    "fno": {"cd_mape_pct": 1.94, "cd_worst_pct": 8.8, "speedup_vs_lbm": 347},
}
