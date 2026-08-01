from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FLOW_SURROGATE_DIR = ROOT / "flow-surrogate"
STRESS_GNN_DIR = ROOT / "stress-gnn"

FLOW_RUNS_DIR = FLOW_SURROGATE_DIR / "runs"
FLOW_DATA_DIR = FLOW_SURROGATE_DIR / "data"
FLOW_RESULTS = FLOW_SURROGATE_DIR / "results" / "metrics.json"

STRESS_RUNS_DIR = STRESS_GNN_DIR / "runs"
STRESS_DATA_DIR = STRESS_GNN_DIR / "data"

# Field downscale factor for JSON transport (192x96 -> 96x48, etc.)
FIELD_DOWNSCALE = 2

# Model quality numbers, hand-copied from RESULTS.md so the UI/agent never
# fabricates an error figure. Keep these in sync if the models are retrained.
FLOW_MODEL_QUALITY = {
    "unet-phys": {"cd_mape_pct": 1.03, "cd_worst_pct": 4.2, "speedup_vs_lbm": 700},
    "unet": {"cd_mape_pct": 1.15, "cd_worst_pct": 3.9, "speedup_vs_lbm": 772},
    "fno": {"cd_mape_pct": 1.94, "cd_worst_pct": 8.8, "speedup_vs_lbm": 347},
}

# Same idea, hand-copied from stress-gnn/results/metrics.json (test split).
STRESS_MODEL_QUALITY = {
    "gnn-mp10-glob": {"peak_err_pct": 6.56, "field_rel_l2": 0.158},
    "gnn-mp10-glob-corr": {"peak_err_pct": 6.04, "field_rel_l2": 0.042},
    "gnn-mp10": {"peak_err_pct": 5.98, "field_rel_l2": 0.155},
}

STRESS_H_COARSE = 0.045
STRESS_GRID_NX = 128
STRESS_GRID_NY = 64
