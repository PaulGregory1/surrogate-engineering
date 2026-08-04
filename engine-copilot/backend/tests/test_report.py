from engine_copilot.report import render_report

import pytest


def test_flow_report_quotes_both_cd_and_the_reduction():
    md = render_report(
        "flow",
        {
            "cd_surrogate_final": 1.4412,
            "history": {"cd_best": [1.90, 1.70, 1.55, 1.4412]},
            "reynolds": 80.0,
            "final_coeffs": [0.05, -0.02, 0.03, 0.01, -0.04, 0.0],
            "target_area": 0.0123,
            "model": "unet-phys",
        },
    )
    assert "1.9000" in md and "1.4412" in md
    assert "-24.15 %" in md  # (1.90 - 1.4412) / 1.90
    assert "1.03 %" in md  # expected surrogate error, never omitted


def test_flow_report_flags_extrapolation_outside_the_training_range():
    md = render_report("flow", {"reynolds": 400.0, "history": {"cd_best": [2.0, 1.8]}})
    assert "Hors domaine d'entraînement" in md


def test_flow_report_without_verification_says_so():
    md = render_report("flow", {"history": {"cd_best": [2.0, 1.8]}})
    assert "Aucune vérification LBM" in md


def test_flow_report_with_verification_quotes_the_gap():
    md = render_report(
        "flow",
        {
            "history": {"cd_best": [2.0, 1.8]},
            "verify": {"cd_lbm": 1.85, "cd_surrogate": 1.80, "gap_pct": 2.7, "wall_time_s": 31.2},
        },
    )
    assert "1.8500" in md and "2.70 %" in md


def test_partial_context_degrades_to_na_instead_of_raising():
    md = render_report("flow", {})
    assert "n/a" in md
    assert md.startswith("# Rapport d'optimisation")


def test_stress_report_lists_the_final_holes():
    md = render_report(
        "stress",
        {
            "peak_vm_initial": 2.5,
            "peak_vm_final": 2.0,
            "final_holes": [[0.1, 0.2, 0.03], [0.3, 0.4, 0.05]],
            "load_angle_deg": -90.0,
            "model": "gnn-mp10-glob",
        },
    )
    assert "-20.00 %" in md
    assert "0.0300" in md and "0.0500" in md
    assert "6.56 %" in md


def test_unknown_project_raises():
    with pytest.raises(ValueError):
        render_report("thermal", {})
