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


def test_flow_report_reads_the_flow_optimize_payload_verbatim():
    """The agent forwards `flow_optimize`'s own result, which spells two keys
    differently from the lab's context. Both spellings must land in the document —
    the tool description promises the payload can be passed through as-is."""
    md = render_report(
        "flow",
        {
            "cd_initial": 1.7660,
            "cd_final": 1.5490,
            "reduction_pct": 12.29,
            "final_coeffs": [0.16, -0.0076, -0.0103, 0.002, 0.014, -0.001],
            "cd_error_pct_expected": 1.03,
            "reynolds": 80.0,
            "iters": 100,
            "convergence_cd_best": [1.7660, 1.62, 1.57, 1.5490],
            "target_area": 306.92,
        },
    )
    assert "1.7660" in md and "1.5490" in md
    assert "-12.29 %" in md
    assert "No convergence history was supplied" not in md
    assert "| Iteration | Best |" in md


def test_flow_report_flags_extrapolation_outside_the_training_range():
    md = render_report("flow", {"reynolds": 400.0, "history": {"cd_best": [2.0, 1.8]}})
    assert "Outside the training range" in md


def test_flow_report_without_verification_says_so():
    md = render_report("flow", {"history": {"cd_best": [2.0, 1.8]}})
    assert "No LBM verification" in md


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
    assert md.startswith("# Optimisation report")


def test_unknown_project_raises():
    with pytest.raises(ValueError):
        render_report("thermal", {})
