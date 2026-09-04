"""Cover the figure script without requiring a completed FI-2010 run.

``render_results`` filters to FI-2010 DeepLOB rows with paper reference values, so a
synthetic smoke run can never exercise it. These tests feed it FI-2010-shaped frames so
the aggregation path is not first executed on the day real results arrive.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib", reason="figure rendering needs the [report] extra")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_results.py"
spec = importlib.util.spec_from_file_location("render_results", SCRIPT)
render_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_results)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(horizon: int, weighted_f1: float, paper_f1: float, **extra) -> dict:
    row = {
        "dataset": "fi2010",
        "model": "deeplob",
        "horizon": horizon,
        "weighted_f1": weighted_f1,
        "paper_f1": paper_f1,
    }
    row.update(extra)
    return row


def test_single_run_aggregates_to_one_row_per_horizon_with_zero_std():
    z = render_results._aggregate(
        _frame([_row(10, 0.79, 0.8340), _row(20, 0.70, 0.7282)])
    )
    assert list(z["horizon"]) == [10, 20]
    assert list(z["n_runs"]) == [1, 1]
    assert list(z["ours_std"]) == [0.0, 0.0]
    assert z.loc[z["horizon"] == 10, "ours_mean"].item() == pytest.approx(0.79)


def test_multi_seed_rows_average_and_expose_dispersion():
    z = render_results._aggregate(
        _frame(
            [
                _row(10, 0.78, 0.8340, base_seed=1),
                _row(10, 0.80, 0.8340, base_seed=2),
                _row(10, 0.79, 0.8340, base_seed=3),
            ]
        )
    )
    assert z["n_runs"].item() == 3
    assert z["ours_mean"].item() == pytest.approx(0.79)
    assert z["ours_std"].item() > 0


def test_non_fi2010_and_non_deeplob_rows_are_excluded():
    frame = _frame(
        [
            _row(10, 0.79, 0.8340),
            {**_row(10, 0.99, 0.8340), "dataset": "synthetic"},
            {**_row(10, 0.99, 0.8340), "model": "linear"},
        ]
    )
    z = render_results._aggregate(frame)
    assert z["n_runs"].item() == 1
    assert z["ours_mean"].item() == pytest.approx(0.79)


def test_rows_without_paper_reference_are_rejected_with_a_clear_error():
    frame = _frame([_row(100, 0.6, None)])
    with pytest.raises(ValueError, match="no FI-2010 DeepLOB rows"):
        render_results._aggregate(frame)


def test_label_reports_the_number_of_runs_behind_the_error_bars():
    single = _frame([_row(10, 0.79, 0.8340)])
    repeated = _frame([_row(10, 0.78, 0.8340), _row(10, 0.80, 0.8340)])
    assert "single run" in render_results._independent_label(
        render_results._aggregate(single)
    )
    assert "n=2" in render_results._independent_label(
        render_results._aggregate(repeated)
    )


def test_figure_is_written_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    metrics = tmp_path / "metrics.csv"
    _frame([_row(10, 0.79, 0.8340), _row(50, 0.77, 0.8035)]).to_csv(metrics, index=False)
    out = tmp_path / "figures" / "gap.png"
    monkeypatch.setattr(
        "sys.argv", ["render_results", str(metrics), "--out", str(out)]
    )
    render_results.main()
    assert out.exists() and out.stat().st_size > 0
