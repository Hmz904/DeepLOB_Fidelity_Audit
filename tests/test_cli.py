from pathlib import Path

from deeplob_replication.cli import main


def test_cli_synthetic_command(tmp_path: Path, monkeypatch, capsys):
    out = tmp_path / "processed"
    monkeypatch.setattr(
        "sys.argv",
        [
            "deeplob-rep",
            "synthetic",
            "--processed-dir",
            str(out),
            "--train-events",
            "150",
            "--test-events",
            "140",
        ],
    )
    main()
    captured = capsys.readouterr().out
    assert "SYNTHETIC" in captured
    assert (out / "synthetic_decimal_train7.npz").exists()
    assert (out / "synthetic_decimal_test3.npz").exists()


def test_multiseed_evaluation_sources_match_seed_specific_base_runs(tmp_path, monkeypatch):
    import pandas as pd
    import yaml

    from deeplob_replication.cli import _run_multiseed

    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "protocol": "author_tf1",
                "run_name": "eval-off",
                "evaluation": {"checkpoint_source_run": "fi2010-author-tf1"},
                "models": {"names": ["deeplob"]},
                "training": {"device": "cpu"},
                "output_dir": str(tmp_path / "outputs"),
            }
        )
    )
    seen = []

    def fake_run(cfg):
        seen.append((cfg.training.seed, cfg.evaluation.checkpoint_source_run))
        return pd.DataFrame(
            [
                {
                    "dataset": "fi2010",
                    "protocol": cfg.protocol,
                    "normalization": cfg.data.normalization,
                    "model": "deeplob",
                    "horizon": 10,
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "weighted_f1": 0.45,
                    "ece": 0.1,
                }
            ]
        )

    monkeypatch.setattr("deeplob_replication.cli.run", fake_run)
    _run_multiseed(str(config), [1, 2])
    assert seen == [
        (1, "fi2010-author-tf1-seed1"),
        (2, "fi2010-author-tf1-seed2"),
    ]
