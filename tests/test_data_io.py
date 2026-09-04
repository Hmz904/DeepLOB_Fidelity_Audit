from pathlib import Path

import numpy as np

from deeplob_replication.data import (
    convert_setup2,
    create_synthetic_setup2,
    load_panel,
    save_panel,
)


def test_synthetic_setup2_roundtrip(tmp_path: Path):
    train, test = create_synthetic_setup2(tmp_path, train_events=160, test_events=140, seed=3)
    assert train.name == "synthetic_decimal_train7.npz"
    assert test.name == "synthetic_decimal_test3.npz"
    train_panel = load_panel(train)
    test_panel = load_panel(test)
    assert train_panel.features.shape == (160, 40)
    assert test_panel.labels.shape == (140, 5)
    copy = tmp_path / "copy.npz"
    save_panel(train_panel, copy)
    np.testing.assert_array_equal(load_panel(copy).labels, train_panel.labels)


def _fi_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = np.zeros((149, n), dtype=np.float32)
    raw[:40] = rng.normal(size=(40, n))
    raw[-5:] = rng.integers(1, 4, size=(5, n))
    return raw


def test_convert_setup2_creates_compact_panels(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw_dir.mkdir()
    names = [
        "Train_Dst_NoAuction_DecPre_CF_7.txt",
        "Test_Dst_NoAuction_DecPre_CF_7.txt",
        "Test_Dst_NoAuction_DecPre_CF_8.txt",
        "Test_Dst_NoAuction_DecPre_CF_9.txt",
    ]
    for i, name in enumerate(names):
        np.savetxt(raw_dir / name, _fi_matrix(120 + i, i))
    convert_setup2(raw_dir, processed, "decimal")
    assert load_panel(processed / "fi2010_decimal_train7.npz").features.shape == (120, 40)
    assert load_panel(processed / "fi2010_decimal_test3.npz").features.shape[0] == 121 + 122 + 123


def test_synthetic_generation_never_overwrites_fi2010_panels(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw_dir.mkdir()
    names = [
        "Train_Dst_NoAuction_DecPre_CF_7.txt",
        "Test_Dst_NoAuction_DecPre_CF_7.txt",
        "Test_Dst_NoAuction_DecPre_CF_8.txt",
        "Test_Dst_NoAuction_DecPre_CF_9.txt",
    ]
    for i, name in enumerate(names):
        np.savetxt(raw_dir / name, _fi_matrix(120 + i, i))
    convert_setup2(raw_dir, processed, "decimal")
    fi_train = processed / "fi2010_decimal_train7.npz"
    before = fi_train.read_bytes()

    synthetic_train, synthetic_test = create_synthetic_setup2(
        processed, train_events=160, test_events=140, seed=9
    )

    assert fi_train.read_bytes() == before
    assert synthetic_train.name == "synthetic_decimal_train7.npz"
    assert synthetic_test.name == "synthetic_decimal_test3.npz"
    assert synthetic_train != fi_train
