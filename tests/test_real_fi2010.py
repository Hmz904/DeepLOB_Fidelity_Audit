import os
from pathlib import Path

import pytest

from deeplob_replication.data import (
    LOBWindowDataset,
    load_panel,
    processed_panel_paths,
    split_train_validation,
)


@pytest.mark.skipif(
    os.environ.get("DEEPLOB_RUN_REAL_DATA") != "1",
    reason="real FI-2010 audit is opt-in",
)
def test_author_decimal_setup2_matches_released_notebook_window_counts():
    train_path, test_path = processed_panel_paths("data/processed", "fi2010", "decimal")
    assert Path(train_path).exists() and Path(test_path).exists()
    train = load_panel(train_path)
    test = load_panel(test_path)

    assert len(train.features) == 254_750
    assert len(test.features) == 139_587

    (train_start, train_end), (val_start, val_end) = split_train_validation(train, 0.20)
    train_ds = LOBWindowDataset(train, 10, 100, train_start, train_end)
    val_ds = LOBWindowDataset(train, 10, 100, val_start, val_end)
    test_ds = LOBWindowDataset(test, 10, 100)
    assert (len(train_ds), len(val_ds), len(test_ds)) == (203_701, 50_851, 139_488)
