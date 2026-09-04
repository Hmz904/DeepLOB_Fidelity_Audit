import numpy as np

from deeplob_replication.data import (
    EventPanel,
    LOBWindowDataset,
    parse_fi2010_matrix,
    split_train_validation,
)


def synthetic_panel(n: int = 130) -> EventPanel:
    x = np.arange(n * 40, dtype=np.float32).reshape(n, 40)
    y = np.tile(np.array([[0, 1, 2, 0, 1]], dtype=np.int64), (n, 1))
    return EventPanel(x, y)


def test_parser_accepts_original_149_by_n_orientation():
    raw = np.zeros((149, 120), dtype=float)
    raw[:40] = 1.0
    raw[-5:] = np.array([[1], [2], [3], [1], [2]])
    p = parse_fi2010_matrix(raw)
    assert p.features.shape == (120, 40)
    assert p.labels.shape == (120, 5)
    assert set(np.unique(p.labels)) == {0, 1, 2}


def test_lazy_window_alignment():
    p = synthetic_panel()
    ds = LOBWindowDataset(p, horizon=10, sequence_length=100)
    x, y = ds[0]
    assert x.shape == (100, 40)
    np.testing.assert_array_equal(x.numpy(), p.features[:100])
    assert y.item() == p.labels[99, 0]


def test_chronological_train_validation_split():
    p = synthetic_panel(1000)
    tr, va = split_train_validation(p, 0.2)
    assert tr == (0, 800)
    assert va == (800, 1000)


def test_validation_windows_do_not_cross_segment_boundary():
    p = synthetic_panel(1000)
    ds = LOBWindowDataset(p, horizon=10, sequence_length=100, start_event=800, end_event=1000)
    x, y = ds[0]
    assert len(ds) == 101
    np.testing.assert_array_equal(x.numpy(), p.features[800:900])
    assert y.item() == p.labels[899, 0]
