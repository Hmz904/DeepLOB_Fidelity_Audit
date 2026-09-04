import numpy as np

from deeplob_replication.metrics import classification_metrics, expected_calibration_error


def test_perfect_classification_metrics():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y, y.copy())
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["weighted_f1"] == 1.0
    assert m["weighted_recall"] == m["accuracy"]
    assert m["n_obs"] == 6


def test_weighted_recall_equals_accuracy_for_single_label_multiclass():
    y = np.array([0, 0, 0, 1, 2])
    pred = np.array([0, 1, 0, 1, 1])
    m = classification_metrics(y, pred)
    assert abs(m["weighted_recall"] - m["accuracy"]) < 1e-12


def test_ece_is_zero_for_perfect_confident_predictions():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    assert expected_calibration_error(p, y) == 0.0


def test_confusion_has_fixed_three_class_shape():
    from deeplob_replication.metrics import confusion

    y = np.array([0, 0, 1, 2])
    pred = np.array([0, 1, 1, 1])
    cm = confusion(y, pred)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y)
