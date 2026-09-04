from deeplob_replication.reference import PAPER_SETUP2


def test_paper_reference_values():
    assert PAPER_SETUP2[10]["f1"] == 0.8340
    assert PAPER_SETUP2[20]["f1"] == 0.7282
    assert PAPER_SETUP2[50]["f1"] == 0.8035


def test_published_recall_equals_accuracy_for_deeplob_rows():
    # Strong evidence that the paper table used support-weighted recall, not macro recall.
    for values in PAPER_SETUP2.values():
        assert values["recall"] == values["accuracy"]
