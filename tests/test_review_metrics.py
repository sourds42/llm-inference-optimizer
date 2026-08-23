import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.review_eval.metrics import score_reviews


def _item(has_bug, pred_bug, severity=None, pred_severity=None, category=None, pred_category=None):
    return {
        "ground_truth": {"has_bug": has_bug, "severity": severity, "category": category},
        "prediction": {"has_bug": pred_bug, "severity": pred_severity, "category": pred_category},
    }


def test_perfect_predictions():
    items = [
        _item(True, True, "critical", "critical", "security", "security"),
        _item(False, False),
    ]
    m = score_reviews(items)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["severity_accuracy"] == 1.0
    assert m["category_accuracy"] == 1.0
    assert m["critical_recall"] == 1.0
    assert m["completeness"] == 1.0


def test_missed_bug_hurts_recall_not_precision():
    items = [
        _item(True, False, "major", None, "logic", None),  # false negative
        _item(False, False),
    ]
    m = score_reviews(items)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0  # no true positives at all -> defined as 0
    assert m["fn"] == 1


def test_false_positive_hurts_precision():
    items = [
        _item(False, True),  # false positive
        _item(True, True, "major", "major", "logic", "logic"),
    ]
    m = score_reviews(items)
    assert m["fp"] == 1
    assert m["precision"] == 0.5


def test_unparseable_prediction_excluded_and_lowers_completeness():
    items = [
        _item(True, True, "major", "major", "logic", "logic"),
        {"ground_truth": {"has_bug": True, "severity": "critical", "category": "security"},
         "prediction": {"has_bug": None, "severity": None, "category": None}},
    ]
    m = score_reviews(items)
    assert m["completeness"] == 0.5
    assert m["n_well_formed"] == 1


def test_critical_recall_only_over_critical_ground_truth():
    items = [
        _item(True, True, "critical", "critical", "security", "security"),
        _item(True, False, "minor", None, "performance", None),  # missed, but not critical
    ]
    m = score_reviews(items)
    assert m["critical_recall"] == 1.0  # the only critical item was caught


def test_empty_input():
    m = score_reviews([])
    assert m["completeness"] == 0.0
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
