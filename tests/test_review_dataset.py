import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local dev layout quirk: agent-review-loop's git root on this machine is
# nested one level deeper than the sibling assumption dataset.py's default
# candidate expects -- a fresh `git clone` (e.g. in Colab) produces the
# flat sibling layout it already handles, so this override is test-only.
_LOCAL_AGENT_REVIEW_LOOP = Path(__file__).resolve().parents[2] / "agent-review-loop" / "agent-review-loop"
if (_LOCAL_AGENT_REVIEW_LOOP / "eval" / "golden_set.py").exists():
    os.environ.setdefault("AGENT_REVIEW_LOOP_PATH", str(_LOCAL_AGENT_REVIEW_LOOP))

from src.review_eval.dataset import build_review_items, summarize, CATEGORIES, SEVERITIES


def test_build_review_items_shape_and_uniqueness():
    items = build_review_items()
    assert len(items) > 30
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    for item in items:
        gt = item["ground_truth"]
        assert gt["category"] in CATEGORIES
        assert gt["severity"] in SEVERITIES
        assert isinstance(gt["has_bug"], bool)
        assert item["spec"] and item["code"]


def test_security_and_performance_categories_present():
    items = build_review_items()
    categories = {i["ground_truth"]["category"] for i in items if i["ground_truth"]["has_bug"]}
    assert "security" in categories
    assert "performance" in categories
    assert "logic" in categories


def test_summarize_counts_match():
    items = build_review_items()
    s = summarize(items)
    assert s["n_items"] == len(items)
    assert s["n_buggy"] + s["n_clean"] == len(items)
