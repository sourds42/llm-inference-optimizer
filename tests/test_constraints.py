import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.optimizer.constraints import Constraints, satisfies, filter_passing


def test_satisfies_all_thresholds():
    c = Constraints(quality_min_recovery_pct=95, p95_latency_ms_max=1000, vram_gb_max=10)
    good = {"quality_recovery_pct": 96, "e2e_p95_ms": 900, "vram_gb": 8}
    bad_quality = {"quality_recovery_pct": 90, "e2e_p95_ms": 900, "vram_gb": 8}
    bad_latency = {"quality_recovery_pct": 96, "e2e_p95_ms": 1500, "vram_gb": 8}
    assert satisfies(good, c)
    assert not satisfies(bad_quality, c)
    assert not satisfies(bad_latency, c)


def test_unset_thresholds_are_unconstrained():
    c = Constraints(quality_min_recovery_pct=95)
    row = {"quality_recovery_pct": 96, "e2e_p95_ms": 999999, "vram_gb": 999}
    assert satisfies(row, c)


def test_filter_passing():
    c = Constraints(quality_min_recovery_pct=95)
    rows = [{"quality_recovery_pct": 96}, {"quality_recovery_pct": 50}]
    assert filter_passing(rows, c) == [rows[0]]


def test_critical_recall_min():
    c = Constraints(quality_min_recovery_pct=0, critical_recall_min=95)
    good = {"quality_recovery_pct": 100, "critical_recall": 100}
    bad = {"quality_recovery_pct": 100, "critical_recall": 80}
    missing = {"quality_recovery_pct": 100}
    assert satisfies(good, c)
    assert not satisfies(bad, c)
    assert not satisfies(missing, c)


def test_critical_recall_min_unset_is_unconstrained():
    c = Constraints(quality_min_recovery_pct=0)
    row = {"quality_recovery_pct": 100}  # no critical_recall field at all
    assert satisfies(row, c)
