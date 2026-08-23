import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.optimizer.pareto import dominates, pareto_frontier


def test_dominates_strictly_better_on_one_equal_on_rest():
    a = {"e2e_p95_ms": 100, "vram_gb": 5, "cost_usd_per_request": 0.001, "quality_recovery_pct": 99}
    b = {"e2e_p95_ms": 150, "vram_gb": 5, "cost_usd_per_request": 0.001, "quality_recovery_pct": 99}
    assert dominates(a, b)
    assert not dominates(b, a)


def test_no_dominance_on_tradeoff():
    a = {"e2e_p95_ms": 100, "vram_gb": 10, "cost_usd_per_request": 0.002, "quality_recovery_pct": 90}
    b = {"e2e_p95_ms": 200, "vram_gb": 5, "cost_usd_per_request": 0.001, "quality_recovery_pct": 99}
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_pareto_frontier_drops_dominated_points():
    dominated = {"e2e_p95_ms": 150, "vram_gb": 5, "cost_usd_per_request": 0.002, "quality_recovery_pct": 95}
    dominator = {"e2e_p95_ms": 100, "vram_gb": 5, "cost_usd_per_request": 0.001, "quality_recovery_pct": 97}
    tradeoff = {"e2e_p95_ms": 50, "vram_gb": 12, "cost_usd_per_request": 0.001, "quality_recovery_pct": 90}
    frontier = pareto_frontier([dominated, dominator, tradeoff])
    assert dominated not in frontier
    assert dominator in frontier
    assert tradeoff in frontier
