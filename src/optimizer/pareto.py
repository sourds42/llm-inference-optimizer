"""
Multi-objective Pareto frontier over experiment results. A point is on the
frontier if no other point dominates it (at least as good on every
objective and strictly better on at least one).
"""

OBJECTIVES = {
    "e2e_p95_ms": "min",
    "vram_gb": "min",
    "cost_usd_per_request": "min",
    "quality_recovery_pct": "max",
}


def _better_or_equal(a, b, direction):
    if a is None or b is None:
        return False
    return a <= b if direction == "min" else a >= b


def _strictly_better(a, b, direction):
    if a is None or b is None:
        return False
    return a < b if direction == "min" else a > b


def dominates(row_a: dict, row_b: dict, objectives: dict = OBJECTIVES) -> bool:
    at_least_as_good_on_all = all(
        _better_or_equal(row_a.get(k), row_b.get(k), d) for k, d in objectives.items()
    )
    strictly_better_on_one = any(
        _strictly_better(row_a.get(k), row_b.get(k), d) for k, d in objectives.items()
    )
    return at_least_as_good_on_all and strictly_better_on_one


def pareto_frontier(rows, objectives: dict = OBJECTIVES) -> list:
    rows = list(rows)
    frontier = []
    for candidate in rows:
        if any(dominates(other, candidate, objectives) for other in rows if other is not candidate):
            continue
        frontier.append(candidate)
    return frontier
