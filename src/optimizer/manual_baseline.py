"""
A simple, explicit heuristic standing in for "a human eyeballing the
results table" -- not a real user study. Documented here so the "manual vs
automated" comparison in the eval layer is honest about what "manual"
actually means.

Heuristic: among configs that clear the quality bar, a human skimming a
table tends to pick the cheapest-looking one without carefully cross-
checking tail latency or VRAM against the full constraint set -- so this
picks by lowest cost among quality-passing rows only, deliberately not
applying the P95/VRAM constraints an automated search would enforce.
"""
from src.optimizer.constraints import Constraints


def manual_pick(rows: list, constraints: Constraints):
    quality_ok = [r for r in rows if (r.get("quality_recovery_pct") or 0) >= constraints.quality_min_recovery_pct]
    if not quality_ok:
        return None
    return min(quality_ok, key=lambda r: r.get("cost_usd_per_request", float("inf")))
