"""
Deterministic constraint filtering -- the only thing allowed to decide
whether a config "passes." Nothing upstream (including the V4 agent) gets
to override this; it only ever reads the outcome.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class Constraints:
    quality_min_recovery_pct: float = 95.0
    p95_latency_ms_max: Optional[float] = None
    vram_gb_max: Optional[float] = None
    cost_usd_per_request_max: Optional[float] = None
    # Optional, unconstrained by default -- used by src/review_eval's
    # code-review evaluation, where a row can also carry a `critical_recall`
    # field (recall restricted to critical-severity ground-truth items).
    critical_recall_min: Optional[float] = None

    @staticmethod
    def load(path: str) -> "Constraints":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return Constraints(**data)


def satisfies(row: dict, constraints: Constraints) -> bool:
    recovery = row.get("quality_recovery_pct")
    if recovery is not None and recovery < constraints.quality_min_recovery_pct:
        return False
    if constraints.p95_latency_ms_max is not None:
        p95 = row.get("e2e_p95_ms")
        if p95 is None or p95 > constraints.p95_latency_ms_max:
            return False
    if constraints.vram_gb_max is not None:
        vram = row.get("vram_gb")
        if vram is None or vram > constraints.vram_gb_max:
            return False
    if constraints.cost_usd_per_request_max is not None:
        cost = row.get("cost_usd_per_request")
        if cost is None or cost > constraints.cost_usd_per_request_max:
            return False
    if constraints.critical_recall_min is not None:
        critical_recall = row.get("critical_recall")
        if critical_recall is None or critical_recall < constraints.critical_recall_min:
            return False
    return True


def filter_passing(rows, constraints: Constraints) -> list:
    return [r for r in rows if satisfies(r, constraints)]
