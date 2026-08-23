"""
Deterministic ground-truth bottleneck classifier, used ONLY by
eval/run_v4_agent_optimizer.py to score the agent's diagnosis accuracy
after the fact. The agent itself never calls this -- it only sees the raw
numbers via src.agent.tools, the same view a human would have.
"""
from __future__ import annotations


def classify_bottleneck(row: dict, constraints, all_rows: list) -> str:
    recovery = row.get("quality_recovery_pct")
    if recovery is not None and recovery < constraints.quality_min_recovery_pct:
        return "quality-cliff"

    if constraints.vram_gb_max:
        vram_ratio = (row.get("vram_gb") or 0) / constraints.vram_gb_max
        if vram_ratio >= 0.9:
            return "memory-bound"

    tps_values = sorted(r.get("tokens_per_sec") or 0 for r in all_rows if r.get("tokens_per_sec"))
    if tps_values:
        median_tps = tps_values[len(tps_values) // 2]
        if median_tps > 0 and (row.get("tokens_per_sec") or 0) < median_tps * 0.7:
            return "compute-bound"

    p95_values = sorted(r.get("e2e_p95_ms") or 0 for r in all_rows if r.get("e2e_p95_ms"))
    if p95_values:
        median_p95 = p95_values[len(p95_values) // 2]
        if median_p95 > 0 and (row.get("e2e_p95_ms") or 0) > median_p95 * 1.3:
            return "latency-bound"

    return "balanced"
