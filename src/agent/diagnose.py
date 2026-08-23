"""
Ask the model to name the likely bottleneck for the latest experiment
result, with a short rationale. The classification is advisory --
eval/run_v4_agent_optimizer.py separately checks it against a deterministic
oracle (src.agent.oracle) for accuracy scoring; nothing downstream trusts
this call's output as ground truth.
"""
import json
import re

DIAGNOSIS_SYSTEM = (
    "You are an LLM-serving performance engineer. Given one benchmark result "
    "and the run's constraints, name the single most likely bottleneck as "
    "exactly one of: memory-bound, compute-bound, latency-bound, quality-cliff, "
    "balanced. Respond ONLY with a JSON object: "
    '{"bottleneck": "...", "rationale": "one sentence"}'
)

METRIC_KEYS = ["quant_method", "vram_gb", "tokens_per_sec", "ttft_p95_ms", "e2e_p95_ms",
               "quality_recovery_pct", "cost_usd_per_request"]


def _metrics_only(row: dict) -> dict:
    return {k: row.get(k) for k in METRIC_KEYS}


def diagnose(model_client, row: dict, constraints_summary: dict) -> dict:
    user = (
        f"Result: {json.dumps(_metrics_only(row))}\n"
        f"Constraints: {json.dumps(constraints_summary)}"
    )
    res = model_client.generate(DIAGNOSIS_SYSTEM, user, temperature=0.3)
    return _parse(res.text)


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"bottleneck": "balanced", "rationale": "unparseable model output"}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"bottleneck": "balanced", "rationale": "unparseable model output"}
    data.setdefault("rationale", "")
    data.setdefault("bottleneck", "balanced")
    return data
