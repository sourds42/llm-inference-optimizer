"""
The deterministic ground truth: runs ONE fully-specified ExperimentConfig
end-to-end (load/quantize -> benchmark -> perplexity -> cost) in-process
and appends a flat result row to results/experiments.jsonl. Every other
layer (optimizer, agent) only ever reads this file.

Failures are recorded, not fatal -- some configs (e.g.
attn_implementation="flash_attention_2" on a T4) are EXPECTED to fail; the
sweep captures that as an honest negative result (row["error"] set, numeric
fields left absent) and moves on to the next config rather than crashing
the whole run.
"""
from __future__ import annotations
import json, gc, time
from dataclasses import asdict
from pathlib import Path

from src.quantize import load_model
from src.benchmark import benchmark
from src.evaluate import perplexity
from src.cost import cost_usd_per_request


def _find_baseline_perplexity(model_id: str, results_dir: str):
    """Scans experiments.jsonl for a prior fp16 row on the same model_id,
    so V2/V3/V4 runs (separate process invocations from V1) still compute
    recovery_pct correctly without relying on in-memory state."""
    path = Path(results_dir) / "experiments.jsonl"
    if not path.exists():
        return None
    best = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("quant_method") == "fp16"
                and row.get("config", {}).get("model_id") == model_id
                and row.get("quality_perplexity")):
            best = row["quality_perplexity"]
    return best


def run_experiment(cfg) -> dict:
    print(f"[experiment] {cfg.id} ({cfg.quant_method}, attn={cfg.attn_implementation})")
    row = {"id": cfg.id, "tier": cfg.tier, "timestamp": time.time(),
           "quant_method": cfg.quant_method, "config": asdict(cfg), "error": None}
    model = tok = None
    try:
        model, tok, size_gb = load_model(cfg)
        row["size_gb"] = size_gb

        bench = benchmark(model, tok, cfg)
        row.update(bench)

        ppl = perplexity(model, tok, cfg.eval_max_tokens, cfg.eval_stride)
        row["quality_perplexity"] = ppl

        if cfg.quant_method == "fp16":
            recovery_pct = 100.0
        else:
            baseline_ppl = _find_baseline_perplexity(cfg.model_id, cfg.results_dir)
            recovery_pct = round(baseline_ppl / ppl * 100, 1) if (baseline_ppl and ppl) else None
        row["quality_recovery_pct"] = recovery_pct

        row["cost_usd_per_request"] = cost_usd_per_request(
            bench.get("throughput_req_s", 0.0), cfg.gpu_hourly_usd)
    except Exception as e:
        print(f"[experiment] {cfg.id} FAILED: {e}")
        row["error"] = str(e)
    finally:
        del model, tok
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_path = Path(cfg.results_dir) / "experiments.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row
