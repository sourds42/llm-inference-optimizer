"""
The deterministic ground truth: runs ONE fully-specified ExperimentConfig
end-to-end (quantize if needed -> serve -> benchmark -> evaluate -> cost)
and appends a flat result row to results/experiments.jsonl. Every other
layer (optimizer, agent) only ever reads this file -- nothing upstream
re-derives pass/fail itself.
"""
from __future__ import annotations
import json, gc, time
from dataclasses import asdict
from pathlib import Path

from src.quantize import quantize
from src.serve import VLLMServer
from src.benchmark import benchmark
from src.evaluate import accuracy_via_server, perplexity_offline
from src.cost import cost_usd_per_request


def _find_baseline_accuracy(model_id: str, results_dir: str):
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
        if row.get("quant_method") == "fp16" and row.get("config", {}).get("model_id") == model_id:
            best = row.get("quality_accuracy")
    return best


def run_experiment(cfg) -> dict:
    print(f"[experiment] {cfg.id} ({cfg.quant_method})")
    q = quantize(cfg)

    srv = VLLMServer(cfg, q["quant_path"], cfg.port, quant_method=cfg.quant_method,
                      served_name=cfg.served_name, results_dir=cfg.results_dir)
    try:
        srv.start().wait_ready()
        bench = benchmark(srv.base_url, cfg.bench_requests, cfg.bench_concurrency, cfg.bench_output_tokens)
        accuracy = accuracy_via_server(srv.base_url, cfg.eval_task, cfg.eval_limit,
                                        tag=cfg.id, results_dir=cfg.results_dir)
    finally:
        srv.stop()

    try:
        perplexity = perplexity_offline(q["quant_path"])
    except Exception as e:
        print("[experiment] perplexity skipped:", e)
        perplexity = None

    if cfg.quant_method == "fp16":
        recovery_pct = 100.0
    else:
        baseline_acc = _find_baseline_accuracy(cfg.model_id, cfg.results_dir)
        recovery_pct = round(accuracy / baseline_acc * 100, 1) if (accuracy and baseline_acc) else None

    throughput_req_s = bench.get("throughput_req_s", 0.0)
    row = {
        "id": cfg.id, "tier": cfg.tier, "timestamp": time.time(),
        "quant_method": cfg.quant_method, "config": asdict(cfg),
        "size_gb": q["size_gb"], "reduction_pct": q.get("reduction_pct"),
        **bench,
        "quality_accuracy": accuracy, "quality_perplexity": perplexity,
        "quality_recovery_pct": recovery_pct,
        "cost_usd_per_request": cost_usd_per_request(throughput_req_s, cfg.gpu_hourly_usd),
    }

    out_path = Path(cfg.results_dir) / "experiments.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")

    gc.collect()
    return row
