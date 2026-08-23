"""
In-process benchmark: generate directly with the already-loaded HF model --
no server, no separate client process, no vLLM. Deliberately simple:
keeping the Colab install and the benchmark harness itself lightweight was
an explicit priority over exactly matching a production serving stack's
percentile methodology.

TTFT is approximated with a separate max_new_tokens=1 pass rather than true
token-by-token streaming (transformers' TextIteratorStreamer doesn't
cleanly support batch_size > 1) -- documented here rather than silently
assumed exact. TPOT is the remaining generation time divided evenly across
the rest of the batch's tokens, not each token's individually measured gap.
"""
from __future__ import annotations
import time
import subprocess

_assistant_models = {}


def _load_assistant(model_id, dtype):
    if model_id not in _assistant_models:
        from transformers import AutoModelForCausalLM
        _assistant_models[model_id] = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="cuda")
    return _assistant_models[model_id]


def _run_batch(model, tok, prompts, cfg):
    import torch
    inputs = tok(prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=cfg.context_len).to(model.device)
    gen_kwargs = dict(do_sample=True, temperature=0.7, use_cache=cfg.use_cache,
                       pad_token_id=tok.pad_token_id)
    if cfg.assistant_model_id:
        gen_kwargs["assistant_model"] = _load_assistant(cfg.assistant_model_id, model.dtype)

    with torch.no_grad():
        t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=1, **gen_kwargs)
        ttft = time.perf_counter() - t0

        t1 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, **gen_kwargs)
        e2e = ttft + (time.perf_counter() - t1)

    n_generated = int((out.shape[1] - inputs["input_ids"].shape[1]) * out.shape[0])
    tpot = (e2e - ttft) / max(cfg.max_new_tokens - 1, 1)
    return ttft, e2e, n_generated, tpot


def sample_gpu_stats() -> dict:
    """Point-in-time VRAM/utilization sample via nvidia-smi. Returns zeros
    (not an error) if nvidia-smi isn't available, so a benchmark run never
    fails just because GPU telemetry is missing."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        mem_mb, util_pct = out.split(",")
        return {"vram_gb": round(float(mem_mb) / 1024, 2), "gpu_util_pct": float(util_pct)}
    except Exception:
        return {"vram_gb": 0.0, "gpu_util_pct": 0.0}


def benchmark(model, tok, cfg) -> dict:
    """Runs cfg.bench_requests prompts in batches of cfg.batch_size,
    entirely in-process against the already-loaded model."""
    n_batches = max(1, cfg.bench_requests // cfg.batch_size)

    ttfts, e2es, tpots, total_tokens = [], [], [], 0
    wall0 = time.perf_counter()
    for b in range(n_batches):
        prompts = [f"Write a short paragraph about topic number {b * cfg.batch_size + i}:"
                   for i in range(cfg.batch_size)]
        ttft, e2e, n_gen, tpot = _run_batch(model, tok, prompts, cfg)
        ttfts.append(ttft); e2es.append(e2e); tpots.append(tpot); total_tokens += n_gen
    wall = time.perf_counter() - wall0

    pct = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(len(xs) * q))] if xs else 0.0
    n_reqs = n_batches * cfg.batch_size
    result = {
        "requests": n_reqs, "concurrency": cfg.batch_size, "wall_s": round(wall, 2),
        "tokens_per_sec": round(total_tokens / wall, 1) if wall > 0 else 0.0,
        "throughput_req_s": round(n_reqs / wall, 3) if wall > 0 else 0.0,
        "ttft_p50_ms": round(pct(ttfts, .50) * 1000),
        "ttft_p95_ms": round(pct(ttfts, .95) * 1000),
        "tpot_p50_ms": round(pct(tpots, .50) * 1000, 2),
        "tpot_p95_ms": round(pct(tpots, .95) * 1000, 2),
        "e2e_p50_ms": round(pct(e2es, .50) * 1000),
        "e2e_p95_ms": round(pct(e2es, .95) * 1000),
        "e2e_p99_ms": round(pct(e2es, .99) * 1000) if e2es else 0,
    }
    result.update(sample_gpu_stats())
    return result
