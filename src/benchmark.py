"""
Streaming async load-test, extended from llm-inference-lab/src/benchmark.py
to also report TPOT (mean inter-token latency after the first token) and a
VRAM/GPU-utilization sample taken right after the load -- vLLM's own
/metrics doesn't expose instantaneous VRAM, and torch isn't necessarily
importable in the same process as the benchmark client.
"""
from __future__ import annotations
import asyncio, subprocess, time


async def _one(aclient, prompt, out_tokens):
    t0 = time.perf_counter()
    ttft = None
    token_times = []
    stream = await aclient.completions.create(
        model="model", prompt=prompt, max_tokens=out_tokens, temperature=0.7, stream=True)
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].text:
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            token_times.append(now)
    e2e = time.perf_counter() - t0
    tpot = ((token_times[-1] - token_times[0]) / (len(token_times) - 1)) if len(token_times) > 1 else 0.0
    return ttft or 0.0, e2e, len(token_times), tpot


async def _run(base_url, n_reqs, conc, out_tokens):
    from openai import AsyncOpenAI
    aclient = AsyncOpenAI(base_url=base_url, api_key="x")
    # distinct prompts so we don't inflate prefix-cache hits
    prompts = [f"Write {out_tokens} tokens explaining ML-serving idea number {i}:" for i in range(n_reqs)]
    sem = asyncio.Semaphore(conc)
    rows = []

    async def worker(p):
        async with sem:
            rows.append(await _one(aclient, p, out_tokens))

    wall0 = time.perf_counter()
    await asyncio.gather(*(worker(p) for p in prompts))
    wall = time.perf_counter() - wall0

    ttfts = sorted(r[0] for r in rows)
    e2es = sorted(r[1] for r in rows)
    tpots = sorted(r[3] for r in rows if r[3] > 0)
    toks = sum(r[2] for r in rows)
    pct = lambda xs, q: xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0.0
    return {
        "requests": n_reqs, "concurrency": conc, "wall_s": round(wall, 2),
        "tokens_per_sec": round(toks / wall, 1),
        "throughput_req_s": round(n_reqs / wall, 3),
        "ttft_p50_ms": round(pct(ttfts, .50) * 1000),
        "ttft_p95_ms": round(pct(ttfts, .95) * 1000),
        "tpot_p50_ms": round(pct(tpots, .50) * 1000, 2),
        "tpot_p95_ms": round(pct(tpots, .95) * 1000, 2),
        "e2e_p50_ms": round(pct(e2es, .50) * 1000),
        "e2e_p95_ms": round(pct(e2es, .95) * 1000),
        "e2e_p99_ms": round(pct(e2es, .99) * 1000),
    }


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


def benchmark(base_url, requests=40, concurrency=8, output_tokens=64) -> dict:
    """Synchronous entry point. GPU stats are sampled immediately after the
    load test so the reading reflects steady-state usage, not an idle server."""
    result = asyncio.run(_run(base_url, requests, concurrency, output_tokens))
    result.update(sample_gpu_stats())
    return result
