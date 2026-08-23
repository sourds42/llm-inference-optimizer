"""
Drives the review-item dataset (src/review_eval/dataset.py) through an
already-loaded model, in-process (no server), computing the same benchmark
fields as src/benchmark.py (TTFT/TPOT/tokens/VRAM/throughput/cost) over
real review generations instead of throwaway paragraph prompts -- plus the
review-specific quality fields from src/review_eval/metrics.py.

TTFT/TPOT use the same two-pass approximation as src/benchmark.py (a
separate max_new_tokens=1 pass) -- same documented tradeoff, done per-item
here rather than per-batch since review generation isn't batched (each
item needs its own full JSON response, not a shared padded batch).
Greedy decoding (do_sample=False), not the generic benchmark's sampling --
appropriate for a structured-output classification task where we want a
reproducible verdict, not creative variation.
"""
from __future__ import annotations
import time
from dataclasses import asdict

from src.review_eval.prompts import REVIEW_SYSTEM, build_review_prompt, parse_review
from src.review_eval.metrics import score_reviews
from src.benchmark import sample_gpu_stats
from src.cost import cost_usd_per_request


def review_one(model, tok, item: dict, cfg, max_new_tokens: int = 128) -> dict:
    import torch
    prompt = f"{REVIEW_SYSTEM}\n\n{build_review_prompt(item)}"
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=cfg.context_len).to(model.device)
    gen_kwargs = dict(do_sample=False, use_cache=cfg.use_cache, pad_token_id=tok.pad_token_id)

    with torch.no_grad():
        t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=1, **gen_kwargs)
        ttft = time.perf_counter() - t0

        t1 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
        e2e = ttft + (time.perf_counter() - t1)

    n_new = int(out.shape[1] - inputs["input_ids"].shape[1])
    tpot = (e2e - ttft) / max(max_new_tokens - 1, 1)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    prediction = parse_review(text)
    return {
        "item_id": item["id"], "ground_truth": item["ground_truth"], "prediction": prediction,
        "ttft_s": ttft, "e2e_s": e2e, "tpot_s": tpot, "n_tokens": n_new,
    }


def run_review_experiment(cfg, items: list) -> dict:
    """cfg: an ExperimentConfig (quant_method drives which model gets
    loaded). Returns a row with the same shape as src/experiment.py's
    rows, plus review-specific quality fields. quality_recovery_pct here
    is F1 score * 100 -- reusing the existing optimizer/agent code's
    constraint-gate field by naming convention, not a schema change."""
    from src.quantize import load_model

    print(f"[review_experiment] {cfg.id} ({cfg.quant_method}) over {len(items)} items")
    model = tok = None
    row = {"id": cfg.id, "tier": cfg.tier, "timestamp": time.time(),
           "quant_method": cfg.quant_method, "config": asdict(cfg), "error": None}
    try:
        model, tok, size_gb = load_model(cfg)
        row["size_gb"] = size_gb

        per_item = [review_one(model, tok, item, cfg) for item in items]

        ttfts = sorted(r["ttft_s"] for r in per_item)
        e2es = sorted(r["e2e_s"] for r in per_item)
        tpots = sorted(r["tpot_s"] for r in per_item)
        total_tokens = sum(r["n_tokens"] for r in per_item)
        wall = sum(r["e2e_s"] for r in per_item)
        pct = lambda xs, q: xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0.0

        row.update({
            "requests": len(items), "concurrency": 1, "wall_s": round(wall, 2),
            "tokens_per_sec": round(total_tokens / wall, 1) if wall > 0 else 0.0,
            "throughput_req_s": round(len(items) / wall, 3) if wall > 0 else 0.0,
            "ttft_p50_ms": round(pct(ttfts, .50) * 1000),
            "ttft_p95_ms": round(pct(ttfts, .95) * 1000),
            "tpot_p50_ms": round(pct(tpots, .50) * 1000, 2),
            "tpot_p95_ms": round(pct(tpots, .95) * 1000, 2),
            "e2e_p50_ms": round(pct(e2es, .50) * 1000),
            "e2e_p95_ms": round(pct(e2es, .95) * 1000),
            "e2e_p99_ms": round(pct(e2es, .99) * 1000) if e2es else 0,
        })
        row.update(sample_gpu_stats())

        quality = score_reviews(per_item)
        row.update(quality)
        row["quality_recovery_pct"] = round(quality["f1"] * 100, 1)
        row["cost_usd_per_request"] = cost_usd_per_request(row["throughput_req_s"], cfg.gpu_hourly_usd)
    except Exception as e:
        print(f"[review_experiment] {cfg.id} FAILED: {e}")
        row["error"] = str(e)
    finally:
        del model, tok
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    import json
    from pathlib import Path
    out_path = Path(cfg.results_dir) / "review_experiments.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")

    return row
