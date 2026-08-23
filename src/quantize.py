"""
Post-training quantization across the FP16 -> INT8/INT4 (bitsandbytes) ->
GPTQ -> AWQ ladder the spec asks for. GPTQ/AWQ recipes are ported from
LLM_capstone_quality_and_speed.ipynb's `build_recipe` (llmcompressor,
lm_head excluded from quantization to protect output quality); bnb
INT8/INT4 are added as a fast, no-calibration baseline distinct from
GPTQ/AWQ's calibrated methods, so the ladder is four genuinely different
techniques rather than GPTQ at two bit-widths counted twice.
"""
from __future__ import annotations
import gc
from pathlib import Path


def dir_size_gb(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / 1e9, 3)


def _build_recipe(quant_method: str):
    if quant_method == "gptq_w4a16":
        from llmcompressor.modifiers.quantization import GPTQModifier
        return GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])
    if quant_method == "gptq_w8a16":
        from llmcompressor.modifiers.quantization import GPTQModifier
        return GPTQModifier(scheme="W8A16", targets="Linear", ignore=["lm_head"])
    if quant_method == "awq_w4a16":
        from llmcompressor.modifiers.awq import AWQModifier
        return AWQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])
    raise ValueError(f"no llmcompressor recipe for {quant_method}")


def quantize(cfg) -> dict:
    """Quantize cfg.model_id per cfg.quant_method. Returns size info.
    Memoized: skips re-quantizing if the output dir already exists."""
    from huggingface_hub import snapshot_download
    base_dir = snapshot_download(cfg.model_id)
    base_gb = dir_size_gb(base_dir)

    if cfg.quant_method == "fp16":
        return {"quant_path": base_dir, "size_gb": base_gb, "reduction_pct": 0.0}

    if cfg.quant_method in ("bnb_int8", "bnb_int4"):
        # bitsandbytes quantizes at *load* time, not storage time -- vLLM
        # loads the original FP16 checkpoint and applies bnb quantization
        # when the server starts (see src/runtime_opts.py's
        # quantization_serve_flags). So there's no separate on-disk
        # artifact and no disk-size reduction to report for this method --
        # only the VRAM/quality/speed tradeoffs measured at serve time.
        return {"quant_path": base_dir, "size_gb": base_gb, "reduction_pct": 0.0}

    out_dir = cfg.quant_path or f"artifacts/{cfg.model_id.split('/')[-1]}-{cfg.quant_method}"
    if Path(out_dir).exists():
        print(f"[quantize] {out_dir} exists -- skipping.")
        quant_gb = dir_size_gb(out_dir)
        return {"quant_path": out_dir, "size_gb": quant_gb,
                "reduction_pct": round((1 - quant_gb / base_gb) * 100, 1)}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    try:
        from llmcompressor import oneshot           # newer API
    except Exception:
        from llmcompressor.transformers import oneshot  # older API

    print(f"[quantize] loading {cfg.model_id} ...")
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype="auto", device_map="cuda")

    print("[quantize] building calibration set (wikitext-2) ...")
    ds = (load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
          .filter(lambda r: len(r["text"]) > 0)
          .shuffle(seed=42).select(range(cfg.calib_samples)))
    ds = ds.map(lambda r: tok(r["text"], truncation=True, max_length=cfg.calib_maxlen))

    recipe = _build_recipe(cfg.quant_method)
    print(f"[quantize] running {cfg.quant_method} oneshot (a few minutes on T4) ...")
    oneshot(model=model, dataset=ds, recipe=recipe,
            max_seq_length=cfg.calib_maxlen,
            num_calibration_samples=cfg.calib_samples,
            output_dir=out_dir)

    del model
    gc.collect(); torch.cuda.empty_cache()

    quant_gb = dir_size_gb(out_dir)
    reduction = round((1 - quant_gb / base_gb) * 100, 1)
    print(f"[quantize] {base_gb} GB -> {quant_gb} GB  ({reduction}% smaller)")
    return {"quant_path": out_dir, "size_gb": quant_gb, "reduction_pct": reduction}
