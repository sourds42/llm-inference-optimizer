"""
Lightweight quantization: FP16 baseline vs. bitsandbytes INT8/INT4. No
calibration dataset, no separate build step -- bitsandbytes quantizes at
model-load time, so "quantizing" here just means loading with a different
BitsAndBytesConfig. This trades away GPTQ/AWQ (which need llmcompressor and
a calibration pass, and vLLM to serve) for a setup that installs and runs
in a couple of minutes instead of ten-plus -- an explicit simplicity
tradeoff, not an oversight.
"""
from __future__ import annotations
from pathlib import Path


def dir_size_gb(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / 1e9, 3)


def load_model(cfg):
    """Loads cfg.model_id with cfg.quant_method applied in-process.
    Returns (model, tokenizer, size_gb). size_gb is the on-disk FP16 size
    for every method -- bitsandbytes quantizes at load time, not storage
    time, so there's no separate on-disk artifact or disk-size reduction to
    report; only load-time VRAM changes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download

    base_dir = snapshot_download(cfg.model_id)
    size_gb = dir_size_gb(base_dir)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right-padding (the default) silently produces wrong generations on a
    # decoder-only model when batch_size > 1 -- padded prompts shift the
    # positions the model actually attends to for the next token. Left
    # padding keeps every sequence's real content ending at the same index.
    tok.padding_side = "left"

    kwargs = {
        "device_map": "cuda",
        "attn_implementation": cfg.attn_implementation,  # "flash_attention_2" is expected to fail on T4 (Turing)
    }
    if cfg.quant_method in ("bnb_int8", "bnb_int4"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True) if cfg.quant_method == "bnb_int8"
            else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        )
    else:
        kwargs["torch_dtype"] = getattr(torch, cfg.dtype, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    return model, tok, size_gb
