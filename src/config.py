"""
A single ExperimentConfig fully describes one point in the search space.
Lightweight-stack version: no vLLM server, no llmcompressor calibration --
bitsandbytes quantizes at model-load time, and runtime-opt knobs map onto
plain `transformers.generate()` kwargs instead of vLLM CLI flags.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import yaml


@dataclass
class ExperimentConfig:
    # identity
    id: str = "baseline"
    tier: str = "V1"                              # V1 | V2 | V3 | V4, for report grouping

    # model + quantization
    model_id: str = "Qwen/Qwen3-0.6B"
    quant_method: str = "fp16"                     # fp16 | bnb_int8 | bnb_int4

    # generation (T4-friendly defaults; Turing has no native bf16/FP8)
    dtype: str = "float16"
    batch_size: int = 4                            # batch-size tuning
    max_new_tokens: int = 64
    context_len: int = 512                         # context-length optimization (prompt truncation)
    use_cache: bool = True                         # KV-cache on/off
    attn_implementation: str = "sdpa"               # sdpa | eager | flash_attention_2 (expected to fail on T4)
    assistant_model_id: Optional[str] = None        # speculative decoding draft model, best-effort/optional

    # benchmark
    bench_requests: int = 20                        # total prompts, run in batches of batch_size

    # evaluation (perplexity only in this lightweight path)
    eval_max_tokens: int = 2048
    eval_stride: int = 512

    # cost -- illustrative $/GPU-hr, check current pricing before treating as real
    gpu_hourly_usd: float = 0.35

    # io
    results_dir: str = "results"

    @staticmethod
    def load(path: Optional[str] = None) -> "ExperimentConfig":
        data = {}
        if path:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        return ExperimentConfig(**data)

    def to_dict(self) -> dict:
        return asdict(self)
