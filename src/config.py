"""
A single ExperimentConfig fully describes one point in the search space --
the same "one YAML fully describes a run" idea as llm-inference-lab's
Config, extended with the runtime-optimization knobs (batch size, context
length, KV-cache/prefix caching, attention backend, speculative decoding)
and quantization method the new search space sweeps over.
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
    quant_method: str = "fp16"                     # fp16 | bnb_int8 | bnb_int4 | gptq_w4a16 | gptq_w8a16 | awq_w4a16
    quant_path: str = ""                           # filled in by quantize() if left empty

    # serving (T4-friendly defaults; Turing has no native bf16/FP8)
    dtype: str = "float16"
    max_model_len: int = 2048
    max_num_seqs: int = 8                          # batch-size knob
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True
    attention_backend: Optional[str] = None        # None = vLLM default; "FLASH_ATTN" to force-try
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 0
    port: int = 8000
    served_name: str = "model"

    # quantization calibration (GPTQ/AWQ)
    calib_samples: int = 256
    calib_maxlen: int = 512

    # benchmark
    bench_requests: int = 40
    bench_concurrency: int = 8
    bench_output_tokens: int = 64

    # evaluation
    eval_task: str = "hellaswag"
    eval_limit: int = 100

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
