"""
Maps ExperimentConfig's runtime-optimization knobs to concrete vLLM CLI
flags, with T4-specific caveats documented next to the knob they affect --
the lab repo's serve.py already established that these need calling out
explicitly rather than silently assumed to work.
"""
from src.config import ExperimentConfig

# vLLM on Turing (T4) has no FlashAttention-2 support -- forcing
# --attention-backend FLASH_ATTN is expected to either fall back or fail to
# start; that's an intentional negative result for the sweep to capture,
# not a bug to hide.
FLASH_ATTN_SUPPORTED_ON_T4 = False


def vllm_args(cfg: ExperimentConfig) -> list:
    """Batch size / context / KV-cache / attention-backend / speculative-
    decoding flags. Context length and GPU memory utilization are passed
    separately by serve.py since they're also used outside this list."""
    args = ["--max-num-seqs", str(cfg.max_num_seqs)]
    if cfg.enable_prefix_caching:
        args.append("--enable-prefix-caching")
    if cfg.attention_backend:
        args += ["--attention-backend", cfg.attention_backend]
    if cfg.speculative_model:
        args += ["--speculative-model", cfg.speculative_model,
                  "--num-speculative-tokens", str(cfg.num_speculative_tokens)]
    return args


def quantization_serve_flags(quant_method: str) -> list:
    """Flags needed at *serve* time (as opposed to quantize.py's *build*
    time). GPTQ/AWQ checkpoints produced by llmcompressor are self-
    describing (compressed-tensors format) and only need the format name.
    bitsandbytes quantizes the original FP16 checkpoint at load time rather
    than producing a separate artifact -- see the caveat in quantize.py.

    NOTE: unlike the GPTQ/AWQ paths (validated in this project's source
    notebooks), bnb serving via vLLM was NOT validated there -- expect this
    path may need debugging on a real T4 run.
    """
    if quant_method in ("gptq_w4a16", "gptq_w8a16", "awq_w4a16"):
        return ["--quantization", "compressed-tensors"]
    if quant_method in ("bnb_int8", "bnb_int4"):
        return ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
    return []  # fp16 baseline
