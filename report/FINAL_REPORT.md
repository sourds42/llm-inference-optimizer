# LLM Inference Optimization -- Final Report

*Template -- fill in each section after running `notebooks/run_on_colab.ipynb`
end to end. Keep the honest-limitations sections; don't delete them once
real numbers land.*

## 1. Problem & targets

- Target model: Qwen/Qwen3-0.6B, Colab T4 (float16, no bf16/FP8)
- Constraints (`configs/constraints.yaml`): quality ≥ 95% of FP16 baseline
  perplexity-recovery, P95 latency ≤ `__` ms, VRAM ≤ `__` GB
- What "quality" means here: sliding-window perplexity computed in-process
  (no lm_eval), expressed as % recovered relative to the FP16 baseline's
  perplexity (not an absolute accuracy score) -- see "what this lightweight
  stack trades away" in the README for why HellaSwag-style accuracy isn't
  part of this run.

## 2. V1 → V2: baseline and the quantization/runtime-opt ladder

*Fill in after `eval/run_v2_quant_sweep.py`.*

- Full one-variable-at-a-time results table (pull from `results/experiments.jsonl`)
- bnb INT8 vs. INT4 vs. FP16 -- size/VRAM at load, perplexity, tokens/sec
- Runtime-opt findings: batch size / context length / KV-cache /
  attention implementation -- what helped, what didn't, what failed
  outright (`attn_implementation=flash_attention_2` on T4 is *expected* to
  fail; report that as a real, documented result, not something to omit)
- Any config that produced an `error` row -- what it was and whether it was
  the expected FlashAttention failure or something else worth debugging

## 3. V3: deterministic optimizer vs. manual selection

*Fill in after `eval/run_v3_auto_optimizer.py`.*

- `results/v3_optimizer.json`: how many configs passed constraints, how many
  landed on the Pareto frontier, which one the automated search picked vs.
  the manual heuristic (`src/optimizer/manual_baseline.py`) -- and by how
  much the automated pick actually differs on the objectives that matter
  (P95 latency, VRAM, cost) once you look past "quality passes."
- Where the manual heuristic's blind spot (ignoring P95/VRAM, picking on
  cost alone) actually cost something concrete, if it did.

## 4. V4: agent-assisted optimization

*Fill in after `eval/run_v4_agent_optimizer.py`, using
`results/v4_agent_optimizer.json`.*

- Diagnosis accuracy vs. the deterministic oracle (`src/agent/oracle.py`)
- Rounds / new experiments needed to reach a constraint-satisfying config,
  vs. V3's full sweep size
- "Unnecessary" experiments (proposed configs that didn't end up on the
  final Pareto frontier)
- Did the agent's best config beat, match, or lose to V3's automated pick?
  By how much, on which objective?
- Model backend actually used for this run (`MockClient` is free/default
  and deterministic but doesn't reason -- note explicitly if the numbers
  above are from the mock or from a real `MODEL_BACKEND=anthropic`/`ollama` run)

## 5. Failure analysis & honest limitations

- Configs that failed outright, and why (pull `row["error"]` from
  `results/experiments.jsonl` -- the FlashAttention-on-T4 failure is
  expected; anything else is worth a closer look)
- Where the cost model (`src/cost.py`) is illustrative rather than a real
  quote -- don't let a reader mistake $/request here for a production number
- TTFT is approximated (separate 1-token pass, not true streaming) and TPOT
  is an even split of remaining generation time, not per-token measured
  gaps -- see `src/benchmark.py`'s docstring; don't overstate precision here
- Single-run measurements: no repeated-trial variance reported unless you
  added it -- note if a number could be noise rather than signal

## 6. This is a prototype, not a production deployment

Explicitly: this system demonstrates the *methodology* (controlled
quantization ladder, deterministic constraint/Pareto search, agent-assisted
diagnosis under a hard tool gate) end-to-end on one small model and one free
GPU tier. It is not load-tested at production traffic, not multi-GPU, and
the cost model is illustrative. What transfers directly to a real production
setting is the architecture and the discipline (deterministic gate before
LLM judgment) -- not these specific numbers.
