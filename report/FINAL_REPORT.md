# LLM Inference Optimization -- Final Report

*Template -- fill in each section after running `notebooks/run_on_colab.ipynb`
end to end. Keep the honest-limitations sections; don't delete them once
real numbers land.*

## 1. Problem & targets

- Target model: Qwen/Qwen3-0.6B, Colab T4 (float16, no bf16/FP8)
- Constraints (`configs/constraints.yaml`): quality ≥ 95% of FP16 baseline
  accuracy, P95 latency ≤ `__` ms, VRAM ≤ `__` GB
- What "quality" means here: HellaSwag accuracy via the live vLLM server,
  expressed as % recovered relative to the FP16 baseline (not an absolute
  score) -- perplexity tracked alongside as a secondary signal.

## 2. V1 → V2: baseline and the quantization/runtime-opt ladder

*Fill in after `eval/run_v2_quant_sweep.py`.*

- Full one-variable-at-a-time results table (pull from `results/experiments.jsonl`)
- Which quantization method won on which axis (size vs. quality vs. speed)
- Runtime-opt findings: batch size / context length / prefix caching /
  FlashAttention / speculative decoding -- what helped, what didn't, what
  failed outright (FlashAttention on T4 is *expected* to fail or fall back;
  report that as a real, documented result)
- Anything that needed debugging beyond what the source notebooks had
  already solved (bnb serving was not validated pre-Colab-run -- document
  what broke and how it was fixed, or that it wasn't)

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

- Configs that failed outright, and why (T4 kernel limits, VRAM OOM,
  unvalidated bnb path, etc. -- pull from `results/vllm_*.log`)
- Where the cost model (`src/cost.py`) is illustrative rather than a real
  quote -- don't let a reader mistake $/request here for a production number
- Subprocess isolation (`src/serve.py`) is not a real sandbox -- no
  `Docker --network=none`/seccomp -- same caveat the sibling
  `agent-review-loop` repo flags for its own tool execution
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
