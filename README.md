# LLM Inference Optimization System

A production-oriented (prototype, not production-deployed) system that
determines the best model/serving configuration under constraints for
**quality, latency, throughput, GPU memory, and cost** -- and layers an LLM
agent on top that reasons about *which experiment to try next*, while a
deterministic tool stays the only thing allowed to say whether a config
actually passes.

Built as a V1 -> V2 -> V3 -> V4 story:

```
V1  FP16 baseline  ──▶  V2  quantization + runtime-opt sweep  ──▶  V3  deterministic optimizer  ──▶  V4  agent-assisted optimizer
    Qwen3-0.6B, T4        one variable at a time:                    constraint filter                diagnose → propose → experiment
                          bnb INT8/INT4, GPTQ, AWQ,                  + Pareto frontier                loop, capped + patience-stopped
                          batch size, context length,                + manual-selection                (src/agent/graph.py)
                          KV-cache, FlashAttention,                  comparison
                          speculative decoding                       (src/optimizer/)
```

## Why an agent here, and why so little of it

The agent (`src/agent/`) only ever *diagnoses* the latest result and
*proposes* the next experiment to run. It never decides whether a config
passes -- `src/optimizer/constraints.py` and `src/optimizer/pareto.py` are
the only code paths allowed to do that, and `src/experiment.py` is the only
code path allowed to produce a measurement. This mirrors the sibling
[agent-review-loop](https://github.com/sourds42/agent-review-loop) repo's
"tool gate before reviewer" principle, applied here to optimization instead
of code generation: put the LLM where reasoning genuinely helps (bottleneck
diagnosis, next-experiment selection under partial information), and keep
everything measurable deterministic.

## Repo layout

```
configs/
  search_space.yaml     one base config + axes swept one-at-a-time (V2)
  constraints.yaml       quality/latency/VRAM/cost targets (V3/V4)
src/
  config.py              ExperimentConfig -- one YAML fully describes a run
  quantize.py             FP16 → bnb INT8/INT4 → GPTQ-W4A16/W8A16 → AWQ-W4A16
  serve.py                vLLM server lifecycle (subprocess, fail-fast readiness, VRAM-release sleep)
  runtime_opts.py         batch size / context / KV-cache / FlashAttention / speculative decoding → vLLM flags
  benchmark.py            TTFT, TPOT, tokens/sec, throughput, E2E p50/p95/p99, VRAM/GPU-util
  evaluate.py             HellaSwag accuracy (via server) + sliding-window perplexity
  cost.py                 $/request from a documented $/GPU-hr rate
  experiment.py           THE deterministic ground truth -- runs one config end-to-end, appends to results/experiments.jsonl
  optimizer/               V3: search_space, constraints, pareto, search (replay + active), manual_baseline
  agent/                   V4: tools (read-only), diagnose, propose, graph (LangGraph loop), oracle (eval-only ground truth)
  model_client.py         MockClient (free/default) / OllamaClient / AnthropicClient -- pluggable V4 backend
  report.py               results/experiments.jsonl → comparison.json + charts
eval/
  run_v1_baseline.py · run_v2_quant_sweep.py · run_v3_auto_optimizer.py · run_v4_agent_optimizer.py · compare_v1_to_v4.py
notebooks/run_on_colab.ipynb   single T4 entrypoint, all four stages
tests/                    deterministic only -- no GPU needed, run right now with `pytest tests/`
report/FINAL_REPORT.md    case study writeup (fill in after a real Colab run)
```

## Quickstart

**Colab (T4, does the real GPU work):**
Open `notebooks/run_on_colab.ipynb` in Colab, `Runtime → T4 GPU`, run cells
top to bottom. Each V is a separate cell/script -- V3 and the final report
don't need the GPU at all, they just read whatever `results/experiments.jsonl`
has so far.

**Local (deterministic layers only, no GPU required):**
```bash
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
pytest tests/                          # 13 tests: config, constraints, pareto, cost, agent graph smoke
python -m eval.run_v3_auto_optimizer   # ranks whatever results/experiments.jsonl already has
python -m eval.run_v4_agent_optimizer  # MockClient by default; MODEL_BACKEND=ollama|anthropic for a real run
```

## What's real vs. what needs a Colab run to prove

Everything under `src/optimizer/`, `src/agent/`, `src/config.py`, and
`src/cost.py` is deterministic, unit-tested, and runs on this machine right
now (no GPU here to test against). `src/quantize.py` / `serve.py` /
`benchmark.py` / `evaluate.py` are careful ports of code already validated
on a T4 in this project's source notebooks (`LLM_capstone_quality_and_speed.ipynb`
and `llm-inference-lab`), extended for the fuller quantization ladder and
runtime-opt sweep -- but the extensions (bnb INT8/INT4 serving in
particular, and forcing FlashAttention on Turing) were **not** validated
there and are expected to need debugging on a real run. Results tables below
are placeholders until that run happens.

## Results (fill in after `notebooks/run_on_colab.ipynb`)

| | FP16 baseline (V1) | V3 automated best | V3 manual pick | V4 agent best |
|---|---|---|---|---|
| Quant method | fp16 | `__` | `__` | `__` |
| Size on disk | `__` GB | `__` GB | `__` GB | `__` GB |
| Quality recovery | 100% | `__`% | `__`% | `__`% |
| Tokens/sec | `__` | `__` | `__` | `__` |
| TTFT p50/p95 | `__` / `__` ms | `__` / `__` ms | `__` / `__` ms | `__` / `__` ms |
| TPOT p50/p95 | `__` / `__` ms | `__` / `__` ms | `__` / `__` ms | `__` / `__` ms |
| E2E p50/p95 | `__` / `__` ms | `__` / `__` ms | `__` / `__` ms | `__` / `__` ms |
| VRAM | `__` GB | `__` GB | `__` GB | `__` GB |
| $/request | `__` | `__` | `__` | `__` |

V4 agent metrics (from `results/v4_agent_optimizer.json`): diagnosis
accuracy vs. the deterministic oracle, number of new experiments run
(vs. V3's full sweep), how many were "unnecessary" (didn't end up on the
final Pareto frontier), and whether it beat V3's automated/manual picks --
see [report/FINAL_REPORT.md](report/FINAL_REPORT.md) for the full writeup.

![comparison](results/comparison.png)
![pareto](results/pareto.png)

## T4 / Turing constraints (inherited from the validated notebooks)

`float16` only (no native bf16) · no FP8 (Hopper/Ada only) · no
FlashAttention-2 (Ampere+ only -- included in the V2 sweep anyway as an
honest negative result, not silently skipped) · single server at a time to
fit 15GB VRAM.

## Testing

```
pytest tests/   # 13 tests, all deterministic, no GPU needed
```

## License

MIT-equivalent -- see the sibling [agent-review-loop](https://github.com/sourds42/agent-review-loop)
repo's conventions; add a LICENSE file if you want this formalized.
