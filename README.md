# LLM Inference Optimization System

A production-oriented (prototype, not production-deployed) system that
determines the best model/serving configuration under constraints for
**quality, latency, throughput, GPU memory, and cost** -- and layers an LLM
agent on top that reasons about *which experiment to try next*, while a
deterministic tool stays the only thing allowed to say whether a config
actually passes.

Built as a V1 -> V2 -> V3 -> V4 story, deliberately on a **lightweight
stack** (`transformers` + `bitsandbytes`, in-process generation -- no
vLLM/llmcompressor/lm-eval) so a Colab setup takes minutes, not ten-plus:

```
V1  FP16 baseline  ──▶  V2  quantization + runtime-opt sweep  ──▶  V3  deterministic optimizer  ──▶  V4  agent-assisted optimizer
    Qwen3-0.6B, T4        one variable at a time:                    constraint filter                diagnose → propose → experiment
                          bnb INT8/INT4, batch size,                 + Pareto frontier                loop, capped + patience-stopped
                          context length, KV-cache,                  + manual-selection                (src/agent/graph.py)
                          attention implementation                   comparison
                          (src/quantize.py, src/benchmark.py)        (src/optimizer/)
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
  quantize.py             loads the model in-process: FP16 or bitsandbytes INT8/INT4 (quantizes at load time, no separate build step)
  benchmark.py            in-process generate() timing: TTFT (approx.), TPOT, tokens/sec, throughput, E2E p50/p95/p99, VRAM/GPU-util
  evaluate.py             sliding-window perplexity (no lm_eval)
  cost.py                 $/request from a documented $/GPU-hr rate
  experiment.py           THE deterministic ground truth -- runs one config end-to-end, appends to results/experiments.jsonl (failures recorded, not fatal)
  optimizer/               V3: search_space, constraints, pareto, search (replay + active), manual_baseline
  agent/                   V4: tools (read-only), diagnose, propose, graph (LangGraph loop), oracle (eval-only ground truth)
  model_client.py         MockClient (free/default) / OllamaClient / AnthropicClient -- pluggable V4 backend
  report.py               results/experiments.jsonl → comparison.json + charts
eval/
  run_v1_baseline.py · run_v2_quant_sweep.py · run_v3_auto_optimizer.py · run_v4_agent_optimizer.py · compare_v1_to_v4.py
app/demo.py               local results-dashboard (Gradio, no GPU needed) -- `python -m app.demo`
notebooks/run_on_colab.ipynb   single T4 entrypoint, all four stages
tests/                    deterministic only -- no GPU needed, run right now with `pytest tests/`
report/FINAL_REPORT.md    case study writeup (fill in after a real Colab run)
```

## Quickstart

**Colab (T4, does the real GPU work):**
Open `notebooks/run_on_colab.ipynb` in Colab, `Runtime → T4 GPU`, run cells
top to bottom. Install is a couple of minutes (`transformers` +
`bitsandbytes`, nothing heavier). Each V is a separate cell/script -- V3 and
the final report don't need the GPU at all, they just read whatever
`results/experiments.jsonl` has so far.

**Local (deterministic layers + results dashboard, no GPU required):**
```bash
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
pytest tests/                          # 13 tests: config, constraints, pareto, cost, agent graph smoke
python -m eval.run_v3_auto_optimizer   # ranks whatever results/experiments.jsonl already has
python -m eval.run_v4_agent_optimizer  # MockClient by default; MODEL_BACKEND=ollama|anthropic for a real run
python -m app.demo                     # results dashboard in the browser
```

## What's real vs. what needs a Colab run to prove

Everything under `src/optimizer/`, `src/agent/`, `src/config.py`,
`src/cost.py`, and `app/demo.py` is deterministic, unit-tested, and runs on
this machine right now (verified: 13/13 tests pass, no GPU here to test
against). `src/quantize.py` / `benchmark.py` / `evaluate.py` are plain
`transformers` + `bitsandbytes` code -- individually standard APIs, but the
full sweep (including the deliberately-included
`attn_implementation=flash_attention_2` config that's expected to fail on
T4) hasn't been run end-to-end on a real T4 yet. Results tables below are
placeholders until that run happens.

## Results (fill in after `notebooks/run_on_colab.ipynb`)

| | FP16 baseline (V1) | V3 automated best | V3 manual pick | V4 agent best |
|---|---|---|---|---|
| Quant method | fp16 | `__` | `__` | `__` |
| Quality recovery (perplexity-based) | 100% | `__`% | `__`% | `__`% |
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

Run `python -m app.demo` for a live, browser-based version of this table
plus the full V2 sweep and the charts below.

![comparison](results/comparison.png)
![pareto](results/pareto.png)

## What this lightweight stack trades away

Chosen deliberately for setup speed and to keep the demo focused on
*results and analysis* rather than fighting installs:

- **No GPTQ/AWQ** (would need `llmcompressor` + a calibration pass) --
  quantization is bitsandbytes INT8/INT4 only.
- **No vLLM serving** -- no PagedAttention/continuous-batching/prefix-cache
  metrics; benchmarking is direct in-process `generate()` calls instead of
  hitting an OpenAI-compatible server.
- **No lm_eval / HellaSwag** -- quality signal is perplexity only.
- **TTFT is approximated** via a separate `max_new_tokens=1` pass, not true
  per-token streaming (see `src/benchmark.py`'s docstring).
- **Speculative decoding isn't in the default sweep** -- no smaller draft
  model in the same tokenizer family was readily available for
  Qwen3-0.6B; the `assistant_model_id` field exists on `ExperimentConfig`
  if you want to try it with a compatible draft model.

## T4 / Turing constraints

`float16` only (no native bf16) · no FP8 (Hopper/Ada only) · no
FlashAttention-2 (Ampere+ only -- included in the V2 sweep anyway as an
honest negative result, not silently skipped).

## Testing

```
pytest tests/   # 13 tests, all deterministic, no GPU needed
```

## License

MIT-equivalent -- see the sibling [agent-review-loop](https://github.com/sourds42/agent-review-loop)
repo's conventions; add a LICENSE file if you want this formalized.
