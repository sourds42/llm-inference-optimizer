"""
Local results dashboard for the V1 -> V4 story. Loads results/comparison.json,
results/v3_optimizer.json, results/v4_agent_optimizer.json, and the raw
results/experiments.jsonl sweep. No GPU needed -- this is what you run
locally to show the demo and analyse results after a Colab run (there's no
live-generation tab: the lightweight stack loads models in-process rather
than behind a server, so there's no long-lived endpoint to hit from a
separate local process).

Run: python -m app.demo
"""
from __future__ import annotations
import os, json
from pathlib import Path
import gradio as gr

RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))


def _load_json(name: str):
    f = RESULTS / name
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _load_jsonl(name: str) -> list:
    f = RESULTS / name
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def _story_table():
    story = _load_json("comparison.json")
    if not story:
        return [["No results yet", "run notebooks/run_on_colab.ipynb first", "", "", ""]]
    cols = ["V1_baseline", "V3_automated_best", "V3_manual_best", "V4_agent_best"]
    keys = ["quant_method", "size_gb", "tokens_per_sec", "ttft_p95_ms", "tpot_p95_ms",
            "e2e_p95_ms", "vram_gb", "quality_recovery_pct", "cost_usd_per_request"]
    rows = []
    for k in keys:
        row = [k]
        for c in cols:
            v = (story.get(c) or {}).get(k, "—")
            row.append(str(v))
        rows.append(row)
    return rows


def _v3_summary():
    v3 = _load_json("v3_optimizer.json")
    if not v3:
        return "No V3 results yet -- run `python -m eval.run_v3_auto_optimizer`."
    auto = v3.get("automated", {})
    manual = v3.get("manual_chosen")
    return (
        f"**Automated (V3):** {auto.get('n_passing', 0)}/{auto.get('n_considered', 0)} configs pass "
        f"constraints, {len(auto.get('frontier', []))} on the Pareto frontier. "
        f"Chosen: `{(auto.get('chosen') or {}).get('id', '—')}`\n\n"
        f"**Manual pick:** `{(manual or {}).get('id', '—')}`"
    )


def _v4_summary():
    v4 = _load_json("v4_agent_optimizer.json")
    if not v4:
        return "No V4 results yet -- run `python -m eval.run_v4_agent_optimizer`."
    return (
        f"**Verdict:** {v4.get('verdict', '—')} after {v4.get('rounds', '—')} round(s), "
        f"{v4.get('n_new_experiments', '—')} new experiment(s) "
        f"({v4.get('unnecessary_experiments', '—')} not on the final Pareto frontier)\n\n"
        f"**Diagnosis accuracy vs. oracle:** {v4.get('diagnosis_accuracy', '—')}\n\n"
        f"**Model backend:** `{v4.get('model_backend', '—')}`\n\n"
        f"**Agent's best config:** `{(v4.get('agent_best') or {}).get('id', '—')}`"
    )


def _sweep_table():
    rows = _load_jsonl("experiments.jsonl")
    if not rows:
        return [["No experiments yet", "", "", "", "", "", "", ""]]
    keys = ["id", "quant_method", "size_gb", "tokens_per_sec", "e2e_p95_ms",
            "vram_gb", "quality_recovery_pct", "cost_usd_per_request", "error"]
    return [[str(r.get(k, "—")) for k in keys] for r in rows]


def build():
    with gr.Blocks(title="LLM Inference Optimizer") as app:
        gr.Markdown(
            "# LLM Inference Optimizer\n"
            "V1 FP16 baseline → V2 quantization/runtime-opt sweep (transformers + "
            "bitsandbytes, in-process) → V3 deterministic optimizer → V4 "
            "agent-assisted optimizer. Results below come from `results/` -- run "
            "`notebooks/run_on_colab.ipynb` on a T4 to populate them."
        )
        gr.Markdown("### V1 → V4 story (`results/comparison.json`)")
        gr.Dataframe(
            value=_story_table(),
            headers=["metric", "V1 baseline", "V3 automated", "V3 manual", "V4 agent"],
            interactive=False,
        )
        for png in ("comparison.png", "pareto.png"):
            p = RESULTS / png
            if p.exists():
                gr.Image(value=str(p), label=png, show_label=True)

        gr.Markdown("### V3 -- deterministic optimizer")
        gr.Markdown(_v3_summary())

        gr.Markdown("### V4 -- agent-assisted optimizer")
        gr.Markdown(_v4_summary())

        gr.Markdown("### Full V2 sweep (`results/experiments.jsonl`)")
        gr.Dataframe(
            value=_sweep_table(),
            headers=["id", "quant_method", "size_gb", "tokens/s", "e2e_p95_ms",
                     "vram_gb", "quality_recovery_%", "$/request", "error"],
            interactive=False,
        )
    return app


if __name__ == "__main__":
    build().launch()
