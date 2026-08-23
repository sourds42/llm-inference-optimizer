"""
Aggregates the V1->V4 story into results/comparison.json plus charts,
extending llm-inference-lab/src/report.py's comparison.json + PNG pattern
to the larger V1-V4 comparison this project makes.
"""
from __future__ import annotations
import json
from pathlib import Path


def build_report(story: dict, all_rows: list, results_dir: str = "results") -> dict:
    rd = Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)

    (rd / "comparison.json").write_text(json.dumps(story, indent=2, default=str))
    print("[report] wrote", rd / "comparison.json")

    try:
        _charts(story, all_rows, rd)
    except Exception as e:  # charts are nice-to-have, never fatal
        print("[report] chart step skipped:", e)
    return story


def _charts(story, all_rows, rd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [k for k in ("V1_baseline", "V3_automated_best", "V3_manual_best", "V4_agent_best")
              if story.get(k)]
    if not labels:
        return
    get = lambda k, field: (story[k] or {}).get(field, 0) or 0

    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    ax[0].bar(labels, [get(k, "size_gb") for k in labels]);              ax[0].set_title("Size on disk (GB) ↓")
    ax[1].bar(labels, [get(k, "tokens_per_sec") for k in labels]);       ax[1].set_title("Throughput (tok/s) ↑")
    ax[2].bar(labels, [get(k, "e2e_p95_ms") for k in labels]);           ax[2].set_title("E2E p95 (ms) ↓")
    ax[3].bar(labels, [get(k, "quality_recovery_pct") for k in labels]); ax[3].set_title("Quality recovery (%) ↑")
    for a in ax:
        a.grid(axis="y", alpha=.3)
        a.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(rd / "comparison.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in all_rows:
        ax.scatter(r.get("e2e_p95_ms", 0), r.get("quality_recovery_pct") or 0,
                    s=(r.get("vram_gb", 1) or 1) * 40 + 40, alpha=.5, color="#888")
    for k, color in zip(labels, ["#333", "#3B9EFF", "#2FD3A5", "#9b59b6"]):
        cfg = story[k]
        ax.scatter(cfg.get("e2e_p95_ms", 0), cfg.get("quality_recovery_pct") or 0,
                    s=220, color=color, edgecolor="k", label=k)
    ax.set_xlabel("E2E p95 latency (ms)  ←  faster")
    ax.set_ylabel("Quality recovery (%)  ↑  better")
    ax.set_title("Pareto view — all experiments (grey) vs chosen configs")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rd / "pareto.png", dpi=120); plt.close(fig)
    print("[report] wrote comparison.png + pareto.png")
