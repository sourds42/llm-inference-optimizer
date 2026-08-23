"""
Run: python -m eval.compare_v1_to_v4

Final V1->V4 story: pulls the FP16 baseline row and the best V3-automated /
V3-manual / V4-agent configs, and lays out the before/after numbers the
spec asks for (quality, VRAM, P50/P95 latency, TTFT, TPOT, throughput,
tokens/sec, cost) via src.report.

No GPU required -- reads whatever results/*.json(l) a prior Colab run left behind.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.tools import get_results_so_far
from src.report import build_report


def main():
    rows = get_results_so_far("results")
    baseline = next((r for r in rows if r["quant_method"] == "fp16"), None)

    v3_path, v4_path = "results/v3_optimizer.json", "results/v4_agent_optimizer.json"
    v3 = json.load(open(v3_path)) if os.path.exists(v3_path) else None
    v4 = json.load(open(v4_path)) if os.path.exists(v4_path) else None

    story = {
        "V1_baseline": baseline,
        "V3_automated_best": v3["automated"]["chosen"] if v3 else None,
        "V3_manual_best": v3["manual_chosen"] if v3 else None,
        "V4_agent_best": v4["agent_best"] if v4 else None,
    }
    build_report(story, rows, results_dir="results")
    print(json.dumps({k: (v.get("id") if isinstance(v, dict) else v) for k, v in story.items()}, indent=2))


if __name__ == "__main__":
    main()
