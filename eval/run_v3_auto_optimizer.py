"""
Run: python -m eval.run_v3_auto_optimizer

V3 -- the deterministic optimizer. Default replay mode ranks the V2 sweep
already sitting in results/experiments.jsonl (constraint filter -> Pareto
frontier -> lowest-cost pick); no new GPU experiments required. Also runs
the manual-selection heuristic for the "manual vs automated" comparison the
spec asks for.

No GPU required -- this can run locally against results from a prior Colab run.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.tools import get_results_so_far
from src.optimizer.constraints import Constraints
from src.optimizer.search import replay_search
from src.optimizer.manual_baseline import manual_pick


def main():
    constraints = Constraints.load("configs/constraints.yaml")
    rows = get_results_so_far("results")
    if not rows:
        print("[V3] no experiments.jsonl found -- run eval.run_v2_quant_sweep first.")
        return

    auto = replay_search(rows, constraints)
    manual = manual_pick(rows, constraints)

    out = {"automated": auto, "manual_chosen": manual}
    with open("results/v3_optimizer.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"[V3] {auto['n_passing']}/{auto['n_considered']} configs pass constraints, "
          f"{len(auto['frontier'])} on the Pareto frontier")
    print("[V3] automated pick:", auto["chosen"] and auto["chosen"]["id"])
    print("[V3] manual pick:   ", manual and manual["id"])


if __name__ == "__main__":
    main()
