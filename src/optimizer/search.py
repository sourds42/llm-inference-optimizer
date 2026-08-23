"""
V3 deterministic optimizer.

Default mode (`replay_search`): filter the already-collected V2 experiment
log by Constraints, rank the survivors via the Pareto frontier, and pick the
lowest-cost frontier point as the single recommended config. This mode
never touches the GPU, so V3 can always be demoed even without a live Colab
session.

Optional mode (`active_search`): given a `run_experiment` callback
(typically src.experiment.run_experiment), additionally probes new configs
drawn from the remaining search space, up to a fixed budget, re-ranking
after each new result -- useful when running inside Colab with GPU time to
spend.
"""
from src.optimizer.constraints import Constraints, filter_passing
from src.optimizer.pareto import pareto_frontier


def best_from_frontier(frontier: list):
    if not frontier:
        return None
    return min(frontier, key=lambda r: r.get("cost_usd_per_request", float("inf")))


def replay_search(rows: list, constraints: Constraints) -> dict:
    passing = filter_passing(rows, constraints)
    frontier = pareto_frontier(passing)
    chosen = best_from_frontier(frontier)
    return {
        "mode": "replay",
        "n_considered": len(rows),
        "n_passing": len(passing),
        "frontier": frontier,
        "chosen": chosen,
    }


def active_search(rows: list, remaining_configs: list, constraints: Constraints,
                   run_experiment, budget: int = 5) -> dict:
    rows = list(rows)
    tried = []
    for cfg in remaining_configs[:budget]:
        result = run_experiment(cfg)
        rows.append(result)
        tried.append(result)
    out = replay_search(rows, constraints)
    out["mode"] = "active"
    out["newly_tried"] = tried
    return out
