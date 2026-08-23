"""
Read-only accessors exposed to the optimization agent. Every function here
reads already-computed, deterministic ground truth (results/experiments.jsonl,
constraints, the Pareto frontier) -- there is no tool that lets the agent
mark a result as passing or failing.
"""
import json
from pathlib import Path
from src.optimizer.constraints import Constraints, filter_passing
from src.optimizer.pareto import pareto_frontier


def get_results_so_far(results_dir: str = "results") -> list:
    path = Path(results_dir) / "experiments.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def get_remaining_search_space(all_configs: list, results_dir: str = "results") -> list:
    tried_ids = {r["id"] for r in get_results_so_far(results_dir)}
    return [c for c in all_configs if c.id not in tried_ids]


def get_constraint_status(constraints: Constraints, results_dir: str = "results") -> dict:
    rows = get_results_so_far(results_dir)
    passing = filter_passing(rows, constraints)
    return {"n_total": len(rows), "n_passing": len(passing), "any_passing": len(passing) > 0}


def get_pareto_frontier(constraints: Constraints, results_dir: str = "results") -> list:
    rows = filter_passing(get_results_so_far(results_dir), constraints)
    return pareto_frontier(rows)
