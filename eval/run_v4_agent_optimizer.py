"""
Run: python -m eval.run_v4_agent_optimizer

V4 -- the agent-assisted optimizer. Defaults to MockClient (free,
deterministic, reproducible) -- set MODEL_BACKEND=ollama or
MODEL_BACKEND=anthropic for a real-reasoning run (same env-var convention
as the sibling agent-review-loop repo's eval/run_eval.py).

Computes the V4-specific eval metrics the spec calls for: diagnosis
accuracy (against the deterministic oracle in src.agent.oracle, which the
agent itself never sees), number of new experiments run vs V3's full
sweep, "unnecessary" experiments (newly tried configs that didn't end up on
the final Pareto frontier), success rate, and the agent's best config vs
V3's automated and manual picks.

The GPU-bound parts (each new experiment run_experiment triggers) require
Colab; with MODEL_BACKEND=mock and an empty results/ dir this will still
run its control flow correctly against whatever experiments.jsonl already
has -- it just won't find a passing config until real GPU rows exist.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_client import MockClient, OllamaClient, AnthropicClient
from src.optimizer.search_space import load_search_space, enumerate_configs
from src.optimizer.constraints import Constraints, filter_passing
from src.optimizer.manual_baseline import manual_pick
from src.optimizer.search import replay_search, best_from_frontier
from src.optimizer.pareto import pareto_frontier
from src.agent.oracle import classify_bottleneck
from src.agent.graph import build_graph, new_state
from src.agent.tools import get_results_so_far
from src.experiment import run_experiment


def build_model():
    backend = os.environ.get("MODEL_BACKEND", "mock").lower()
    if backend == "ollama":
        return OllamaClient(model=os.environ.get("OLLAMA_MODEL", "codellama:13b"))
    if backend == "anthropic":
        return AnthropicClient(model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    return MockClient(seed=42, correct_rate=0.7)


def main():
    constraints = Constraints.load("configs/constraints.yaml")
    base, axes = load_search_space("configs/search_space.yaml")
    all_configs = enumerate_configs(base, axes, tier="V4")
    configs_by_id = {c.id: c for c in all_configs}

    initial_rows = get_results_so_far("results")  # seed with V1/V2's existing rows, if any
    model = build_model()
    app = build_graph(model, run_experiment, configs_by_id, constraints)
    state = new_state(initial_rows, list(configs_by_id.keys()))
    out = app.invoke(state)

    # --- diagnosis accuracy vs the deterministic oracle ---
    diagnoses = [e for i, e in enumerate(out["log"]) if e["node"] == "diagnose" and "bottleneck" in e]
    scored, correct = 0, 0
    for i, entry in enumerate(diagnoses):
        row_index = len(initial_rows) + i - 1  # the row diagnose_node looked at this round
        if 0 <= row_index < len(out["rows"]):
            oracle_label = classify_bottleneck(out["rows"][row_index], constraints, out["rows"])
            scored += 1
            correct += int(oracle_label == entry["bottleneck"])
    diagnosis_accuracy = round(correct / scored, 3) if scored else None

    # --- experiment efficiency / unnecessary experiments ---
    n_new_experiments = sum(1 for e in out["log"] if e["node"] == "experiment" and not e.get("skipped"))
    new_rows = [r for r in out["rows"] if r["id"] in out["tried_ids"]]
    passing = filter_passing(out["rows"], constraints)
    frontier = pareto_frontier(passing)
    final_frontier_ids = {r["id"] for r in frontier}
    unnecessary_experiments = sum(1 for r in new_rows if r["id"] not in final_frontier_ids)
    agent_best = best_from_frontier(frontier)

    # --- comparison against V3 automated + manual ---
    v3 = replay_search(get_results_so_far("results"), constraints)
    manual = manual_pick(get_results_so_far("results"), constraints)

    summary = {
        "verdict": out["verdict"],
        "rounds": out["round"],
        "n_new_experiments": n_new_experiments,
        "unnecessary_experiments": unnecessary_experiments,
        "diagnosis_accuracy": diagnosis_accuracy,
        "success": agent_best is not None,
        "agent_best": agent_best,
        "v3_automated_best": v3["chosen"],
        "manual_best": manual,
        "model_backend": model.name,
    }
    with open("results/v4_agent_optimizer.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
