"""
diagnose -> propose -> [experiment runs deterministically] -> record, capped
at MAX_ROUNDS and stopped early on either a constraint-satisfying config
being found or PATIENCE rounds with no improvement in the best cost among
passing configs. Mirrors agent-review-loop/src/graph.py's
generate -> tool_check -> review shape: the agent proposes, the
deterministic tool (src.experiment.run_experiment via src.optimizer) is the
only thing that gets to say whether a result is good.
"""
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from src.agent.diagnose import diagnose
from src.agent.propose import propose
from src.optimizer.constraints import Constraints, filter_passing
from src.optimizer.pareto import pareto_frontier
from src.optimizer.search import best_from_frontier
from src.model_client import ModelClient

MAX_ROUNDS = 5
PATIENCE = 2


class AgentState(TypedDict):
    rows: List[dict]
    tried_ids: List[str]
    remaining_ids: List[str]
    diagnosis: Optional[dict]
    proposal: Optional[dict]
    best_cost: Optional[float]
    rounds_no_improvement: int
    round: int
    verdict: Optional[str]
    log: List[dict]
    total_cost_usd: float
    total_latency_s: float


def build_graph(model: ModelClient, run_experiment, configs_by_id: dict, constraints: Constraints):
    """configs_by_id: {config_id: ExperimentConfig}.
    run_experiment: ExperimentConfig -> result dict (same schema as experiments.jsonl rows)."""

    def diagnose_node(state: AgentState) -> AgentState:
        state["round"] += 1
        latest = state["rows"][-1] if state["rows"] else None
        constraint_summary = {
            "quality_min_recovery_pct": constraints.quality_min_recovery_pct,
            "p95_latency_ms_max": constraints.p95_latency_ms_max,
            "vram_gb_max": constraints.vram_gb_max,
        }
        res = (diagnose(model, latest, constraint_summary) if latest
               else {"bottleneck": "unknown", "rationale": "no results yet"})
        state["diagnosis"] = res
        state["log"].append({"node": "diagnose", "round": state["round"], **res})
        return state

    def propose_node(state: AgentState) -> AgentState:
        history = [{k: r.get(k) for k in
                    ("id", "quant_method", "vram_gb", "tokens_per_sec", "e2e_p95_ms", "quality_recovery_pct")}
                   for r in state["rows"][-5:]]
        prop = propose(model, state["diagnosis"], history, state["remaining_ids"])
        state["proposal"] = prop
        state["log"].append({"node": "propose", "round": state["round"], **prop})
        return state

    def experiment_node(state: AgentState) -> AgentState:
        cfg_id = state["proposal"].get("config_id")
        if cfg_id not in configs_by_id:
            state["log"].append({"node": "experiment", "round": state["round"], "skipped": True})
            return state
        cfg = configs_by_id[cfg_id]
        result = run_experiment(cfg)
        state["rows"].append(result)
        state["tried_ids"].append(cfg_id)
        state["remaining_ids"] = [c for c in state["remaining_ids"] if c != cfg_id]
        state["log"].append({"node": "experiment", "round": state["round"], "config_id": cfg_id})
        return state

    def record_node(state: AgentState) -> AgentState:
        passing = filter_passing(state["rows"], constraints)
        frontier = pareto_frontier(passing)
        best = best_from_frontier(frontier)
        best_cost = best.get("cost_usd_per_request") if best else None

        improved = best_cost is not None and (state["best_cost"] is None or best_cost < state["best_cost"])
        state["rounds_no_improvement"] = 0 if improved else state["rounds_no_improvement"] + 1
        if improved:
            state["best_cost"] = best_cost

        if best is not None:
            state["verdict"] = "satisfied"
        elif not state["remaining_ids"]:
            state["verdict"] = "exhausted"
        elif state["round"] >= MAX_ROUNDS:
            state["verdict"] = "budget_exhausted"
        elif state["rounds_no_improvement"] >= PATIENCE:
            state["verdict"] = "no_improvement"
        else:
            state["verdict"] = "continue"
        state["log"].append({"node": "record", "round": state["round"],
                              "verdict": state["verdict"], "best_cost": best_cost})
        return state

    def route(state: AgentState) -> str:
        return "continue" if state["verdict"] == "continue" else "stop"

    graph = StateGraph(AgentState)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("propose", propose_node)
    graph.add_node("experiment", experiment_node)
    graph.add_node("record", record_node)
    graph.set_entry_point("diagnose")
    graph.add_edge("diagnose", "propose")
    graph.add_edge("propose", "experiment")
    graph.add_edge("experiment", "record")
    graph.add_conditional_edges("record", route, {"continue": "diagnose", "stop": END})
    return graph.compile()


def new_state(initial_rows: List[dict], all_config_ids: List[str]) -> AgentState:
    tried = {r["id"] for r in initial_rows}
    return {
        "rows": list(initial_rows),
        "tried_ids": list(tried),
        "remaining_ids": [c for c in all_config_ids if c not in tried],
        "diagnosis": None, "proposal": None,
        "best_cost": None, "rounds_no_improvement": 0, "round": 0,
        "verdict": None, "log": [],
        "total_cost_usd": 0.0, "total_latency_s": 0.0,
    }
