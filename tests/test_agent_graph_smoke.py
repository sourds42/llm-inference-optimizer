import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ExperimentConfig
from src.model_client import MockClient
from src.optimizer.constraints import Constraints
from src.agent.graph import build_graph, new_state, MAX_ROUNDS


def _fake_rows():
    return {
        "cfg_bad_1": {"id": "cfg_bad_1", "quant_method": "gptq_w4a16", "vram_gb": 6,
                      "tokens_per_sec": 40, "e2e_p95_ms": 2000, "quality_recovery_pct": 90,
                      "cost_usd_per_request": 0.002},
        "cfg_bad_2": {"id": "cfg_bad_2", "quant_method": "awq_w4a16", "vram_gb": 6,
                      "tokens_per_sec": 40, "e2e_p95_ms": 2200, "quality_recovery_pct": 88,
                      "cost_usd_per_request": 0.002},
        "cfg_good": {"id": "cfg_good", "quant_method": "gptq_w8a16", "vram_gb": 5,
                     "tokens_per_sec": 60, "e2e_p95_ms": 800, "quality_recovery_pct": 97,
                     "cost_usd_per_request": 0.0008},
    }


def _configs(ids):
    return {i: ExperimentConfig(id=i) for i in ids}


def test_agent_finds_a_satisfying_config():
    rows = _fake_rows()
    model = MockClient(seed=1, correct_rate=1.0)
    constraints = Constraints(quality_min_recovery_pct=95, p95_latency_ms_max=1000, vram_gb_max=10)
    configs_by_id = _configs(["cfg_bad_1", "cfg_good"])

    def run_experiment(cfg):
        return rows[cfg.id]

    app = build_graph(model, run_experiment, configs_by_id, constraints)
    state = new_state([], list(configs_by_id.keys()))
    out = app.invoke(state)

    assert out["verdict"] == "satisfied"
    assert out["round"] <= MAX_ROUNDS
    assert any(r["id"] == "cfg_good" for r in out["rows"])


def test_agent_exhausts_when_nothing_passes():
    rows = _fake_rows()
    model = MockClient(seed=2, correct_rate=1.0)
    constraints = Constraints(quality_min_recovery_pct=99)  # nothing in _fake_rows() clears this
    configs_by_id = _configs(["cfg_bad_1", "cfg_bad_2"])

    def run_experiment(cfg):
        return rows[cfg.id]

    app = build_graph(model, run_experiment, configs_by_id, constraints)
    state = new_state([], list(configs_by_id.keys()))
    out = app.invoke(state)

    assert out["verdict"] == "exhausted"
    assert out["round"] <= MAX_ROUNDS
    assert set(out["tried_ids"]) == {"cfg_bad_1", "cfg_bad_2"}
