"""
Run: python -m eval.run_v2_quant_sweep

V2 -- the full one-variable-at-a-time sweep from configs/search_space.yaml
(quantization ladder + runtime-opt knobs), each run through
src.experiment.run_experiment and appended to results/experiments.jsonl.

GPU required -- run this from notebooks/run_on_colab.ipynb, after V1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.search_space import load_search_space, enumerate_configs
from src.experiment import run_experiment


def main():
    base, axes = load_search_space("configs/search_space.yaml")
    configs = enumerate_configs(base, axes, tier="V2")
    print(f"[V2] running {len(configs)} configs (one-variable-at-a-time sweep)")
    for cfg in configs:
        run_experiment(cfg)


if __name__ == "__main__":
    main()
