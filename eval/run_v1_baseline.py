"""
Run: python -m eval.run_v1_baseline

V1 -- the FP16 baseline. A single ExperimentConfig(quant_method="fp16"),
run end-to-end through src.experiment.run_experiment. Every later V
(quantization, optimizer, agent) measures itself against this row.

GPU required (vLLM + a real model) -- run this from notebooks/run_on_colab.ipynb.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.experiment import run_experiment


def main():
    cfg = ExperimentConfig(id="V1__fp16_baseline", tier="V1", quant_method="fp16")
    row = run_experiment(cfg)
    print("[V1] baseline row:", row)


if __name__ == "__main__":
    main()
