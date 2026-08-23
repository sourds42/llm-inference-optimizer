"""
Builds the V2 experiment grid from configs/search_space.yaml: a `base`
config plus a set of `axes`, each varied ONE AT A TIME off the base (per
the spec's "changing one variable at a time" requirement) so every
measured delta is attributable to a single knob.
"""
from dataclasses import replace
import yaml
from src.config import ExperimentConfig


def load_search_space(path: str):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base = ExperimentConfig(**data.get("base", {}))
    axes = data.get("axes", {})
    return base, axes


def enumerate_configs(base: ExperimentConfig, axes: dict, tier: str = "V2") -> list:
    configs = [replace(base, id=f"{tier}__baseline", tier=tier)]
    for axis_name, values in axes.items():
        for value in values:
            if getattr(base, axis_name) == value:
                continue  # already covered by the baseline row
            cfg_id = f"{tier}__{axis_name}={value}"
            configs.append(replace(base, id=cfg_id, tier=tier, **{axis_name: value}))
    return configs
