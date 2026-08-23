import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ExperimentConfig


def test_defaults():
    cfg = ExperimentConfig()
    assert cfg.model_id == "Qwen/Qwen3-0.6B"
    assert cfg.quant_method == "fp16"
    assert cfg.dtype == "float16"


def test_load_from_yaml_overrides_defaults():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("quant_method: bnb_int4\nbatch_size: 16\n")
        path = f.name
    try:
        cfg = ExperimentConfig.load(path)
        assert cfg.quant_method == "bnb_int4"
        assert cfg.batch_size == 16
        assert cfg.dtype == "float16"  # untouched default
    finally:
        os.unlink(path)


def test_to_dict_roundtrip():
    cfg = ExperimentConfig(id="x", quant_method="bnb_int8")
    d = cfg.to_dict()
    assert d["id"] == "x"
    assert d["quant_method"] == "bnb_int8"
