"""
Launch and manage a vLLM OpenAI-compatible server as a background process.

Ported from llm-inference-lab/src/serve.py (already proven on a 15GB T4:
fail-fast readiness polling with log-tail-on-crash, and an explicit
VRAM-release sleep on stop) and extended to route every runtime-optimization
knob under test (batch size, context length, prefix caching, attention
backend, speculative decoding, quantization method) through the same
server, not a parallel code path -- see src/runtime_opts.py for the knob ->
flag mapping.
"""
from __future__ import annotations
import subprocess, time, urllib.request
from pathlib import Path

from src.runtime_opts import vllm_args, quantization_serve_flags


class VLLMServer:
    def __init__(self, cfg, model_path: str, port: int, *, quant_method: str = "fp16",
                 served_name: str = "model", results_dir: str = "results"):
        self.cfg = cfg
        self.model_path = model_path
        self.port = port
        self.quant_method = quant_method
        self.served_name = served_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._proc = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def start(self) -> "VLLMServer":
        self.stop()
        self._log = open(self.results_dir / f"vllm_{self.port}.log", "w")
        args = [
            "vllm", "serve", self.model_path,
            "--dtype", self.cfg.dtype,
            "--max-model-len", str(self.cfg.max_model_len),
            "--gpu-memory-utilization", str(self.cfg.gpu_memory_utilization),
            "--port", str(self.port),
            "--served-model-name", self.served_name,
        ] + vllm_args(self.cfg) + quantization_serve_flags(self.quant_method)
        self._proc = subprocess.Popen(args, stdout=self._log, stderr=subprocess.STDOUT)
        return self

    def wait_ready(self, timeout: int = 300) -> "VLLMServer":
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://localhost:{self.port}/v1/models", timeout=2)
                return self
            except Exception:
                if self._proc and self._proc.poll() is not None:
                    tail = (self.results_dir / f"vllm_{self.port}.log").read_text()[-3000:]
                    raise RuntimeError(f"vLLM exited early:\n{tail}")
                time.sleep(2)
        raise TimeoutError(f"vLLM on :{self.port} not ready within {timeout}s")

    def metrics(self, keys=("num_requests_running", "num_requests_waiting",
                             "gpu_cache_usage", "prefix")) -> dict:
        raw = urllib.request.urlopen(f"http://localhost:{self.port}/metrics", timeout=5).read().decode()
        out = {}
        for line in raw.splitlines():
            if line.startswith("#"):
                continue
            if any(k in line for k in keys):
                name, _, val = line.partition(" ")
                out[name] = val.strip()
        return out

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()
        if self._log:
            self._log.close()
        self._proc = None
        time.sleep(3)  # give the driver a moment to release VRAM

    def __enter__(self):
        return self.start().wait_ready()

    def __exit__(self, *exc):
        self.stop()
