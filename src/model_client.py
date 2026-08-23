"""
Single entry point for all model calls used by the V4 optimization agent.
Same interface/backends as the sibling agent-review-loop repo's
model_client.py: swap the backend here, nothing else in src/agent changes.

Backends:
  MockClient      - deterministic, no network, used for CI and the default demo run
  OllamaClient    - local small model (already installed: codellama:13b, deepseek-r1:7b)
  AnthropicClient - hosted API, used for a real-reasoning comparison run
"""
import time
import random
import re
import json
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ModelClient:
    name = "base"

    def generate(self, system: str, user: str, temperature: float = 0.7,
                 max_tokens: int = 512) -> GenerationResult:
        raise NotImplementedError


class MockClient(ModelClient):
    """
    Deterministic stand-in for CI and the free/default demo run. It doesn't
    reason -- for a diagnosis prompt it runs a simplified rule-of-thumb over
    the embedded result metrics, and for a proposal prompt it picks the
    first untried config id. Both are wrong `1 - correct_rate` of the time
    (a random alternative substituted for diagnosis), so the V4 eval layer
    has real diagnosis-accuracy signal instead of a trivial 100%.
    """
    name = "mock-small-llm"
    BOTTLENECKS = ["memory-bound", "compute-bound", "latency-bound", "quality-cliff", "balanced"]

    def __init__(self, seed: int = 42, correct_rate: float = 0.7):
        self.rng = random.Random(seed)
        self.correct_rate = correct_rate

    def generate(self, system: str, user: str, temperature: float = 0.7,
                 max_tokens: int = 512) -> GenerationResult:
        t0 = time.time()
        time.sleep(0.005)  # tiny simulated local-CPU latency
        sys_l = system.lower()
        if "bottleneck" in sys_l:
            text = self._diagnose(user)
        elif "config_id" in sys_l:
            text = self._propose(user)
        else:
            text = "{}"
        latency = time.time() - t0
        return GenerationResult(
            text=text, latency_s=latency,
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(text.split()), cost_usd=0.0,
        )

    def _diagnose(self, user: str) -> str:
        row = self._extract_json(user, "Result:")
        guess = self._heuristic_bottleneck(row) if row else "balanced"
        if self.rng.random() >= self.correct_rate:
            guess = self.rng.choice([b for b in self.BOTTLENECKS if b != guess])
        return json.dumps({"bottleneck": guess,
                            "rationale": f"mock heuristic over {sorted(row.keys()) if row else 'no data'}"})

    def _propose(self, user: str) -> str:
        m = re.search(r"Untried config ids:\s*(\[.*?\])", user, re.S)
        ids = json.loads(m.group(1)) if m else []
        choice = ids[0] if ids else None
        return json.dumps({"config_id": choice, "hypothesis": "mock: trying the next untried config in order"})

    @staticmethod
    def _extract_json(user: str, marker: str):
        idx = user.find(marker)
        if idx == -1:
            return None
        rest = user[idx + len(marker):]
        m = re.search(r"\{.*?\}(?=\n|$)", rest, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _heuristic_bottleneck(row: dict) -> str:
        if row.get("quality_recovery_pct") is not None and row["quality_recovery_pct"] < 95:
            return "quality-cliff"
        if (row.get("vram_gb") or 0) > 12:
            return "memory-bound"
        if (row.get("tokens_per_sec") or 999) < 20:
            return "compute-bound"
        if (row.get("e2e_p95_ms") or 0) > 1500:
            return "latency-bound"
        return "balanced"


class OllamaClient(ModelClient):
    """Local small model via Ollama's OpenAI-compatible endpoint."""
    name = "ollama-local"

    def __init__(self, model: str = "codellama:13b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, system: str, user: str, temperature: float = 0.7,
                 max_tokens: int = 512) -> GenerationResult:
        import requests
        t0 = time.time()
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False,
            },
            timeout=60,
        ).json()
        latency = time.time() - t0
        text = resp.get("message", {}).get("content", "")
        return GenerationResult(
            text=text, latency_s=latency,
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
            cost_usd=0.0,
        )


class AnthropicClient(ModelClient):
    """Hosted API -- used for a real-reasoning upper-capability comparison run."""
    name = "anthropic-hosted"
    PRICE_IN = 3.0 / 1_000_000   # illustrative, check current pricing
    PRICE_OUT = 15.0 / 1_000_000

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def generate(self, system: str, user: str, temperature: float = 0.7,
                  max_tokens: int = 512) -> GenerationResult:
        import anthropic
        client = anthropic.Anthropic()
        t0 = time.time()
        resp = client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        latency = time.time() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")
        cost = resp.usage.input_tokens * self.PRICE_IN + resp.usage.output_tokens * self.PRICE_OUT
        return GenerationResult(
            text=text, latency_s=latency,
            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
            cost_usd=cost,
        )
