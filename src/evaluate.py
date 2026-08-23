"""
Two quality signals, both ported from the user's validated Colab notebooks:

- accuracy_via_server: lm_eval's "local-completions" backend against the
  live vLLM server (ported from llm-inference-lab/src/evaluate.py) -- scores
  the actual serving deployment, not just the raw weights.
- perplexity_offline: the sliding-window NLL perplexity used in
  LLM_capstone_quality_and_speed.ipynb -- doesn't need a running server, so
  it's used right after quantization for a quick quality read.
"""
from __future__ import annotations
import subprocess, json, glob
from pathlib import Path


def accuracy_via_server(base_url, task="hellaswag", limit=100, tag="model", results_dir="results"):
    out_dir = Path(results_dir) / f"lmeval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lm_eval", "--model", "local-completions",
        "--model_args",
        f"model=model,base_url={base_url}/completions,num_concurrent=4,max_retries=3,tokenized_requests=False",
        "--tasks", task, "--limit", str(limit),
        "--output_path", str(out_dir), "--batch_size", "1",
    ]
    print("[evaluate]", " ".join(cmd))
    subprocess.run(cmd, check=False)

    files = glob.glob(str(out_dir) + "/**/results*.json", recursive=True)
    if not files:
        print("[evaluate] no results json found -- check lm_eval output above.")
        return None
    res = json.load(open(sorted(files)[-1]))["results"][task]
    acc = res.get("acc,none", res.get("acc"))
    print(f"[evaluate] {tag} {task} acc = {acc}")
    return acc


def perplexity_offline(model_path: str, max_tokens: int = 4096, stride: int = 512) -> float:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cuda")
    model.eval()

    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    encodings = tok(text, return_tensors="pt")
    seq_len = min(encodings.input_ids.size(1), max_tokens)

    nlls = []
    prev_end = 0
    end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + stride, seq_len)
        trg_len = end - prev_end
        ids = encodings.input_ids[:, begin:end].to(model.device)
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(ids, labels=target_ids)
            nlls.append(out.loss * trg_len)
        prev_end = end
        if end == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end).item()
    del model
    torch.cuda.empty_cache()
    return round(ppl, 3)
