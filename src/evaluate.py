"""
Quality signal: sliding-window NLL perplexity, ported from
LLM_capstone_quality_and_speed.ipynb, computed directly on the already-
loaded model -- no separate lm_eval harness/install needed. Traded away:
HellaSwag accuracy (would need lm_eval); perplexity alone is the quality
signal in this lightweight path.
"""
from __future__ import annotations


def perplexity(model, tok, max_tokens: int = 2048, stride: int = 512) -> float:
    import torch
    from datasets import load_dataset

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

    return round(torch.exp(torch.stack(nlls).sum() / end).item(), 3)
