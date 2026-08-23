"""
Ask the model which untried config to try next, given the diagnosis and
recent run history. The proposal is a *suggestion* -- src.experiment runs
it exactly as any other config would, through the same deterministic
pipeline; the agent has no shortcut to mark results itself.
"""
import json
import re

PROPOSAL_SYSTEM = (
    "You are an LLM-serving performance engineer choosing the next "
    "experiment to run. Given the diagnosis, recent run history, and the "
    "list of untried config ids, pick exactly one untried config id and "
    "state a one-sentence hypothesis for why it should help. Respond ONLY "
    'with a JSON object: {"config_id": "...", "hypothesis": "..."}'
)


def propose(model_client, diagnosis: dict, history: list, remaining_ids: list) -> dict:
    user = (
        f"Diagnosis: {json.dumps(diagnosis)}\n"
        f"History (last 5): {json.dumps(history[-5:])}\n"
        f"Untried config ids: {json.dumps(remaining_ids)}"
    )
    res = model_client.generate(PROPOSAL_SYSTEM, user, temperature=0.5)
    return _parse(res.text, remaining_ids)


def _parse(text: str, remaining_ids: list) -> dict:
    fallback = {"config_id": remaining_ids[0] if remaining_ids else None,
                "hypothesis": "unparseable model output, falling back to next untried config"}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return fallback
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback
    if data.get("config_id") not in remaining_ids:
        data["config_id"] = remaining_ids[0] if remaining_ids else None
        data.setdefault("hypothesis", "")
    return data
