"""
The review prompt: given a task spec and a candidate implementation, the
model must respond with ONLY a JSON object describing its verdict.
Parsing mirrors the JSON-fence extraction pattern already used in
src/agent/diagnose.py and agent-review-loop/src/graph.py::extract_code.
"""
import json
import re

REVIEW_SYSTEM = (
    "You are a careful code reviewer. Given a requirement and a candidate "
    "Python implementation, decide if it has a bug. Respond ONLY with a "
    'JSON object: {"has_bug": true|false, '
    '"category": "logic"|"security"|"performance"|"style"|"none", '
    '"severity": "critical"|"major"|"minor"|"none", '
    '"description": "one sentence"}'
)


def build_review_prompt(item: dict) -> str:
    return (f"Requirement: {item['spec']}\n\n"
            f"Candidate implementation:\n```python\n{item['code']}\n```")


def parse_review(text: str) -> dict:
    fallback = {"has_bug": None, "category": None, "severity": None,
                "description": "unparseable model output"}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return fallback
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback
    if "has_bug" not in data or not isinstance(data["has_bug"], bool):
        return fallback
    data.setdefault("category", "none")
    data.setdefault("severity", "none")
    data.setdefault("description", "")
    return data
