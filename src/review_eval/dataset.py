"""
Assembles the fixed code-review evaluation dataset by importing
agent-review-loop's existing golden set (reused as-is, untouched -- that
repo and its own tests are never modified) and layering new
security/performance examples plus category/severity ground truth on top,
neither of which exist in the source repo today (confirmed by inspection:
only 10/34 tasks have a buggy variant at all, none are security- or
performance-flavored, and there's no structured category/severity field
anywhere).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

CATEGORIES = ("logic", "security", "performance", "style", "none")
SEVERITIES = ("critical", "major", "minor", "none")


def _find_agent_review_loop() -> Path:
    """Colab clones both repos as siblings under /content/, so the default
    relative candidate is what a fresh checkout actually looks like.
    AGENT_REVIEW_LOOP_PATH overrides it for other layouts."""
    env = os.environ.get("AGENT_REVIEW_LOOP_PATH")
    candidates = [env] if env else []
    candidates += ["../agent-review-loop"]
    for c in candidates:
        if not c:
            continue
        p = Path(c).resolve()
        if (p / "eval" / "golden_set.py").exists():
            return p
    raise FileNotFoundError(
        "Could not find agent-review-loop's eval/golden_set.py. Set "
        "AGENT_REVIEW_LOOP_PATH to its checkout root, or clone it as a "
        "sibling directory (../agent-review-loop)."
    )


def load_golden_set():
    """Imports GOLDEN_SET / SOLUTIONS / BUGGY_VARIANTS directly from
    agent-review-loop's own file -- no copy, no edits to that repo."""
    repo = _find_agent_review_loop()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from eval.golden_set import GOLDEN_SET, SOLUTIONS, BUGGY_VARIANTS  # noqa: E402
    return GOLDEN_SET, SOLUTIONS, BUGGY_VARIANTS


# The 10 golden-set tasks that already have a buggy variant -- all logic
# bugs (off-by-one, mutable-default, bare-except, etc.), per inspection.
_LOGIC_BUG_IDS = (
    "fizzbuzz", "off_by_one_range", "nth_fib_negative", "empty_list_average",
    "mutable_default_trap", "chunk_list", "rotate_array",
    "string_to_int_strict", "valid_parentheses", "binary_search",
)

# A representative (not exhaustive) subset of the no-buggy-variant
# golden-set tasks, included as clean-only true-negative controls.
_CLEAN_ONLY_IDS = (
    "is_even", "reverse_string", "sum_list", "two_sum", "quicksort",
    "divide_safe", "float_equality",
)

# New content: agent-review-loop has zero security- or performance-
# flavored bugs. Each tuple: (id, spec, clean_code, buggy_code, category, severity).
NEW_ITEMS = [
    ("render_greeting",
     "write render_greeting(name) returning an HTML greeting like "
     "'<p>Hello, NAME</p>' safe against HTML/script injection in `name`",
     "import html\ndef render_greeting(name):\n    return f'<p>Hello, {html.escape(name)}</p>'",
     "def render_greeting(name):\n    return f'<p>Hello, {name}</p>'  "
     "# XSS: no HTML escaping of user input",
     "security", "critical"),
    ("run_query",
     "write run_query(cursor, user_input) selecting from `users` where name "
     "matches user_input, using a parameterized query (not string formatting)",
     "def run_query(cursor, user_input):\n"
     "    return cursor.execute('SELECT * FROM users WHERE name = ?', (user_input,))",
     "def run_query(cursor, user_input):\n"
     "    return cursor.execute(f\"SELECT * FROM users WHERE name = '{user_input}'\")  "
     "# SQL injection via f-string",
     "security", "critical"),
    ("load_config",
     "write load_config(path) reading and parsing a YAML config file "
     "safely, not executing arbitrary YAML tags",
     "import yaml\ndef load_config(path):\n    with open(path) as f:\n        return yaml.safe_load(f)",
     "import yaml\ndef load_config(path):\n    with open(path) as f:\n"
     "        return yaml.load(f, Loader=yaml.UnsafeLoader)  "
     "# arbitrary code exec via YAML tags",
     "security", "critical"),
    ("compute",
     "write compute(expr) evaluating a simple arithmetic expression string "
     "like '2 + 3 * 4' without allowing arbitrary code execution",
     "import ast, operator\n"
     "_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}\n"
     "def _eval(node):\n"
     "    if isinstance(node, ast.BinOp):\n"
     "        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))\n"
     "    if isinstance(node, ast.Constant):\n        return node.value\n"
     "    raise ValueError('unsupported expression')\n"
     "def compute(expr):\n    return _eval(ast.parse(expr, mode='eval').body)",
     "def compute(expr):\n    return eval(expr)  # arbitrary code execution",
     "security", "critical"),
    ("has_duplicate",
     "write has_duplicate(items) returning True if any value appears more "
     "than once in the list",
     "def has_duplicate(items):\n    seen = set()\n    for x in items:\n"
     "        if x in seen:\n            return True\n        seen.add(x)\n    return False",
     "def has_duplicate(items):\n    for i in range(len(items)):\n"
     "        for j in range(len(items)):\n            if i != j and items[i] == items[j]:\n"
     "                return True\n    return False  "
     "# O(n^2) nested-loop comparison, O(n) possible with a set",
     "performance", "minor"),
    ("dedupe2",
     "write dedupe2(items) removing duplicates but preserving first-seen "
     "order, efficiently for large lists",
     "def dedupe2(items):\n    seen = set()\n    out = []\n    for x in items:\n"
     "        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
     "def dedupe2(items):\n    out = []\n    for x in items:\n"
     "        if x not in out:\n            out.append(x)\n    return out  "
     "# O(n^2): `in out` is a linear scan of a growing list",
     "performance", "minor"),
    ("count_word",
     "write count_word(text, word) counting occurrences of word in text, "
     "called repeatedly with the same text and different words",
     "def count_word(text, word):\n    words = text.split()\n    return words.count(word)",
     "def count_word(text, word):\n"
     "    return len([w for w in text.split() for _ in [0] if w == word])  "
     "# re-splits text every call, redundant work",
     "performance", "minor"),
    ("build_lookup",
     "write build_lookup(records) returning a function that looks up a "
     "record by id, called many times against the same records list",
     "def build_lookup(records):\n    index = {r['id']: r for r in records}\n"
     "    return lambda rid: index.get(rid)",
     "def build_lookup(records):\n"
     "    return lambda rid: next((r for r in records if r['id'] == rid), None)  "
     "# O(n) linear scan per lookup instead of O(1) via a dict",
     "performance", "minor"),
]


def build_review_items() -> list:
    """Returns a fixed list of review items:
    {id, spec, code, ground_truth: {has_bug, category, severity}}.
    Every logic-bug/new-item task contributes both its clean and buggy
    version (a real positive and a real negative from the same spec);
    clean-only ids contribute just the clean version."""
    golden_set, solutions, buggy = load_golden_set()
    by_id = {t["id"]: t for t in golden_set}
    items = []

    def clean_item(item_id, spec, code):
        return {"id": f"{item_id}__clean", "spec": spec, "code": code,
                "ground_truth": {"has_bug": False, "category": "none", "severity": "none"}}

    def buggy_item(item_id, spec, code, category, severity):
        return {"id": f"{item_id}__buggy", "spec": spec, "code": code,
                "ground_truth": {"has_bug": True, "category": category, "severity": severity}}

    for tid in _LOGIC_BUG_IDS:
        spec = by_id[tid]["spec"]
        items.append(clean_item(tid, spec, solutions[tid]))
        items.append(buggy_item(tid, spec, buggy[tid], "logic", "major"))

    for tid in _CLEAN_ONLY_IDS:
        items.append(clean_item(tid, by_id[tid]["spec"], solutions[tid]))

    for item_id, spec, clean_code, buggy_code, category, severity in NEW_ITEMS:
        items.append(clean_item(item_id, spec, clean_code))
        items.append(buggy_item(item_id, spec, buggy_code, category, severity))

    return items


def summarize(items: list) -> dict:
    """Counts by category/severity/clean-vs-buggy, for the notebook's dataset section."""
    from collections import Counter
    return {
        "n_items": len(items),
        "n_buggy": sum(1 for i in items if i["ground_truth"]["has_bug"]),
        "n_clean": sum(1 for i in items if not i["ground_truth"]["has_bug"]),
        "by_category": dict(Counter(i["ground_truth"]["category"] for i in items if i["ground_truth"]["has_bug"])),
        "by_severity": dict(Counter(i["ground_truth"]["severity"] for i in items if i["ground_truth"]["has_bug"])),
    }
