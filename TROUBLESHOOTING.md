# Colab run — issues hit and approach so far

Running log of problems found while getting `notebooks/run_on_colab.ipynb`
working end to end, and what was tried for each. Kept as a real record
(not cleaned up after the fact) since catching your own setup lying to you
is part of the point of building a reproducible pipeline.

## 1. LangGraph crash on V4: `AttributeError: module 'langchain' has no attribute 'debug'`

**Symptom:** `eval.run_v4_agent_optimizer` crashed inside
`langgraph/pregel/__init__.py` → `langchain_core/runnables/config.py`, in
`get_callback_manager_for_config`, trying to read `langchain.debug`.

**Root cause:** Colab's base image ships a `langchain` package that doesn't
match the `langchain-core`/`langgraph` versions this repo pins. `langchain_core`
has a legacy compatibility shim that does `import langchain; ... langchain.debug`
for old global debug/verbose flags, guarded only against `ImportError` — not
`AttributeError`. If a mismatched `langchain` is present (attribute removed
or relocated in that version), it crashes instead of falling through.

**Fix applied** (`requirements.txt`, `notebooks/run_on_colab.ipynb`):
- Pinned `langchain`, `langchain-core`, and `langgraph` together in the same
  generation (`langchain`/`langchain-core` `>=0.3,<0.4`, `langgraph` `>=0.2,<0.3`).
- Added a `pip uninstall -y -q langchain langchain-core langchain-community langgraph`
  step in the notebook's install cell, before installing the pinned set —
  so a preinstalled mismatched version can't linger and cause skew.

**Status:** verified locally (installed the pinned trio into an isolated
venv here — no GPU needed for this part — and ran
`eval.run_v4_agent_optimizer` end to end with no crash). Not yet confirmed
against a live Colab session for this specific error, since debugging moved
on to issue #2 before this could be re-checked live.

## 2. V1 baseline fails: `Torch not compiled with CUDA enabled`

**Symptom:** `eval.run_v1_baseline` loads the model, then fails immediately
with `Torch not compiled with CUDA enabled` — Colab's GPU runtime is
selected (`nvidia-smi` shows a T4), but `torch.cuda.is_available()` is
`False` by the time our code runs.

**Root cause:** installing `bitsandbytes` + `transformers` + `accelerate`
together (this project's lightweight stack, chosen to avoid vLLM's slow
install) appears to cause pip's dependency resolver to silently replace
Colab's pre-baked CUDA-enabled `torch` with a generic CPU-only wheel from
PyPI's default index. Exact trigger not fully isolated — one of those three
packages' transitive requirements is forcing a `torch` reinstall without an
explicit `--index-url` for a CUDA build.

**Fix attempt #1 (insufficient):** added a verify cell right after install —
`if not torch.cuda.is_available(): pip install --force-reinstall torch --index-url .../cu121`.
Confirmed with the user this was actually run *and* the runtime was
restarted afterward (required, since a live-imported `torch` module doesn't
hot-reload after pip replaces files on disk) — **still failed.** Conclusion:
the hardcoded `cu121` tag doesn't match this Colab image's actual CUDA
version, so the "fix" just installed a different, still-wrong build.

**Fix attempt #2 (current):** stopped guessing the CUDA tag entirely.
- Added a cell immediately after `nvidia-smi`, *before any pip installs run*,
  that snapshots `torch.__version__` (e.g. `"2.6.0+cu124"` — Colab's own
  known-good build) to `/content/_baseline_torch_spec.txt`.
- The post-install verify cell, if it finds `torch.cuda.is_available()` is
  `False`, reads that snapshot back and reinstalls the *exact* captured
  `torch==<version> --index-url .../<captured-cuda-tag>` instead of a
  hardcoded guess.
- Still requires a runtime restart after the reinstall (same reason as
  attempt #1) — the cell says so explicitly.

**Status:** confirmed fixed. V1's actual run: model loaded, full benchmark
ran on the real T4 (`tokens_per_sec: 69.5`, `vram_gb: 1.72`,
`gpu_util_pct: 41.0`, TTFT/TPOT/E2E all populated) — both this fix and the
langchain fix in #1 are working end to end. Ran into issue #3 next.

## 3. Perplexity step fails: `Invalid HF URI ... Repository id must be 'namespace/name', got 'wikitext'`

**Symptom:** after benchmarking succeeds, `src/evaluate.py::perplexity()`
fails calling `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")`.

**Root cause:** the bare `"wikitext"` dataset id is a legacy script-based
Hub dataset that's been deprecated/removed from resolution in current
`datasets` versions — it now has to be referenced by its namespaced,
Parquet-converted replacement.

**Fix applied** (`src/evaluate.py`): changed the dataset id to
`"Salesforce/wikitext"` (same subset/split args otherwise).

**Status:** pushed, not yet confirmed against a live Colab run.

## Caveat that cost a round-trip: notebook file vs. cloned repo are separate

`git pull` inside the cloned repo directory (from the notebook's `git clone`
cell) only updates `src/`/`eval/` — the `.ipynb` file itself, which is what's
actually open and running in the Colab UI, is a separate copy fetched once
when the notebook was opened. Pushing fixes to GitHub doesn't retroactively
update an already-open Colab tab; the notebook has to be re-opened from
GitHub (`File → Open notebook → GitHub`) or otherwise re-fetched to pick up
new cells. Worth remembering before assuming a push "didn't work" when it's
actually just not loaded yet.

## Open questions

- Is `bitsandbytes`, `transformers`, or `accelerate` (or a combination)
  actually the one forcing the torch reinstall? Not yet isolated —
  attempt #2 sidesteps the question by restoring whatever was there
  originally rather than needing to know.
- Once V1 runs clean end to end (benchmark + perplexity), V2's sweep
  includes `attn_implementation=flash_attention_2`,
  which is *expected* to fail on T4 (Turing has no FA2 support) — that's a
  planned negative result, not a bug, and should show up as an `error` field
  on that one row in `results/experiments.jsonl`, not a crash.
