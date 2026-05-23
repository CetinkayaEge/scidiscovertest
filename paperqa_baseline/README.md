# PaperQA2 SOTA Baseline

This sub-package integrates [PaperQA2](https://github.com/Future-House/paper-qa) as a
state-of-the-art RAG baseline and produces a side-by-side metric comparison against the
SciDiscover multi-agent pipeline.

---

## What is PaperQA2?

PaperQA2 (Future House, 2024) is an agent-based RAG system designed specifically for
scientific literature.  It uses an iterative search-and-summarise agent loop, OpenAI
embeddings for retrieval, and GPT-4o for synthesis.  It has been benchmarked on
*LitQA2* and outperforms GPT-4o with naive RAG.

We use it as a **SOTA ceiling** — a fully-automated, production-quality system run
against the same corpus and queries as our pipeline.

---

## Directory layout

```
paperqa_baseline/
  env_utils.py          # context manager that swaps OPENAI_* env vars
  run_paperqa.py        # main runner — exports papers, runs PaperQA2, saves JSON
  compare_results.py    # loads both result files, prints Markdown comparison table
  run_paperqa_hpc.sh    # SLURM submission script for the HPC cluster
  README.md             # this file
  results/              # created at runtime
    paperqa_results.json       ← PaperQA2 per-query outputs
    comparison_table.md        ← side-by-side Markdown table
  paper_texts/          # created at runtime (one .txt per paper, ~22 K files)
  pqa_index/            # created at runtime (PaperQA2 vector index cache)
```

---

## Setup

### 1. Install PaperQA2

```bash
# from the project root, with the venv active:
pip install paper-qa
```

PaperQA2 requires **Python ≥ 3.11** and pulls in LiteLLM, tiktoken, and pydantic ≥ 2.

> **Conflict check:** PaperQA2 requires `pydantic~=2.0,>=2.10.1`.  Our pipeline already
> pins `pydantic>=2.0`, so there is no conflict.  If you see version errors, upgrade:
> `pip install "pydantic>=2.10.1"`.

### 2. Add your real OpenAI API key

In your `.env` file (never commit the real key!), add:

```bash
# Used ONLY by PaperQA2 — never for the main pipeline
OPENAI_API_KEY_SOTA=sk-<your-real-openai-key>
```

The `run_paperqa.py` script will temporarily swap `OPENAI_API_KEY` →
`OPENAI_API_KEY_SOTA` and remove `OPENAI_BASE_URL` for the duration of the
PaperQA2 run, restoring both on exit.

---

## Running locally (dry-run first)

```bash
# Quick sanity check — 2 queries, ~2 reference papers, < 1 minute:
PYTHONPATH=. python paperqa_baseline/run_paperqa.py --dry-run

# Full run — 40 queries, ~22 K papers:
PYTHONPATH=. python paperqa_baseline/run_paperqa.py
```

After `run_paperqa.py` completes, generate the comparison table:

```bash
# Uses reports/eval_results.json (SciDiscover) and
#     paperqa_baseline/results/paperqa_results.json (PaperQA2)
PYTHONPATH=. python paperqa_baseline/compare_results.py

# Specify a particular SciDiscover report:
PYTHONPATH=. python paperqa_baseline/compare_results.py \
    --our-results reports/eval_full.json
```

---

## Running on HPC (SLURM)

```bash
sbatch paperqa_baseline/run_paperqa_hpc.sh
```

The script:
1. Sources `.env` (which must contain `OPENAI_API_KEY_SOTA`).
2. Installs `paper-qa` into the existing `.venv` if missing.
3. Runs `run_paperqa.py` (full 40-query run).
4. Runs `compare_results.py` immediately after.

Logs → `/cta/users/sait.kacmaz/logs/scidiscover-paperqa-<jobid>.out`

Estimated runtime: **20–40 min** on first run (embedding ~22 K papers via
OpenAI API); **10–15 min** on repeat runs (index cached in `pqa_index/`).

---

## Estimated OpenAI API cost

| Step | Model | Tokens | Approx. cost |
|------|-------|--------|-------------|
| Indexing (~22 K papers × ~250 tokens) | text-embedding-3-small | ~5.5 M | < $0.12 |
| 40 queries × synthesis | gpt-4o-mini | ~80 K | < $0.05 |
| **Total** | | | **< $0.20** |

The embedding index is cached; re-running queries only incurs the gpt-4o-mini cost.

---

## Metrics compared

| Metric | SciDiscover | PaperQA2 |
|--------|:-----------:|:--------:|
| Abstention rate (answerable) | ✓ | ✓ |
| Correct abstention rate (unanswerable) | ✓ | ✓ |
| Avg claims / citations per query | ✓ | ✓ |
| Avg latency (s) | ✓ | ✓ |
| ROUGE-L vs reference | ✓ | ✓ |
| Citation coverage | ✓ | — |
| Claim support rate (VerifierAgent) | ✓ | — |
| Retrieval recall@k | ✓ | — |
| Synthesizer hallucination rate | ✓ | — |
| RAGAS Faithfulness / Relevancy / Recall | ✓ | — |

**"—"** means the metric cannot be computed from PaperQA2 output because PaperQA2
does not expose per-claim citation coverage, NLI verification labels, or chunk-level
retrieval statistics in a structured format.

---

## How to read `comparison_table.md`

- Open `paperqa_baseline/results/comparison_table.md` in any Markdown viewer.
- The table lists both systems side-by-side.  "—" in the PaperQA2 column means
  the metric is SciDiscover-specific.
- The "Notes on LLM fairness" section explains the model difference (Qwen2.5 vs
  gpt-4o-mini) that limits strict comparability.

---

## Assumptions a teammate should verify before the HPC run

1. **`OPENAI_API_KEY_SOTA` is a real, funded OpenAI key** with access to
   `gpt-4o-mini` and `text-embedding-3-small`.
2. **`data/raw/papers.jsonl` is complete** on the HPC node.  If the corpus was
   built locally and not synced, the paper_texts/ directory will be empty.
3. **Python ≥ 3.11** is available in the HPC environment (`python3 --version`).
4. **Outbound HTTPS to `api.openai.com`** is not blocked by the cluster firewall.
   PaperQA2 makes direct HTTPS calls to the OpenAI API.
5. **Disk space**: ~22 K .txt files (≈ 50 MB) + PaperQA2 index (≈ 200 MB).
   Ensure the project directory has ≥ 500 MB free.
6. **`reports/eval_results.json` exists** on the HPC node for the comparison step.
   If it does not, `compare_results.py` will still produce a PaperQA2-only table
   with a warning.
