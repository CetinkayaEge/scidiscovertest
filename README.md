# SciDiscover

Multi-agent scientific discovery system with evidence-grounded reasoning over open-access research corpora in **sustainability** and **healthcare**.

## Setup

```bash
# 1. Clone and enter project
git clone <repo-url>
cd scidiscovertest

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root with the relevant API key(s):

```bash
ANTHROPIC_API_KEY=sk-ant-...   # for claude-* models
GOOGLE_API_KEY=AIza...         # for gemini-* models

# Local model served via OpenAI-compatible server (e.g. vLLM)
OPENAI_BASE_URL=https://<your-endpoint>/v1
OPENAI_API_KEY=<your-key>
```

## LLM Configuration

Set the active model in `configs/demo.yaml`:

```yaml
llm:
  model: gemini-2.5-flash   # or claude-*, local-*
  max_tokens: 4096
```

| Prefix | Provider | Required env var |
|--------|----------|-----------------|
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini-*` | Google Gemini | `GOOGLE_API_KEY` |
| `local-*` | OpenAI-compatible local server | `OPENAI_BASE_URL` + `OPENAI_API_KEY` |

## Run the Pipeline

```bash
# Full pipeline: ingest → chunk → embed → index
.venv/bin/python -m scidiscover.run_demo --config configs/demo.yaml

# Skip steps already completed
.venv/bin/python -m scidiscover.run_demo --config configs/demo.yaml --skip-ingestion
.venv/bin/python -m scidiscover.run_demo --config configs/demo.yaml --skip-ingestion --skip-chunking
```

### Run individual steps

```bash
# PMC ingestion only
.venv/bin/python -m scidiscover.ingest.pmc_ingest \
    --from-date 2023-01-01 --until-date 2025-12-31 \
    --max-papers 8000 --skip-empty-abstract \
    --raw-output data/raw/papers.jsonl \
    --manifest-output docs/corpus_manifest.csv

# OpenAlex ingestion (appends to existing papers.jsonl)
.venv/bin/python -m scidiscover.ingest.openalex_ingest \
    --queries "sustainability" "healthcare" "climate change" \
    --from-year 2020 --max-papers 15000 \
    --email your@email.com \
    --raw-output data/raw/papers.jsonl \
    --manifest-output docs/corpus_manifest.csv \
    --is-oa --has-abstract

# Chunking only
.venv/bin/python -m scidiscover.process.chunker --config configs/demo.yaml
```

## Launch the UI

```bash
streamlit run app.py
```

## Pipeline Overview

| Step | Module | Input | Output |
|------|--------|-------|--------|
| Ingest | `scidiscover.ingest.pmc_ingest` / `openalex_ingest` | PMC + OpenAlex APIs | `data/raw/papers.jsonl`, `docs/corpus_manifest.csv` |
| Chunk | `scidiscover.process.chunker` | `data/raw/papers.jsonl` | `data/processed/chunks.jsonl` |
| Index | `scidiscover.index.builder` | `data/processed/chunks.jsonl` | `embeddings/chunks.npy`, `index/faiss.index`, `index/chunk_ids.txt`, `index/index.meta.json` |

## Corpus

Papers are collected from two open-access sources, restricted to the **sustainability** and **healthcare** domains:

- **PMC Open Access** — 2023–2025, up to 8,000 papers via NCBI OA API + OAI-PMH
- **OpenAlex** — 2021+, up to 15,000 papers across 16 topic queries

Current corpus: **~21,967 papers**, **37,415 chunks**.

> All OpenAlex queries must remain within the sustainability or healthcare domains. Do not add queries from unrelated fields.

## Configuration

All pipeline parameters live in `configs/demo.yaml`. Key sections:

| Section | Key parameters |
|---------|---------------|
| `llm` | `model`, `max_tokens` |
| `corpus` | Output paths for `papers.jsonl` and manifest |
| `chunking` | `chunk_size` (200 tokens), `overlap` (40 tokens) |
| `retrieval` | Embedding model, `top_k`, output paths |
| `sources.pmc` | `from_date`, `until_date`, `max_papers` |
| `sources.openalex` | `queries`, `from_year`, `max_papers` |

## Output Schemas

**`data/raw/papers.jsonl`** — one JSON object per line:
```
paper_id, title, abstract, year, authors, venue, doi, url, source, retrieved_date, license_note
```

**`data/processed/chunks.jsonl`** — one JSON object per line:
```
chunk_id, paper_id, section, text, token_len
```
Chunk text format: `title | abstract text`. Chunk ID format: `paper_id||SECTION||offset`.

**`docs/corpus_manifest.csv`**:
```
paper_id, source, url/doi, retrieved_date, license_note
```

## Verification

```bash
wc -l data/raw/papers.jsonl          # total papers
wc -l data/processed/chunks.jsonl    # total chunks
cat index/index.meta.json            # index stats
wc -l docs/corpus_manifest.csv       # manifest rows (includes header)
```

## Evaluation

Generate a fresh eval testset from the current corpus, then run evaluation:

```bash
# Generate queries + labels from current chunks
.venv/bin/python -m scidiscover.eval.generate_testset --config configs/demo.yaml --n-queries 30

# Run evaluation (RAGAS + custom metrics)
.venv/bin/python -m scidiscover.eval.run --config configs/demo.yaml
```

## SOTA Baseline Comparison

We compare SciDiscover against **PaperQA2** (Future House, 2024) — a production-quality,
agent-based scientific RAG system — using the same corpus and benchmark queries.

See [`paperqa_baseline/`](paperqa_baseline/) for full setup and run instructions.

**Quick start:**

```bash
# 1. Install PaperQA2
pip install paper-qa

# 2. Add OPENAI_API_KEY_SOTA=sk-<your-key> to .env

# 3. Dry-run (2 queries, ~1 min)
PYTHONPATH=. python paperqa_baseline/run_paperqa.py --dry-run

# 4. Full run (40 queries, ~30 min first time)
PYTHONPATH=. python paperqa_baseline/run_paperqa.py

# 5. Generate comparison table
PYTHONPATH=. python paperqa_baseline/compare_results.py
```

Results are saved to `paperqa_baseline/results/` and a Markdown comparison table is
written to `paperqa_baseline/results/comparison_table.md`.

| Metric | Compared? |
|--------|:---------:|
| Abstention rate (answerable / unanswerable) | ✓ both |
| Avg citations per query | ✓ both |
| Avg latency (s) | ✓ both |
| ROUGE-L vs reference | ✓ both |
| Citation coverage / support rate / recall@k | SciDiscover only |
| RAGAS faithfulness / relevancy / recall | SciDiscover only |
