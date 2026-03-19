# SciDiscover-LLM

Multi-agent scientific discovery system with evidence-grounded reasoning over open-access research corpora in sustainability and healthcare.

## Setup

```bash
# 1. Clone and enter project
git clone <repo-url>
cd Sci-Discovery-pubmed

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Run the Pipeline

Run the full end-to-end pipeline (Steps 1-3) with a single command:

```bash
python -m scidiscover.run_demo --config configs/demo.yaml
```

### Skip individual steps

```bash
# Skip ingestion (use existing papers.jsonl)
python -m scidiscover.run_demo --config configs/demo.yaml --skip-ingestion

# Skip chunking
python -m scidiscover.run_demo --config configs/demo.yaml --skip-chunking

# Skip index building
python -m scidiscover.run_demo --config configs/demo.yaml --skip-index
```

### Run individual modules

```bash
# Step 1a: PMC ingestion only
python -m scidiscover.ingestion.pmc_ingest \
    --from-date 2025-01-01 --until-date 2025-12-31 \
    --max-papers 3000 --skip-empty-abstract \
    --raw-output data/raw/papers.jsonl \
    --manifest-output docs/corpus_manifest.csv

# Step 1b: OpenAlex ingestion (appends to existing papers.jsonl)
python -m scidiscover.ingestion.openalex_ingest \
    --queries "machine learning" "deep learning" \
    --from-year 2023 --max-papers 500 \
    --email your@email.com \
    --raw-output data/raw/papers.jsonl \
    --manifest-output docs/corpus_manifest.csv

# Step 2: Chunking
python -m scidiscover.chunking.chunker

# Step 3: Query after index is built
python -m scidiscover.retrieval.demo_query \
    --config configs/demo.yaml \
    --query "machine learning in healthcare" --top-k 5
```

## Configuration

All pipeline parameters are in `configs/demo.yaml`:

| Section | Key parameters |
|---|---|
| `corpus` | Output paths for papers.jsonl and manifest |
| `chunking` | chunk_size (200), overlap (40) |
| `retrieval` | Model name, top_k, output paths |
| `sources.pmc` | Date range, max_papers, filters |
| `sources.openalex` | Queries, from_year, max_papers, email |

## Pipeline Steps

| Step | Module | Input | Output |
|---|---|---|---|
| 1 | `scidiscover.ingestion` | API queries | `data/raw/papers.jsonl`, `docs/corpus_manifest.csv` |
| 2 | `scidiscover.chunking` | papers.jsonl | `data/processed/chunks.jsonl` |
| 3 | `scidiscover.retrieval` | chunks.jsonl | `embeddings/chunks.npy`, `index/faiss.index`, `index/chunk_ids.txt`, `index/index.meta.json` |

## Data Sources

The pipeline ingests from two open-access sources (configurable in demo.yaml):

- **PMC Open Access** — healthcare papers via NCBI OA API + OAI-PMH metadata
- **OpenAlex** — broad scholarly metadata via OpenAlex Works API

## Output Schemas

**papers.jsonl** — one JSON object per line:
```
paper_id, title, abstract, year, authors, venue, doi, url, source, retrieved_date, license_note
```

**chunks.jsonl** — one JSON object per line:
```
chunk_id, paper_id, section, text, token_len
```
Chunk IDs use `||` separator: `paper_id||section||offset`

**corpus_manifest.csv**:
```
paper_id, source, url/doi, retrieved_date, license_note
```

## Verification

After running the pipeline, check:

```bash
# Papers ingested
wc -l data/raw/papers.jsonl

# Chunks created
wc -l data/processed/chunks.jsonl

# Index metadata
cat index/index.meta.json

# Manifest
head docs/corpus_manifest.csv

# Run a test query
python -m scidiscover.retrieval.demo_query \
    --config configs/demo.yaml \
    --query "climate change impact" --top-k 3
```
