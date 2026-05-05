# SciDiscover: Project Overview

## What It Is

SciDiscover is a multi-agent scientific discovery system. Given a research query, it retrieves relevant open-access papers, summarizes each one, synthesizes a coherent evidence-grounded answer, and optionally verifies every claim against the source chunks. All outputs include traceable citations back to specific text chunks.

The corpus currently holds ~5,788 papers from two sources: PubMed Central (PMC) and OpenAlex.

---

## End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  OFFLINE PIPELINE  (run_demo.py)                                 │
│                                                                  │
│  PMC OA API ──────┐                                              │
│                   ├──► data/raw/papers.jsonl                     │
│  OpenAlex API ────┘         │                                    │
│                             ▼                                    │
│                      chunker.py                                  │
│                             │                                    │
│                             ▼                                    │
│                   datav2/processed/chunks.jsonl                  │
│                             │                                    │
│                             ▼                                    │
│                       builder.py                                 │
│                  (SentenceTransformer embed)                     │
│                             │                                    │
│              ┌──────────────┼───────────────┐                   │
│              ▼              ▼               ▼                   │
│    embeddingsv2/    indexv2/faiss.index  indexv2/chunk_ids.txt  │
│    chunks.npy                                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  ONLINE PIPELINE  (app.py / Streamlit)                           │
│                                                                  │
│  User Query                                                      │
│       │                                                          │
│       ▼                                                          │
│  RetrieverAgent  ──── FAISS search ──► top-k chunks (k=20)      │
│       │                                                          │
│       ▼ (optional)                                               │
│  RerankerAgent   ──── CrossEncoder ──► reranked top-k chunks     │
│       │                                                          │
│       ▼                                                          │
│  SummarizerAgent ──── LLM (parallel, 4 workers) ──► summaries   │
│       │                                                          │
│       ▼                                                          │
│  SynthesizerAgent ─── LLM (JSON mode) ──► draft answer + claims │
│       │                                                          │
│       ▼ (optional)                                               │
│  VerifierAgent   ──── LLM (JSON mode) ──► verified claims        │
│       │                                                          │
│       ▼                                                          │
│  Final Answer → Streamlit UI                                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Entry Points

| Entry Point | Purpose |
|-------------|---------|
| `scidiscover/run_demo.py` | CLI orchestrator: ingestion → chunking → indexing |
| `app.py` | Streamlit web UI for running queries |
| `scidiscover/ingest/pmc_ingest.py` | Standalone PMC ingestion |
| `scidiscover/ingest/openalex_ingest.py` | Standalone OpenAlex ingestion |
| `scidiscover/process/chunker.py` | Standalone chunking |
| `scidiscover/index/builder.py` | Standalone FAISS index build |

### CLI Usage

```bash
# Full pipeline
python -m scidiscover.run_demo --config configs/demo.yaml

# Skip steps already done
python -m scidiscover.run_demo --config configs/demo.yaml --skip-ingestion
python -m scidiscover.run_demo --config configs/demo.yaml --skip-ingestion --skip-chunking

# Launch UI
streamlit run app.py
```

---

## Corpus Scale (current)

| Metric | Value |
|--------|-------|
| Total papers | ~5,788 |
| PMC papers | ~3,000 |
| OpenAlex papers | ~2,788 |
| Total chunks (v2) | 10,219 |
| Total chunks (v1) | 15,294 |
| Embedding dimensions | 384 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |

---

## Active Data Version

The project runs on **datav2** throughout the online pipeline. All config paths in `configs/demo.yaml` point to `datav2/`, `embeddingsv2/`, and `indexv2/`. See [03_processing_pipeline.md](03_processing_pipeline.md) for the difference between v1 and v2.
