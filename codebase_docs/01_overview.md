# SciDiscover: Project Overview

## What It Is

SciDiscover is a multi-agent scientific discovery system. Given a research query, it retrieves relevant open-access papers, summarizes each one, synthesizes a coherent evidence-grounded answer, and optionally verifies every claim against the source chunks. All outputs include traceable citations back to specific text chunks.

The corpus collects papers from up to three sources — PubMed Central (PMC), OpenAlex, and Web of Science (WOS) — restricted to the **sustainability** and **healthcare** domains.

---

## End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  OFFLINE PIPELINE  (run_demo.py)                                 │
│                                                                  │
│  PMC OA API ──────┐                                              │
│                   ├──► data/raw/papers.jsonl                     │
│  OpenAlex API ────┤         │                                    │
│  WOS API ─────────┘         │                                    │
│                             ▼                                    │
│                      chunker.py                                  │
│                             │                                    │
│                             ▼                                    │
│                   data/processed/chunks.jsonl                    │
│                             │                                    │
│                             ▼                                    │
│                       builder.py                                 │
│                  (SentenceTransformer embed)                     │
│                             │                                    │
│              ┌──────────────┼───────────────┐                   │
│              ▼              ▼               ▼                   │
│    embeddings/      index/faiss.index   index/chunk_ids.txt     │
│    chunks.npy                                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  ONLINE PIPELINE  (app.py / Streamlit, scidiscover/eval/run.py)  │
│                                                                  │
│  User Query                                                      │
│       │                                                          │
│       ▼ (optional, mutually exclusive)                           │
│  QueryDecomposerAgent ─► sub-queries (retrieve each, merge)     │
│  HyDEAgent            ─► hypothetical abstract (used for embed) │
│       │                                                          │
│       ▼                                                          │
│  RetrieverAgent  ──── FAISS search ──► top-k chunks             │
│       │                                                          │
│       ▼ (optional)                                               │
│  RerankerAgent   ──── CrossEncoder ──► reranked top-k chunks     │
│       │                                                          │
│       ▼                                                          │
│  SummarizerAgent ──── LLM (parallel, 4 workers) ──► summaries   │
│       │   │                                                      │
│       │   └─► (alt: SingleAgentBaseline replaces Sum+Syn+Ver)   │
│       │                                                          │
│       ▼                                                          │
│  SynthesizerAgent ─── LLM (JSON mode) ──► draft answer + claims │
│       │                                                          │
│       ▼ (optional, mutually exclusive with critique loop)        │
│  VerifierAgent   ──── LLM (JSON mode) ──► verified claims        │
│       │                                                          │
│       ▼ (optional, replaces standalone Verifier)                 │
│  CritiqueLoopAgent ─► iterative Synthesizer→Verifier refinement │
│       │                                                          │
│       ▼                                                          │
│  Final Answer → Streamlit UI / reports/eval_*.json              │
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
| `scidiscover/ingest/wos_ingest.py` | Standalone Web of Science ingestion (requires `WOS_API_KEY`) |
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

## Corpus Scale

| Metric | Value |
|--------|-------|
| Target papers (PMC) | up to 8,000 (2023–2025) |
| Target papers (OpenAlex) | up to 15,000 (2021+, 16 queries) |
| Target papers (WOS) | up to 5,000 (2020–2024, 16 queries; requires API key) |
| Embedding dimensions | 384 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |

See `docs/corpus_manifest.csv` for the exact current paper count after each ingestion run.
