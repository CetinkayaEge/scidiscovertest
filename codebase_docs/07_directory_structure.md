# Directory Structure

Full project directory tree with descriptions.

```
scidiscovertest/
│
├── app.py                          Streamlit web UI
├── requirements.txt                Python dependencies
├── README.md                       Project README
│
├── configs/
│   └── demo.yaml                   Main pipeline configuration (single source of truth)
│
├── prompts/
│   ├── summarizer.txt              System prompt for SummarizerAgent
│   ├── synthesizer.txt             System prompt for SynthesizerAgent
│   └── verifier.txt                System prompt for VerifierAgent
│
├── scidiscover/                    Main Python package
│   ├── __init__.py
│   ├── run_demo.py                 CLI pipeline orchestrator (ingestion→chunking→indexing)
│   │
│   ├── ingest/
│   │   ├── pmc_ingest.py           PMC OA + OAI-PMH ingestion
│   │   └── openalex_ingest.py      OpenAlex Works API ingestion
│   │
│   ├── process/
│   │   └── chunker.py              Token-window chunker (title prepended to abstract)
│   │
│   ├── index/
│   │   └── builder.py              FAISS index builder (embed + index chunks)
│   │
│   ├── agents/
│   │   ├── retriever.py            Core FAISS retrieval (Retriever class)
│   │   ├── retriever_agent.py      RetrieverAgent wrapper
│   │   ├── reranker.py             RerankerAgent (CrossEncoder, optional)
│   │   ├── summarizer.py           SummarizerAgent (parallel LLM calls)
│   │   ├── synthesizer.py          SynthesizerAgent (JSON-mode LLM)
│   │   └── verifier.py             VerifierAgent (claim verification, optional)
│   │
│   └── eval/
│       └── run.py                  Evaluation runner (custom metrics + RAGAS scoring)
│
├── utils/
│   ├── llm_client.py               Multi-provider LLM router for pipeline (Anthropic/Gemini/local)
│   ├── eval_llm.py                 LLM factory for evaluation (OpenRouter or Gemini via LangChain)
│   └── schemas.py                  Data validation schemas
│
│
├── embeddings/
│   └── chunks.npy                  float32 embedding matrix, shape (num_chunks, 384)
│
├── index/
│   ├── faiss.index                 FAISS FlatIP binary index
│   ├── chunk_ids.txt               One chunk ID per line, order matches index rows
│   └── index.meta.json             Model name, dim, chunk count, chunk policy
│
├── docs/
│   └── corpus_manifest.csv         Paper provenance (paper_id, source, doi, date, license)
│
├── codebase_docs/                  Codebase documentation (this directory)
│   ├── README.md
│   ├── 01_overview.md
│   ├── 02_data_collection.md
│   ├── 03_processing_pipeline.md
│   ├── 04_indexing.md
│   ├── 05_agents_and_rag.md
│   ├── 06_configuration.md
│   └── 07_directory_structure.md
│
├── logs/                           Runtime traces (created on first pipeline run)
│   ├── config_snapshot.yaml        Copy of demo.yaml at run time
│   ├── retrieval_traces.jsonl
│   ├── summarizer_traces.jsonl
│   ├── synthesizer_traces.jsonl
│   └── verifier_traces.jsonl
│
├── outputs/                        Query result artifacts
│   ├── paper_summaries.json
│   ├── synthesis_result.json
│   ├── verification.jsonl
│   └── answers.jsonl
│
├── eval/                           RAGAS-based evaluation test set
│   ├── generate_ragas_testset.py   RAGAS TestsetGenerator + unanswerable + out-of-domain queries
│   ├── build_labels.py             Matches reference_contexts → expected_paper_ids + chunk_ids
│   ├── queries.jsonl               Generated queries (query_id, query, reference, reference_contexts,
│   │                               domain, difficulty, query_type, expected_paper_ids, expected_chunk_ids)
│   └── labels.jsonl                Retrieval ground truth (query_id, expected_paper_ids, expected_chunk_ids)
│
├── data/
│   ├── raw/
│   │   └── papers.jsonl            Raw ingested papers
│   └── processed/
│       └── chunks.jsonl            Chunked papers
│
├── reports/                        Evaluation results
│   ├── eval_results.json           Latest run output
│   └── eval_*.json                 Named ablation runs (full_gemini, no_verifier_hpc, etc.)
│
└── tests/                          Test suite
```

---

## Notes on Data Flow

- Ingestion writes to `data/raw/papers.jsonl`.
- The chunker reads from `data/raw/papers.jsonl` and writes to `data/processed/chunks.jsonl`.
- The retriever reads paper URLs and DOIs from `data/raw/papers.jsonl` — no copy step needed.
- All directories listed above are mandatory — the pipeline expects them to exist even when empty.
