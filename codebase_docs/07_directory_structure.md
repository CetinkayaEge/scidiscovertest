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
│   │   └── chunker.py              Token-window chunker (produces datav2 chunks)
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
│       ├── run.py                  RAGAS evaluation runner
│       └── generate_testset.py     Test dataset generation
│
├── utils/
│   ├── llm_client.py               Multi-provider LLM router (Anthropic/Gemini/local)
│   └── schemas.py                  Data validation schemas
│
├── data/                           *** V1 data (inactive, kept for reference) ***
│   ├── raw/
│   │   └── papers.jsonl            Raw ingested papers (~12 MB, 5,788 papers)
│   └── processed/
│       └── chunks.jsonl            V1 chunks (15,294 — title as separate TITLE chunks)
│
├── datav2/                         *** V2 data (ACTIVE) ***
│   ├── raw/
│   │   └── (empty — papers.jsonl was deleted after chunking to save space)
│   └── processed/
│       └── chunks.jsonl            V2 chunks (9,506 — title prepended to abstract)
│
├── embeddings/                     V1 embeddings (inactive)
│   └── chunks.npy                  float32 array (15,294 × 384)
│
├── embeddingsv2/                   V2 embeddings (ACTIVE)
│   └── chunks.npy                  float32 array (9,506 × 384)
│
├── index/                          V1 FAISS index (inactive)
│   ├── faiss.index                 FAISS FlatIP index (15,294 chunks)
│   ├── chunk_ids.txt               15,294 chunk IDs
│   └── index.meta.json             {model, dim:384, num_chunks:15294, k:5}
│
├── indexv2/                        V2 FAISS index (ACTIVE)
│   ├── faiss.index                 FAISS FlatIP index (9,506 chunks)
│   ├── chunk_ids.txt               9,506 chunk IDs
│   └── index.meta.json             {model, dim:384, num_chunks:9506, k:20}
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
├── eval/                           Evaluation data
│   ├── queries.jsonl
│   └── labels.jsonl
│
├── reports/                        Evaluation results
│   └── eval_results.json
│
└── tests/                          Test suite
```

---

## Notes on Data Directories

- `data/` and `datav2/` both exist. Only `datav2/` is used by the active pipeline.
- `datav2/raw/papers.jsonl` was deleted (as shown in git history). The canonical raw papers file is `data/raw/papers.jsonl`.
- `index/` and `embeddings/` (V1) exist on disk but are not referenced in `configs/demo.yaml`.
- All of the above directories are mandatory — the pipeline expects them to exist.
