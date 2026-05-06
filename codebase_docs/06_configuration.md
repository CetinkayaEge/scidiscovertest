# Configuration Reference: configs/demo.yaml

Full annotated walkthrough of every section in `configs/demo.yaml`.

---

## LLM

```yaml
llm:
  model: gemini-2.5-flash     # Active provider+model. Prefix determines provider:
                               #   claude-*  → Anthropic
                               #   gemini-*  → Google Gemini
                               #   local-*   → OpenAI-compatible local server
  max_tokens: 4096             # Max output tokens per LLM call
```

---

## Corpus (Raw Data Paths)

```yaml
corpus:
  raw_output: data/raw/papers.jsonl         # Where both ingestors write papers
  manifest_output: docs/corpus_manifest.csv  # Provenance CSV
```

---

## Chunking

```yaml
chunking:
  input_path: data/raw/papers.jsonl          # Source papers
  output_path: data/processed/chunks.jsonl   # Chunked output
  chunk_size: 200                             # Tokens per chunk (whitespace-tokenized)
  overlap: 40                                 # Overlapping tokens between consecutive chunks
```

---

## Retrieval (FAISS Index)

```yaml
retrieval:
  chunks_input: data/processed/chunks.jsonl       # Chunks to embed and index
  papers_input: data/raw/papers.jsonl             # Used to look up url/doi at retrieval time
  embeddings_output: embeddings/chunks.npy        # Saved embedding matrix (float32)
  index_output: index/faiss.index                 # FAISS FlatIP binary index
  chunk_ids_output: index/chunk_ids.txt           # Maps index row → chunk_id
  meta_output: index/index.meta.json              # Model name, dims, chunk count
  model_name: sentence-transformers/all-MiniLM-L6-v2  # Embedding model
  top_k: 20                                       # Chunks returned per query
```

**Note:** Both ingestion and retrieval use `data/raw/papers.jsonl` — no copy step needed.

---

## Reranker (Optional)

```yaml
reranker:
  enabled: false                                          # Toggle in UI sidebar too
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 10                                              # Chunks kept after reranking
  min_score: -10.0                                       # Score threshold (logit scale)
  traces_output: logs/reranker_traces.jsonl
```

---

## Summarizer

```yaml
summarizer:
  prompt_path: prompts/summarizer.txt
  traces_output: logs/summarizer_traces.jsonl
  output_path: outputs/paper_summaries.json
  max_workers: 4           # ThreadPoolExecutor workers. Keep ≤3 for Gemini free tier (~10 RPM)
```

---

## Synthesizer

```yaml
synthesizer:
  prompt_path: prompts/synthesizer.txt
  traces_output: logs/synthesizer_traces.jsonl
  output_path: outputs/synthesis_result.json
```

---

## Verifier

```yaml
verifier:
  prompt_path: prompts/verifier.txt
  traces_output: logs/verifier_traces.jsonl
  output_path: outputs/verification.jsonl
  answers_output: outputs/answers.jsonl
  min_citation_coverage: 0.60    # Fraction of claims that must have ≥1 citation
  min_support_rate: 0.40         # Fraction of cited claims that must be SUPPORTED
```

If either threshold is not met, the pipeline returns "Insufficient evidence…" instead of the draft answer.

---

## Sources: PMC

```yaml
sources:
  pmc:
    enabled: true
    from_date: "2023-01-01"          # ISO date, start of PMC OA date range
    until_date: "2025-12-31"         # ISO date, end of PMC OA date range
    max_papers: 8000                 # Hard cap on papers fetched from PMC
    skip_empty_abstract: true        # Discard papers with no abstract
```

---

## Sources: OpenAlex

> **Domain constraint:** queries must belong to either the **sustainability** or **healthcare** domain only. Do not add queries outside these two domains.

> **Year filter note:** The API filter is `publication_year > from_year` (strict inequality). `from_year: 2020` returns papers from 2021 onward.

> **Cap behavior:** `max_papers` is a shared counter across all queries. If it is reached on an early query, remaining queries never run. Always set it high enough to let all queries contribute.

```yaml
  openalex:
    enabled: true
    queries:                         # Each query runs cursor-paginated separately
      - "sustainability"
      - "healthcare"
      - "renewable energy"
      - "climate change"
      - "public health"
      - "environmental science"
      - "circular economy"
      - "carbon emissions"
      - "sustainable agriculture"
      - "biodiversity conservation"
      - "mental health"
      - "infectious disease"
      - "chronic disease"
      - "digital health"
      - "global health equity"
      - "cancer research"
    from_year: 2020                  # Papers from 2021+ (strict >)
    max_papers: 15000                # Hard cap across all queries combined
    is_oa: true                      # Open-access only
    has_abstract: true               # Must have abstract
```

---

## Evaluation

```yaml
eval:
  queries_path: eval/queries.jsonl
  labels_path: eval/labels.jsonl
  ragas_model: gemini-2.5-flash    # Model used for RAGAS metric computation
  output_path: reports/eval_results.json
```

---

## Levers for Increasing Paper Count

| What to change | Parameter | File |
|----------------|-----------|------|
| More PMC papers | `sources.pmc.max_papers` | configs/demo.yaml |
| Wider PMC date range | `sources.pmc.from_date` / `until_date` | configs/demo.yaml |
| More OpenAlex papers | `sources.openalex.max_papers` | configs/demo.yaml |
| Earlier OpenAlex papers | `sources.openalex.from_year` | configs/demo.yaml |
| More OpenAlex topics | Add entries to `sources.openalex.queries` (sustainability/healthcare only) | configs/demo.yaml |
