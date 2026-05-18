# Configuration Reference: configs/demo.yaml

Full annotated walkthrough of every section in `configs/demo.yaml`.

---

## LLM

```yaml
llm:
  model: openrouter-openai/gpt-5.4-mini    # Active provider+model. Prefix determines provider:
                                            #   claude-*       → Anthropic
                                            #   gemini-*       → Google Gemini
                                            #   openrouter-*   → OpenRouter (OpenAI-compatible)
                                            #   local-*        → OpenAI-compatible local server
  max_tokens: 4096                          # Max output tokens per LLM call
```

The provider prefix is stripped before the model name is sent to the relevant API. For `openrouter-openai/gpt-5.4-mini` the model ID `openai/gpt-5.4-mini` is sent to `https://openrouter.ai/api/v1`. For `local-qwen2.5-72b-instruct-gptq` the model ID `qwen2.5-72b-instruct-gptq` is sent to `OPENAI_BASE_URL` and must match what the vLLM server reports at `/v1/models`.

**Important:** `max_tokens` is the *completion* cap, not the total context. The vLLM server's total context window is 8192 tokens, so `prompt_tokens + max_tokens ≤ 8192`. With `max_tokens: 4096` you have 4096 tokens for the prompt. Setting `max_tokens` close to or above 8192 will cause `BadRequestError: maximum context length exceeded` errors.

**Note on JSON mode:** Local models do not enforce structured output natively. The Synthesizer and Verifier rely on prompt instructions alone — Qwen 2.5 generally follows JSON instructions reliably but may occasionally produce malformed output.

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
  enabled: true                                          # Toggle in UI sidebar too
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 10                                              # Chunks kept after reranking
  min_score: -.inf                                       # No score threshold — keep all top_k
  traces_output: logs/reranker_traces.jsonl
```

`min_score: -.inf` disables score-based filtering entirely — only `top_k` limits the output. Setting a positive threshold (e.g. `0.0`) can cause all chunks to be dropped on low-relevance queries.

**Note on `eval.top_k_recall` vs `retrieval.top_k`:** The eval pipeline (`scidiscover/eval/run.py`) uses `eval.top_k_recall` as the retriever k — `retrieval.top_k` is read only by `scidiscover/run_demo.py` and `tests/demo_query.py`. For a meaningful reranker ablation in eval, pass `--retrieval-k 30` (wider pool) so the reranker has chunks to filter from before reducing to `reranker.top_k`.

---

## Query Transformer (Optional)

Pre-retrieval query transformations. Mutually exclusive — pick one strategy.

```yaml
query_transformer:
  strategy: none           # none | decompose | hyde
  decomposer:
    prompt_path: prompts/query_decomposer.txt
    traces_output: logs/query_decomposer_traces.jsonl
    max_sub_queries: 3
  hyde:
    prompt_path: prompts/hyde.txt
    traces_output: logs/hyde_traces.jsonl
```

| Strategy | Behavior |
|----------|----------|
| `none` | Raw query goes directly to retriever (default) |
| `decompose` | LLM splits query into ≤`max_sub_queries` sub-queries; each retrieves top-k chunks; results merged by `chunk_id` keeping highest score |
| `hyde` | LLM writes a hypothetical scientific abstract for the query; that abstract is embedded for retrieval; original query is restored on the resulting evidence pack so downstream agents see it |

CLI override: `--query-transformer {none,decompose,hyde}`.

---

## Summarizer

```yaml
summarizer:
  prompt_path: prompts/summarizer.txt
  traces_output: logs/summarizer_traces.jsonl
  output_path: outputs/paper_summaries.json
  max_workers: 1           # ThreadPoolExecutor workers.
                           #   1   = sequential (safest, default for local vLLM)
                           #   3–4 = OK for Gemini free tier (~10 RPM)
                           #   8+  = HPC-grade local vLLM
```

`max_workers` only affects parallelism for paper summarization within a single query — it does not change token usage per request. If the vLLM server can handle parallel requests, raising this is the simplest way to speed up evaluation runs.

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
  queries_path: eval/queries.jsonl         # Generated by eval/generate_ragas_testset.py
  results_output: reports/eval_results.json
  top_k_recall: 10                         # k for Recall@k metric
  max_queries: null                        # Set to an integer to cap queries (e.g. 5 for quick test)
  ragas_model: gemini-2.5-flash            # Model for RAGAS scoring + testset generation
                                           # OpenRouter: openai/gpt-5.4-mini, openai/gpt-4o-mini, etc.
                                           # Gemini:     gemini-2.5-flash
```

**Provider routing for `ragas_model`** (handled by `utils/eval_llm.py`):

| Model format | Provider | Required env var | JSON mode |
|---|---|---|---|
| `openai/*`, `anthropic/*`, `meta-llama/*`, any `/` | OpenRouter | `OPENROUTER_API_KEY` | `response_format: {"type": "json_object"}` |
| `gemini-*` | Google Gemini | `GOOGLE_API_KEY` | `response_mime_type: "application/json"` |

`ragas_model` is independent of `llm.model` — it applies only to RAGAS metric scoring and testset generation, not the RAG pipeline itself.

**OpenRouter limitation (`n=1` only):** OpenRouter does not support `n>1` for chat completions ([known issue](https://github.com/OpenRouterTeam/openrouter-runner/issues/99)). RAGAS requests `n=3` for stability — when going through OpenRouter, only 1 generation is returned and you'll see "LLM returned 1 generations instead of requested 3" warnings. Scores remain valid but slightly less stable. Use a Gemini model (`gemini-2.5-flash`) for full `n=3` support via `candidate_count`.

---

## Levers for Increasing Paper Count

| What to change | Parameter | File |
|----------------|-----------|------|
| More PMC papers | `sources.pmc.max_papers` | configs/demo.yaml |
| Wider PMC date range | `sources.pmc.from_date` / `until_date` | configs/demo.yaml |
| More OpenAlex papers | `sources.openalex.max_papers` | configs/demo.yaml |
| Earlier OpenAlex papers | `sources.openalex.from_year` | configs/demo.yaml |
| More OpenAlex topics | Add entries to `sources.openalex.queries` (sustainability/healthcare only) | configs/demo.yaml |
