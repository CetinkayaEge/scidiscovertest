# Agents and RAG

## Agent Pipeline Overview

Up to eight agents can run for each query. Query transformers, Reranker, Verifier, and CritiqueLoopAgent are optional and toggled via config or the UI sidebar. The single-agent baseline replaces the Summarizer + Synthesizer + Verifier chain with a single LLM call.

```
User Query
    │
    ▼ (optional, choose one)
QueryDecomposerAgent ─► sub-queries (retrieve each, merge by chunk_id)
HyDEAgent            ─► hypothetical abstract (used as retrieval string)
    │
    ▼
RetrieverAgent   ──► top-k chunks from FAISS
    │
    ▼ (optional)
RerankerAgent    ──► CrossEncoder reranked subset
    │
    ▼
SummarizerAgent  ──► per-paper summaries with chunk citations (parallel)
    │   │
    │   └─► (alternate: SingleAgentBaseline replaces Summarizer + Synthesizer + Verifier with one LLM call)
    │
    ▼
SynthesizerAgent ──► draft answer + key claims with citation IDs (JSON)
    │
    ├─► (optional) VerifierAgent ──► verified claims, final answer or abstention
    │
    └─► (optional) CritiqueLoopAgent ──► iterative Synthesizer→Verifier refinement
    │
    ▼
Streamlit UI / outputs/answers.jsonl
```

---

## 0. Query Transformer Agents (Optional)

Two pre-retrieval transformations are available. They are mutually exclusive — set `query_transformer.strategy` in `configs/demo.yaml` to `none` (default), `decompose`, or `hyde`. Override at the CLI with `--query-transformer {none,decompose,hyde}`.

### QueryDecomposerAgent

**File:** `scidiscover/agents/query_decomposer.py`
**Prompt:** `prompts/query_decomposer.txt`

Breaks a complex query into ≤`max_sub_queries` (default 3) simpler sub-queries via JSON-mode LLM call. The pipeline calls `RetrieverAgent.run()` for each sub-query and merges results by `chunk_id`, keeping the highest score per chunk. Original query is preserved in the resulting `evidence_pack` so downstream agents see it.

### HyDEAgent (Hypothetical Document Embeddings)

**File:** `scidiscover/agents/hyde_agent.py`
**Prompt:** `prompts/hyde.txt`

LLM generates a short hypothetical scientific abstract that *would* answer the query. The retriever embeds and searches with that hypothetical abstract instead of the raw query. The original query is restored on the resulting `evidence_pack` for the Summarizer/Synthesizer.

**Configuration (demo.yaml):**
```yaml
query_transformer:
  strategy: none   # none | decompose | hyde
  decomposer:
    prompt_path: prompts/query_decomposer.txt
    traces_output: logs/query_decomposer_traces.jsonl
    max_sub_queries: 3
  hyde:
    prompt_path: prompts/hyde.txt
    traces_output: logs/hyde_traces.jsonl
```

---

## 1. RetrieverAgent

**Files:**
- `scidiscover/agents/retriever_agent.py` — thin orchestrator wrapper
- `scidiscover/agents/retriever.py` — core FAISS query logic

### Retriever Class (retriever.py)

```python
class Retriever:
    def __init__(self, chunks_input, index_output, chunk_ids_output,
                 traces_output, model_name, papers_input)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]
```

**Retrieval process:**
1. Load FAISS index from `index_output`
2. Load chunk ID list from `chunk_ids_output` (maps integer row index → chunk_id)
3. Load chunk metadata dict from `chunks_input` (chunk_id → {paper_id, section, text, token_len})
4. Load paper metadata from `papers_input` (paper_id → {url, doi, …})
5. Encode query with SentenceTransformer (same model as index, normalized)
6. Run `index.search(query_vec, k)` → returns `(scores, indices)`
7. Map indices → chunk_ids → full chunk metadata + cosine scores
8. Write trace to `logs/retrieval_traces.jsonl`

**Return schema per chunk:**
```json
{
  "chunk_id": "pmc_PMC10002645||ABSTRACT||000",
  "paper_id": "pmc_PMC10002645",
  "section": "ABSTRACT",
  "text": "...",
  "token_len": 210,
  "score": 0.873,
  "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10002645/",
  "doi": "10.1101/2023.02.24.529927"
}
```

---

## 2. RerankerAgent (Optional)

**File:** `scidiscover/agents/reranker.py`

Uses a `CrossEncoder` model to re-score query-chunk pairs. Cross-encoders are slower but more accurate than bi-encoders for relevance ranking because they see query and chunk together.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (configurable)

**Process:**
1. Score all (query, chunk_text) pairs with CrossEncoder
2. Filter by `min_score` threshold
3. Keep `top_k` highest-scored chunks, re-sorted by score

**Configuration (demo.yaml):**
```yaml
reranker:
  enabled: true
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 10
  min_score: -.inf
```

---

## 3. SummarizerAgent

**File:** `scidiscover/agents/summarizer.py`

Summarizes each paper independently using only the chunks retrieved for it. Results feed into the Synthesizer.

**Process per paper:**
1. Group retrieved chunks by `paper_id`
2. Select best 1–3 chunks (prefer `section == "ABSTRACT"`)
3. Build context string:
   ```
   [pmc_PMC10002645||ABSTRACT||000]
   How Healthy Older Adults Enact Lateral Maneuvers...

   [pmc_PMC10002645||ABSTRACT||001]
   ...
   ```
4. Call LLM: `system=prompts/summarizer.txt`, `user=<context>`
5. Every statement in the output must be cited as `[chunk_id]`

**Parallelization:** `ThreadPoolExecutor` with `max_workers` (default: 1 for local vLLM, raise to 4–8 if the server can handle parallel requests; 3 for Gemini's ~10 RPM rate limit).

**Error handling:** If LLM call fails, returns `[LLM ERROR: {message}]` marker. Downstream agents filter these out.

**Outputs:**
- `outputs/paper_summaries.json`
- `logs/summarizer_traces.jsonl`

**Configuration (demo.yaml):**
```yaml
summarizer:
  prompt_path: prompts/summarizer.txt
  traces_output: logs/summarizer_traces.jsonl
  output_path: outputs/paper_summaries.json
  max_workers: 1
```

---

## 4. SynthesizerAgent

**File:** `scidiscover/agents/synthesizer.py`

Synthesizes all paper summaries into a single coherent answer. Runs in **JSON mode** — the LLM must return structured JSON validated by `SynthesizerLLMOutput` (Pydantic).

**Process:**
1. Filter out abstained/errored summaries (those starting with `"Insufficient evidence"` or `"[LLM ERROR"`)
2. Build multi-paper context:
   ```
   Paper: pmc_PMC10002645
   Evidence chunks: pmc_PMC10002645||ABSTRACT||000, ...
   Summary: ...

   ---

   Paper: https://openalex.org/W123456
   ...
   ```
3. Call LLM in JSON mode: `system=prompts/synthesizer.txt`, `user=<context>`
4. Parse and validate JSON response via `SynthesizerLLMOutput.model_validate_json()` (strips markdown fences as fallback)

**Expected LLM output schema:**
```json
{
  "draft_answer": "...",
  "key_claims": [
    {
      "claim": "...",
      "citation_ids": ["pmc_PMC10002645||ABSTRACT||000"]
    }
  ],
  "limitations_and_uncertainty": ["..."]
}
```

**Abstention:** If LLM response starts with `"ABSTAIN:"`, the agent returns empty claims and no draft answer.

**Citation tracking:** Looks up each `citation_id` in the evidence from Summarizer to build the final `evidence` list. If LLM hallucinates a chunk_id not in the corpus, it is caught downstream by Verifier.

**`revise()` method (used by CritiqueLoopAgent):**

```python
def revise(self, summary_pack, failing_claims, previous_draft) -> dict
```

Uses the `critique_prompt_path` prompt (`prompts/synthesizer_critique.txt`). Receives a list of failing claims (those not supported by the verifier) and the previous draft, and produces a revised synthesis that fixes or removes the unsupported claims. Returns the same schema as `run()`. Only available when `critique_prompt_path` is provided at init.

**Outputs:**
- `outputs/synthesis_result.json`
- `logs/synthesizer_traces.jsonl`

---

## 5. VerifierAgent (Optional)

**File:** `scidiscover/agents/verifier.py`

Verifies each key claim against the actual chunk text it cites. Runs in **JSON mode**. LLM output is parsed and validated by `VerifierLLMOutput` (Pydantic).

**Pre-screening (rule-based, before LLM call):**
- Claims with zero citations → immediately marked `UNSUPPORTED`
- Claims with citations → sent to LLM

**LLM verification context per claim:**
```
Claim 1:
<claim text>
Cited chunks:
  [chunk_id]
  <actual chunk text or "[Chunk not found in evidence set]">
```

**LLM output per claim:**
```json
{"status": "SUPPORTED", "confidence": 0.92, "notes": "..."}
```

Valid statuses: `SUPPORTED`, `UNSUPPORTED`, `CONFLICT`, `UNKNOWN`

**Confidence score (0.0 – 1.0):**

The LLM assigns a confidence float to each status decision:

| Range | Meaning |
|-------|---------|
| 0.9 – 1.0 | Direct, unambiguous textual evidence |
| 0.6 – 0.8 | Claim is plausible from context but not explicitly stated |
| 0.3 – 0.5 | Weak, indirect, or partial match |
| 0.0 – 0.2 | Highly ambiguous or contradictory signals |

Rule-based pre-screened claims (no citations → UNSUPPORTED) get `confidence: 1.0`.
LLM error / parse failure / no-return cases get `confidence: 0.0`.

**Abstention logic:**
```
citation_coverage = claims_with_at_least_one_citation / total_claims
support_rate = supported_claims / (supported + unsupported + conflict)

if citation_coverage >= 0.60 AND support_rate >= 0.40:
    return draft_answer
else:
    return "Insufficient evidence in retrieved corpus to answer reliably."
```

**`verification_summary` fields:**
```json
{
  "citation_coverage": 0.80,
  "support_rate": 0.75,
  "avg_confidence": 0.83,
  "total_claims": 5,
  "supported": 3,
  "unsupported": 1,
  "conflict": 0,
  "unknown": 1
}
```

**Citation normalization:** Normalizes single-pipe `|` to double-pipe `||` in chunk IDs (compensates for occasional LLM output format errors).

**`_compute_verification()` internal method:**

```python
def _compute_verification(self, synthesis: dict, evidence_pack: dict) -> dict
```

Pure verification logic with no disk I/O. Called by both `run()` (with disk writes) and `CritiqueLoopAgent` (without disk writes, for mid-loop checks). Returns the full verification dict including `needs_revision` and `failing_claims` fields used by the critique loop.

**Outputs:**
- `outputs/verification.jsonl` — includes `confidence` field per claim
- `outputs/answers.jsonl`
- `logs/verifier_traces.jsonl` — includes `confidence` per claim in trace

**Configuration (demo.yaml):**
```yaml
verifier:
  prompt_path: prompts/verifier.txt
  traces_output: logs/verifier_traces.jsonl
  verification_output: outputs/verification.jsonl
  answers_output: outputs/answers.jsonl
  min_citation_coverage: 0.60
  min_support_rate: 0.40
```

**Note:** The config key is `verification_output` (not `output_path`).

---

## 6. CritiqueLoopAgent (Optional)

**File:** `scidiscover/agents/critique_loop.py`

Wraps `SynthesizerAgent` and `VerifierAgent` in an iterative refinement loop. When the verifier finds that support rate falls below threshold, it feeds the failing claims back to the synthesizer as a critique and re-synthesizes. Controlled by `critique_loop.enabled` in `configs/demo.yaml`.

**Process:**
1. Call `synthesizer.run()` to get initial synthesis
2. Call `verifier._compute_verification()` (no disk I/O) to check support rate
3. If `needs_revision` is False or max iterations reached → stop early
4. Otherwise, call `synthesizer.revise(failing_claims=..., previous_draft=...)` using the critique prompt
5. If revised draft is invalid (ABSTAIN / error) → revert to previous synthesis
6. Repeat from step 2 up to `max_iterations` times
7. Call `verifier.run()` (with disk I/O) on the final synthesis

**`run()` API:**
```python
def run(self, summary_pack: dict, evidence_pack: dict) -> dict
```

Returns:
```json
{
  "synthesis":    { /* final SynthesizerAgent output */ },
  "verified":     { /* final VerifierAgent output */ },
  "n_iterations": 1
}
```

**Configuration (demo.yaml):**
```yaml
critique_loop:
  enabled: false           # set to true to activate
  max_iterations: 2
  critique_prompt_path: prompts/synthesizer_critique.txt
  traces_output: logs/critique_loop_traces.jsonl
```

**Trace output:** `logs/critique_loop_traces.jsonl` — logs per-iteration support_rate, citation_coverage, claim counts, and the final answer preview.

---

## Pydantic Validation Models

**File:** `utils/models.py`

All agent-boundary data is validated by Pydantic models. There are two categories:

**Validation models** (strict — raise `ValidationError` on bad data):

| Model | Used at |
|-------|---------|
| `EvidenceChunk` | Retriever / RerankerAgent output |
| `PaperSummary` | SummarizerAgent output |
| `SynthesisPack` | SynthesizerAgent output |
| `VerificationPack` | VerifierAgent output |

**LLM output models** (permissive — unknown fields ignored, missing optionals default gracefully):

| Model | Used by |
|-------|---------|
| `SynthesizerLLMOutput` | SynthesizerAgent JSON parse |
| `VerifierLLMOutput` | VerifierAgent JSON parse |
| `QueryDecomposerOutput` | QueryDecomposerAgent JSON parse |
| `SingleAgentLLMOutput` | SingleAgentBaseline JSON parse |

LLM output models also normalise common LLM mistakes: wrong status strings, out-of-range confidence scores (clamped to `[0.0, 1.0]`), and `null` citation lists (coerced to `[]`).

**File:** `utils/schemas.py`

Thin wrapper around the Pydantic models. Exports `validate_evidence_pack`, `validate_summary_pack`, `validate_synthesis_pack`, `validate_verification_pack` — each accepts the original dict/list format and raises `ValueError` on failure. Existing call sites remain unchanged.

---

## LLM Client

**File:** `utils/llm_client.py`

Multi-provider LLM router. All agents call `call_llm()` — never provider SDKs directly.

### API

```python
configure_llm(model: str, max_tokens: int) -> None
call_llm(system: str, user: str, json_mode: bool = False, max_retries: int = 4) -> str
```

### Provider Routing (by model name prefix)

| Prefix | Provider | Env Var | JSON Mode |
|--------|----------|---------|-----------|
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` | Via prompt instruction |
| `gemini-*` | Google Gemini | `GOOGLE_API_KEY` | `response_mime_type="application/json"` |
| `openrouter-*` | OpenRouter (OpenAI-compatible) | `OPENROUTER_API_KEY` | `response_format={"type":"json_object"}` |
| `local-*` | OpenAI-compatible local server | `OPENAI_BASE_URL` (+ optional `OPENAI_API_KEY`) | Via prompt instruction |

**OpenRouter format:** the prefix `openrouter-` is stripped, and the rest is passed as the model ID (e.g. `openrouter-openai/gpt-5.4-mini` → `openai/gpt-5.4-mini` against `https://openrouter.ai/api/v1`).

- **Rate limit handling:** Exponential backoff (15 s, 30 s, 45 s, 60 s) on 429 / quota errors
- **Thinking blocks:** Strips `<thinking>...</thinking>` blocks from Gemini responses automatically
- **Truncation warning:** Logs warning if output reaches `max_tokens`

### Active Model (demo.yaml)

```yaml
llm:
  model: openrouter-openai/gpt-5.4-mini
  max_tokens: 4096
```

Other tested values for `llm.model`:
- `gemini-2.5-flash`
- `local-qwen2.5-72b-instruct-gptq`
- Any OpenRouter model: `openrouter-anthropic/claude-3.5-sonnet`, `openrouter-openai/gpt-4o-mini`, etc.

**Note:** `max_tokens` is the completion length cap, not the total context. The local vLLM server has a total context window of 8192 tokens, so prompt + `max_tokens` must stay under 8192. With `max_tokens: 4096` you have 4096 tokens for the prompt — plenty for our prompts.

---

## System Prompts

| Agent | Prompt File | Key Rules |
|-------|-------------|-----------|
| QueryDecomposer | `prompts/query_decomposer.txt` | JSON output `{"sub_queries": [...]}`, ≤`max_sub_queries` items |
| HyDE | `prompts/hyde.txt` | Plain-text 3–5 sentence hypothetical scientific abstract |
| Summarizer | `prompts/summarizer.txt` | Every statement cited as `[chunk_id]`, 2–3 bullet points |
| Synthesizer | `prompts/synthesizer.txt` | JSON output, every claim has `citation_ids` array |
| Synthesizer (revise) | `prompts/synthesizer_critique.txt` | Same JSON schema; must fix or remove each failing claim listed in prompt |
| SingleAgentBaseline | `prompts/single_agent_baseline.txt` | Replaces Summarizer + Synthesizer + Verifier with one LLM call |
| Verifier | `prompts/verifier.txt` | JSON output, no outside knowledge, `[Chunk not found]` → UNKNOWN |

---

## Evaluation

Driven by `scidiscover/eval/run.py`, using a RAGAS-generated test set in `eval/`.

### Test Set Composition (`eval/queries.jsonl`)

| Query type | Difficulty | Generator | Purpose |
|---|---|---|---|
| `ragas_single_hop` | easy | RAGAS `SingleHopSpecificQuerySynthesizer` | Single-chunk factual question |
| `ragas_multi_hop` | medium / hard | RAGAS `MultiHopSpecificQuerySynthesizer` + `MultiHopAbstractQuerySynthesizer` | Cross-chunk reasoning |
| `unanswerable` | unanswerable | LLM (asks for details not in abstract) | Should abstain |
| `out_of_domain` | easy / medium / hard | Hardcoded (5 queries) | Should abstain |

`expected_paper_ids` and `expected_chunk_ids` are populated during generation by matching RAGAS's `reference_contexts` against the chunk index.

### Generating the RAGAS Test Set

```bash
# Step 1 — generate queries (reads ragas_model from configs/demo.yaml)
PYTHONPATH=. python eval/generate_ragas_testset.py \
    --n-chunks 300 --testset-size 70 --unanswerable 10

# Override eval model
PYTHONPATH=. python eval/generate_ragas_testset.py \
    --n-chunks 300 --testset-size 70 --unanswerable 10 \
    --eval-model openai/gpt-4o-mini

```

`--eval-model` accepts any OpenRouter model ID (`provider/model-name`) or `gemini-*`. Provider is auto-detected by `utils/eval_llm.py`. JSON mode is enforced for both providers (`response_format` / `response_mime_type`).

`expected_chunk_ids` are resolved and written directly into `queries.jsonl` during generation by matching RAGAS `reference_contexts` against the sampled chunk index. No separate `build_labels.py` step is needed.

### Knowledge Graph Build (RAGAS testset generation)

When `generate_ragas_testset.py` runs, RAGAS applies these transforms in order to build a knowledge graph for query synthesis:

1. **`SummaryExtractor`** — summarize each chunk
2. **`CustomNodeFilter`** — score 1–5 for question potential, drop low scores (patched to skip on intermittent failures)
3. **`EmbeddingExtractor`** — embed chunks with MiniLM
4. **`ThemesExtractor`** + **`NERExtractor`** — extract topics and entities
5. **`CosineSimilarityBuilder`** + **`OverlapScoreBuilder`** — link related chunks
6. **Persona generation** — create researcher personas
7. **Scenario generation** — propose query scenarios from graph traversal
8. **Sample synthesis** — turn scenarios into final query/reference pairs

### Running Evaluation

```bash
# Quick test (5 queries, skip RAGAS scoring)
PYTHONPATH=. python -m scidiscover.eval.run \
    --config configs/demo.yaml --max-queries 5 --skip-ragas

# Full run
PYTHONPATH=. python -m scidiscover.eval.run --config configs/demo.yaml

# Ablation — no verifier
PYTHONPATH=. python -m scidiscover.eval.run \
    --config configs/demo.yaml --skip-verifier \
    --output reports/eval_no_verifier.json

# Override query transformer strategy from CLI
PYTHONPATH=. python -m scidiscover.eval.run \
    --config configs/demo.yaml --query-transformer hyde \
    --output reports/eval_hyde.json

# Decouple retrieval pool size from recall@k (for reranker ablation)
PYTHONPATH=. python -m scidiscover.eval.run \
    --config configs/demo.yaml --retrieval-k 30 \
    --output reports/eval_pool30.json

# Single-agent baseline
PYTHONPATH=. python -m scidiscover.eval.run \
    --config configs/demo.yaml --baseline single_agent \
    --output reports/eval_single_agent.json
```

### Ablation Scripts (in `scripts/`)

Drive multiple eval runs and print a side-by-side comparison table:

```bash
# Compare none vs hyde vs decompose
PYTHONPATH=. python scripts/run_transformer_eval.py --config configs/demo.yaml

# Compare reranker on vs off (matched output size)
PYTHONPATH=. python scripts/run_reranker_eval.py --config configs/demo.yaml

# Common flags supported by both: --max-queries N, --skip-ragas, --skip-verifier
```

### CLI Flags Reference

| Flag | Purpose |
|------|---------|
| `--config` | Path to `configs/demo.yaml` (required) |
| `--max-queries N` | Cap query count for quick tests |
| `--output PATH` | Override output report path |
| `--skip-ragas` | Skip RAGAS scoring (saves ~15min) |
| `--skip-verifier` | Run pipeline without VerifierAgent |
| `--skip-reranker` | Disable RerankerAgent regardless of config |
| `--query-transformer {none,decompose,hyde}` | Override `query_transformer.strategy` |
| `--retrieval-k N` | Override retriever pool size (default: `eval.top_k_recall`). Use a larger value than `top_k_recall` when reranker is enabled so it has a wider pool to filter from |
| `--baseline single_agent` | Replace Summarizer + Synthesizer + Verifier with one LLM call |

### Metrics

| Metric | Source | What it measures |
|---|---|---|
| `recall_at_k` | Custom | Fraction of expected **chunk IDs** found in top-k retrieved chunks (chunk-level, not paper-level) |
| `recall_at_k_by_query_type` | Custom | Same recall split by `ragas_single_hop` / `ragas_multi_hop` / `unanswerable` / `out_of_domain` |
| `hallucination_rate` | Custom | Fraction of `[chunk_id]` citations in summaries not in the retrieved set |
| `citation_coverage` | Custom | Fraction of key claims with at least one citation |
| `support_rate` | Verifier | Fraction of claims marked SUPPORTED (excludes UNKNOWN from denominator) |
| `abstention_rate_answerable` | Custom | Fraction of answerable queries (`ragas_*`) that returned "Insufficient evidence" — lower is better |
| `correct_abstention_rate` | Custom | Fraction of `unanswerable` / `out_of_domain` queries that correctly abstained — higher is better |
| `avg_latency_s` | Custom | Full end-to-end wall-clock time including verifier |
| RAGAS faithfulness | RAGAS + `eval.ragas_model` | Answer grounded in retrieved contexts |
| RAGAS answer relevancy | RAGAS + `eval.ragas_model` | Answer relevant to the question |
| RAGAS context recall | RAGAS + `eval.ragas_model` | Retrieved contexts cover the ground truth |

**RAGAS model:** set via `eval.ragas_model` in `configs/demo.yaml`. OpenRouter models (e.g. `openai/gpt-5.4-mini`, `openai/gpt-4o-mini`) and Gemini models (e.g. `gemini-2.5-flash`) are supported. Note: OpenRouter does not support `n>1` for chat completions ([known limitation](https://github.com/OpenRouterTeam/openrouter-runner/issues/99)) — RAGAS will emit "LLM returned 1 generations instead of requested 3" warnings but scores remain valid. Use a Gemini model directly for `n=3` stability.

Results written to `reports/eval_results.json` with per-query detail and per-domain breakdown.

---

## Trace & Output Files

| File | Written by | Format |
|------|-----------|--------|
| `logs/query_decomposer_traces.jsonl` | QueryDecomposerAgent | JSONL |
| `logs/hyde_traces.jsonl` | HyDEAgent | JSONL |
| `logs/retrieval_traces.jsonl` | RetrieverAgent | JSONL |
| `logs/reranker_traces.jsonl` | RerankerAgent | JSONL |
| `logs/summarizer_traces.jsonl` | SummarizerAgent | JSONL |
| `logs/synthesizer_traces.jsonl` | SynthesizerAgent | JSONL |
| `logs/verifier_traces.jsonl` | VerifierAgent | JSONL |
| `logs/critique_loop_traces.jsonl` | CritiqueLoopAgent | JSONL |
| `logs/single_agent_baseline_traces.jsonl` | SingleAgentBaseline | JSONL |
| `outputs/paper_summaries.json` | SummarizerAgent | JSON |
| `outputs/synthesis_result.json` | SynthesizerAgent | JSON |
| `outputs/verification.jsonl` | VerifierAgent | JSONL |
| `outputs/answers.jsonl` | VerifierAgent | JSONL |

**Note on shared output paths:** all strategies write to the same shared `logs/` and `outputs/` paths by default. Running multiple ablations sequentially overwrites earlier traces. Only the final eval reports (`reports/eval_*.json`) are uniquely named per condition.
