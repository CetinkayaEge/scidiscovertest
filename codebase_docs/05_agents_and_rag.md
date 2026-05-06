# Agents and RAG

## Agent Pipeline Overview

Five agents run in sequence for each query. Reranker and Verifier are optional and toggled via config or the UI sidebar.

```
User Query
    │
    ▼
RetrieverAgent   ──► top-k chunks from FAISS (k=20 by default)
    │
    ▼ (optional)
RerankerAgent    ──► CrossEncoder reranked subset
    │
    ▼
SummarizerAgent  ──► per-paper summaries with chunk citations (parallel)
    │
    ▼
SynthesizerAgent ──► draft answer + key claims with citation IDs (JSON)
    │
    ▼ (optional)
VerifierAgent    ──► verified claims, final answer or abstention (JSON)
    │
    ▼
Streamlit UI / outputs/answers.jsonl
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
  enabled: false
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 10
  min_score: -10.0
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

**Parallelization:** `ThreadPoolExecutor` with `max_workers` (default: 4). Safe for Gemini API's ~10 RPM rate limit with 3–4 workers.

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
  max_workers: 4
```

---

## 4. SynthesizerAgent

**File:** `scidiscover/agents/synthesizer.py`

Synthesizes all paper summaries into a single coherent answer. Runs in **JSON mode** — the LLM must return structured JSON.

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
4. Parse JSON response (strips markdown fences if present as fallback)

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
  "limitations_and_uncertainty": "..."
}
```

**Abstention:** If LLM response starts with `"ABSTAIN:"`, the agent returns empty claims and no draft answer.

**Citation tracking:** Looks up each `citation_id` in the evidence from Summarizer to build the final `evidence` list. If LLM hallucinates a chunk_id not in the corpus, it is caught downstream by Verifier.

**Outputs:**
- `outputs/synthesis_result.json`
- `logs/synthesizer_traces.jsonl`

---

## 5. VerifierAgent (Optional)

**File:** `scidiscover/agents/verifier.py`

Verifies each key claim against the actual chunk text it cites. Runs in **JSON mode**.

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
{"status": "SUPPORTED", "notes": "..."}
```

Valid statuses: `SUPPORTED`, `UNSUPPORTED`, `CONFLICT`, `UNKNOWN`

**Abstention logic:**
```
citation_coverage = claims_with_at_least_one_citation / total_claims
support_rate = supported_claims / (supported + unsupported + conflict)

if citation_coverage >= 0.60 AND support_rate >= 0.40:
    return draft_answer
else:
    return "Insufficient evidence in retrieved corpus to answer reliably."
```

**Citation normalization:** Normalizes single-pipe `|` to double-pipe `||` in chunk IDs (compensates for occasional LLM output format errors).

**Outputs:**
- `outputs/verification.jsonl`
- `outputs/answers.jsonl`
- `logs/verifier_traces.jsonl`

**Configuration (demo.yaml):**
```yaml
verifier:
  prompt_path: prompts/verifier.txt
  traces_output: logs/verifier_traces.jsonl
  output_path: outputs/verification.jsonl
  answers_output: outputs/answers.jsonl
  min_citation_coverage: 0.60
  min_support_rate: 0.40
```

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
| `local-*` | OpenAI-compatible | `OPENAI_BASE_URL` | Via prompt instruction |

- **Rate limit handling:** Exponential backoff (15 s, 30 s, 45 s, 60 s) on 429 / quota errors
- **Thinking blocks:** Strips `<thinking>...</thinking>` blocks from Gemini responses automatically
- **Truncation warning:** Logs warning if output reaches `max_tokens`

### Active Model (demo.yaml)

```yaml
llm:
  model: gemini-2.5-flash
  max_tokens: 4096
```

---

## System Prompts

| Agent | Prompt File | Key Rules |
|-------|-------------|-----------|
| Summarizer | `prompts/summarizer.txt` | Every statement cited as `[chunk_id]`, 2–3 bullet points |
| Synthesizer | `prompts/synthesizer.txt` | JSON output, every claim has `citation_ids` array |
| Verifier | `prompts/verifier.txt` | JSON output, no outside knowledge, `[Chunk not found]` → UNKNOWN |

---

## Trace & Output Files

| File | Written by | Format |
|------|-----------|--------|
| `logs/retrieval_traces.jsonl` | RetrieverAgent | JSONL |
| `logs/summarizer_traces.jsonl` | SummarizerAgent | JSONL |
| `logs/synthesizer_traces.jsonl` | SynthesizerAgent | JSONL |
| `logs/verifier_traces.jsonl` | VerifierAgent | JSONL |
| `outputs/paper_summaries.json` | SummarizerAgent | JSON |
| `outputs/synthesis_result.json` | SynthesizerAgent | JSON |
| `outputs/verification.jsonl` | VerifierAgent | JSONL |
| `outputs/answers.jsonl` | VerifierAgent | JSONL |
