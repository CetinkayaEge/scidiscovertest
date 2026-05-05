# Processing Pipeline: Chunking & Data Versions

## 1. Chunker

**File:** `scidiscover/process/chunker.py`

The chunker reads `data/raw/papers.jsonl`, splits each paper into overlapping token-window chunks, and writes `datav2/processed/chunks.jsonl`.

### Core Functions

```python
tokenize(text) -> List[str]
```
Simple whitespace tokenizer.

```python
chunk_text(text, chunk_size, overlap) -> Generator[(offset, chunk_text, token_len)]
```
Sliding-window chunker. Yields `(offset, text, token_count)` tuples.
- `offset` is the integer index of the first token in the chunk within its section.
- The last chunk of a section may be shorter than `chunk_size`.

```python
run()
```
Main orchestrator. For each paper:
1. Prepends the title to the abstract with ` | ` as separator to form `combined_abstract`.
2. Builds a section map: `{"ABSTRACT": combined_abstract, "BODY_UNK": paper.get("full_text") or paper.get("body")}`
3. Iterates sections, chunks each non-empty section, writes chunks to JSONL.

The title-prepend logic:
```python
title = paper.get("title", "").strip()
abstract = paper.get("abstract", "").strip()
combined_abstract = f"{title} | {abstract}" if title and abstract else title or abstract
```

### Chunk Schema

```json
{
  "chunk_id": "pmc_PMC10002645||ABSTRACT||000",
  "paper_id": "pmc_PMC10002645",
  "section": "ABSTRACT",
  "text": "How Healthy Older Adults Enact Lateral Maneuvers While Walking | Background: ...",
  "token_len": 210
}
```

**Chunk ID format:** `{paper_id}||{section}||{offset:03d}`
- Separator is **double pipe** (`||`)
- Offset is zero-padded to 3 digits

### Configuration (demo.yaml)

```yaml
chunking:
  input_path: data/raw/papers.jsonl
  output_path: datav2/processed/chunks.jsonl
  chunk_size: 200
  overlap: 40
```

---

## 2. data/ vs datav2/ Difference

| Property | V1 (`data/`) | V2 (`datav2/`) |
|----------|-------------|----------------|
| Section types | TITLE, ABSTRACT, BODY_UNK | ABSTRACT, BODY_UNK only |
| Total chunks | 15,294 | 10,219 |
| Title in abstract text | No | Yes, prepended with ` \| ` |
| FAISS index chunk count | 15,294 | 10,219 (after index rebuild) |
| Embedding file size | ~22 MB | ~16 MB (after index rebuild) |

### V1 (`data/processed/chunks.jsonl`) — Title as separate chunk

```json
{"chunk_id": "pmc_PMC10002645||TITLE||000", "section": "TITLE", "text": "How Healthy Older Adults Enact Lateral Maneuvers While Walking", "token_len": 9}
{"chunk_id": "pmc_PMC10002645||ABSTRACT||000", "section": "ABSTRACT", "text": "Background: Walking requires frequent maneuvers to navigate changing environments...", "token_len": 210}
```

The title is a standalone chunk with `section: "TITLE"` and token_len of ~9 tokens.

### V2 (`datav2/processed/chunks.jsonl`) — Title prepended to abstract

```json
{"chunk_id": "pmc_PMC10002645||ABSTRACT||000", "section": "ABSTRACT", "text": "How Healthy Older Adults Enact Lateral Maneuvers While Walking | Background: Walking requires frequent maneuvers...", "token_len": 210}
```

There is **no TITLE chunk**. The title is prepended to the abstract text with a ` | ` separator, making every abstract chunk self-identifying without a separate lookup.

### Why V2 is Superior

When a retrieval returns a chunk, the title is immediately visible in the chunk text itself. No separate lookup is needed to know what paper a chunk belongs to — the LLM sees both title and content in every abstract chunk it receives as context.

---

## 3. Which Version is Active

All paths in `configs/demo.yaml` point to v2:

```yaml
chunking:
  output_path: datav2/processed/chunks.jsonl

retrieval:
  chunks_input: datav2/processed/chunks.jsonl
  embeddings_output: embeddingsv2/chunks.npy
  index_output: indexv2/faiss.index
  chunk_ids_output: indexv2/chunk_ids.txt
  meta_output: indexv2/index.meta.json
```

V1 directories (`data/processed/`, `embeddings/`, `index/`) remain on disk but are not used by the active pipeline.
