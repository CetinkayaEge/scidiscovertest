# Processing Pipeline: Chunking

## Chunker

**File:** `scidiscover/process/chunker.py`

The chunker reads `data/raw/papers.jsonl`, splits each paper into overlapping token-window chunks, and writes `data/processed/chunks.jsonl`.

### Core Functions

```python
tokenize(text) -> List[str]
```
Simple whitespace tokenizer.

```python
chunk_text(text, chunk_size, overlap) -> Generator[(offset, chunk_text, token_len)]
```
Sliding-window chunker. Yields `(offset, text, token_count)` tuples.
- `offset` is the integer index of the chunk within its section (0, 1, 2, …).
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

The title is prepended to every abstract chunk with a ` | ` separator. This means every chunk retrieved by the RAG system is self-identifying — the LLM sees both title and content without a separate lookup.

### Configuration (demo.yaml)

```yaml
chunking:
  input_path: data/raw/papers.jsonl
  output_path: data/processed/chunks.jsonl
  chunk_size: 200
  overlap: 40
```
