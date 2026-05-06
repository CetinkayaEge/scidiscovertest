# Indexing: Embeddings & FAISS

## Overview

After chunking, `scidiscover/index/builder.py` encodes all chunks into 384-dimensional vectors and stores them in a FAISS flat inner-product index. This index is what the retriever queries at runtime.

---

## Index Builder

**File:** `scidiscover/index/builder.py`

### Main Function

```python
build_faiss_index(
    chunks_input,        # path to chunks.jsonl
    embeddings_output,   # path to save .npy embeddings
    index_output,        # path to save faiss.index
    chunk_ids_output,    # path to save chunk_ids.txt
    model_name,          # sentence-transformers model name
    meta_output,         # path to save index.meta.json
    top_k=5,
    chunk_size=200,
    overlap=40
)
```

### Process (Step by Step)

1. **Load chunks** from `chunks_input` JSONL → extract `text` field from each chunk
2. **Load embedding model:** `SentenceTransformer(model_name)` — downloads and caches from HuggingFace on first run
3. **Encode all texts:**
   - Batch size: 64
   - `normalize_embeddings=True` — L2-normalized so inner product = cosine similarity
   - Output dtype: float32
   - Shape: `(num_chunks, 384)`
4. **Build FAISS index:**
   - Type: `IndexFlatIP` (flat inner product — exact search, no approximation)
   - Dimension: 384
   - Calls `index.add(embeddings)`
5. **Save outputs:**
   - `embeddings_output` — NumPy `.npy` array
   - `index_output` — FAISS binary index
   - `chunk_ids_output` — one chunk ID per line, in the same order as index rows
   - `meta_output` — JSON metadata

---

## Index Files

| File | Contents |
|------|----------|
| `embeddings/chunks.npy` | float32 array, shape (num_chunks, 384) |
| `index/faiss.index` | FAISS FlatIP binary index |
| `index/chunk_ids.txt` | chunk IDs, one per line, order matches index rows |
| `index/index.meta.json` | model name, dimensions, chunk count, config |

### index.meta.json

```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "dim": 384,
  "num_chunks": <current_count>,
  "index_type": "FlatIP",
  "k": 20,
  "chunk_policy": {
    "chunk_size": 200,
    "overlap": 40
  }
}
```

---

## Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

- Dimensions: 384
- Architecture: 6-layer MiniLM fine-tuned for sentence similarity
- Speed: Fast; suitable for embedding tens of thousands of chunks on CPU
- Similarity metric: Cosine (via L2-normalized inner product)

The same model is used both at index-build time (to encode chunks) and at query time (to encode the user query). This is critical — query and chunk vectors must come from the same model to be comparable.

---

## Configuration (demo.yaml)

```yaml
retrieval:
  chunks_input: data/processed/chunks.jsonl
  papers_input: data/raw/papers.jsonl
  embeddings_output: embeddings/chunks.npy
  index_output: index/faiss.index
  chunk_ids_output: index/chunk_ids.txt
  meta_output: index/index.meta.json
  model_name: sentence-transformers/all-MiniLM-L6-v2
  top_k: 20
```

**Note:** Both `papers_input` and the ingestion output point to `data/raw/papers.jsonl` — no copy step is needed. The retriever reads URLs and DOIs directly from the same file the ingestors write.

---

## Rebuilding the Index

After adding more papers and re-chunking, the index must be rebuilt:

```bash
# Re-run everything
python -m scidiscover.run_demo --config configs/demo.yaml

# Re-run only chunking + indexing (skip ingestion)
python -m scidiscover.run_demo --config configs/demo.yaml --skip-ingestion

# Re-run only indexing (skip ingestion + chunking)
python -m scidiscover.run_demo --config configs/demo.yaml --skip-ingestion --skip-chunking
```

Index build time scales linearly with the number of chunks. For every ~10,000 chunks it takes under a minute on CPU.
