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

## Index Files (V2)

All active index files are under `indexv2/` and `embeddingsv2/`.

| File | Size | Contents |
|------|------|----------|
| `embeddingsv2/chunks.npy` | ~16 MB | float32 array, shape (10219, 384) |
| `indexv2/faiss.index` | ~16 MB | FAISS FlatIP binary index |
| `indexv2/chunk_ids.txt` | ~390 KB | 10,219 chunk IDs, one per line |
| `indexv2/index.meta.json` | 199 B | Model name, dimensions, config |

### index.meta.json (V2)

```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "dim": 384,
  "num_chunks": 10219,
  "index_type": "FlatIP",
  "k": 20,
  "chunk_policy": {
    "chunk_size": 200,
    "overlap": 40
  }
}
```

---

## V1 vs V2 Index Comparison

| Property | V1 (`index/`) | V2 (`indexv2/`) |
|----------|--------------|----------------|
| Chunks indexed | 15,294 | 10,219 |
| Index file size | ~22 MB | ~16 MB |
| Embeddings size | ~22 MB | ~16 MB |
| top_k stored in meta | 5 | 20 |
| Active | No | Yes |

---

## Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

- Dimensions: 384
- Architecture: 6-layer MiniLM fine-tuned for sentence similarity
- Speed: Fast; suitable for embedding thousands of chunks on CPU
- Similarity metric: Cosine (via L2-normalized inner product)

The same model is used both at index-build time (to encode chunks) and at query time (to encode the user query). This is critical — query and chunk vectors must come from the same model to be comparable.

---

## Configuration (demo.yaml)

```yaml
retrieval:
  chunks_input: datav2/processed/chunks.jsonl
  papers_input: data/raw/papers.jsonl
  embeddings_output: embeddingsv2/chunks.npy
  index_output: indexv2/faiss.index
  chunk_ids_output: indexv2/chunk_ids.txt
  meta_output: indexv2/index.meta.json
  model_name: sentence-transformers/all-MiniLM-L6-v2
  top_k: 20
```

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

Index build time scales linearly with the number of chunks. For the current 10,219 chunks it is fast (under a minute on CPU). At 50,000+ chunks it may take a few minutes.
