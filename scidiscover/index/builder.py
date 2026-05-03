import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_path: str):
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def ensure_parent(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def write_chunk_ids(path_str: str, chunk_ids):
    ensure_parent(path_str)
    with open(path_str, "w", encoding="utf-8") as f:
        for cid in chunk_ids:
            f.write(f"{cid}\n")


def build_faiss_index(
    chunks_input: str,
    embeddings_output: str,
    index_output: str,
    chunk_ids_output: str,
    model_name: str,
    meta_output: str,
    top_k: int = 5,
    chunk_size: int = 200,
    overlap: int = 40,
):
    chunks = load_chunks(chunks_input)

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_input}")

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]

    # cosine similarity via inner product because embeddings are normalized
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    ensure_parent(embeddings_output)
    np.save(embeddings_output, embeddings)

    ensure_parent(index_output)
    faiss.write_index(index, index_output)

    write_chunk_ids(chunk_ids_output, chunk_ids)

    # Write index metadata
    meta = {
        "model_name": model_name,
        "dim": int(dim),
        "num_chunks": len(chunks),
        "index_type": "FlatIP",
        "k": top_k,
        "chunk_policy": {
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
    }
    ensure_parent(meta_output)
    with open(meta_output, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings: {embeddings_output}")
    print(f"Saved index: {index_output}")
    print(f"Saved chunk ids: {chunk_ids_output}")
    print(f"Saved metadata: {meta_output}")
    print(f"Indexed {len(chunks)} chunks with dim={dim}")
