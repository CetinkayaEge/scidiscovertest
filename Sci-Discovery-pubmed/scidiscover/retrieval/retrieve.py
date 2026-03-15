import json
from datetime import datetime, timezone

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks_map(chunks_path: str):
    chunks_by_id = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks_by_id[chunk["chunk_id"]] = chunk
    return chunks_by_id


def load_chunk_ids(path_str: str):
    with open(path_str, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class Retriever:
    def __init__(
        self,
        chunks_input: str,
        index_output: str,
        chunk_ids_output: str,
        traces_output: str,
        model_name: str,
    ):
        self.chunks_by_id = load_chunks_map(chunks_input)
        self.chunk_ids = load_chunk_ids(chunk_ids_output)
        self.index = faiss.read_index(index_output)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.traces_output = traces_output

    def retrieve(self, query: str, k: int = 5):
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue

            chunk_id = self.chunk_ids[idx]
            chunk = self.chunks_by_id[chunk_id]

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "token_len": chunk["token_len"],
                    "score": float(score),
                }
            )

        self.log_trace(query, k, results)
        return results

    def log_trace(self, query: str, k: int, results):
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "top_k": k,
            "model_name": self.model_name,
            "results": [
                {
                    "rank": i + 1,
                    "chunk_id": r["chunk_id"],
                    "paper_id": r["paper_id"],
                    "score": r["score"],
                    "section": r["section"],
                }
                for i, r in enumerate(results)
            ],
        }

        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")