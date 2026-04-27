from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import CrossEncoder


class RerankerAgent:
    def __init__(self, model_name: str, top_k: int,
                 traces_output: str = "logs/reranker_traces.jsonl") -> None:
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        self.top_k = top_k
        self.traces_output = traces_output

    def run(self, evidence_pack: dict) -> dict:
        query = evidence_pack["query"]
        chunks = evidence_pack["chunks"]
        if not chunks:
            return evidence_pack

        pairs = [(query, ch["text"]) for ch in chunks]
        raw_scores = self.model.predict(pairs)

        scored = []
        for chunk, raw in zip(chunks, raw_scores):
            c = dict(chunk)
            c["score"] = float(raw)
            scored.append(c)

        scored.sort(key=lambda c: c["score"], reverse=True)
        reranked = scored[: self.top_k]

        self._log_trace(query, chunks, reranked)
        return {"query": query, "chunks": reranked}

    def _log_trace(self, query, input_chunks, output_chunks):
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "model_name": self.model_name,
            "input_count": len(input_chunks),
            "output_count": len(output_chunks),
            "results": [
                {"rank": i + 1, "chunk_id": ch["chunk_id"],
                 "paper_id": ch["paper_id"], "score": ch["score"],
                 "section": ch["section"]}
                for i, ch in enumerate(output_chunks)
            ],
        }
        Path(self.traces_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
