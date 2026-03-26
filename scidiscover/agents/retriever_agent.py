import json
from pathlib import Path


class RetrieverAgent:
    def __init__(self, retriever, traces_output="logs/retriever_agent_traces.jsonl"):
        self.retriever = retriever
        self.traces_output = traces_output

    def log_trace(self, query, chunks):
        trace = {
            "query": query,
            "num_chunks": len(chunks),
            "chunk_ids": [c["chunk_id"] for c in chunks],
        }

        Path(self.traces_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")

    def run(self, query: str, k: int = 5):
        chunks = self.retriever.retrieve(query, k)

        evidence_pack = {
            "query": query,
            "chunks": chunks,
        }

        self.log_trace(query, chunks)

        return evidence_pack