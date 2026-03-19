import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

# -----------------------------
# CONFIG
# -----------------------------

INDEX_PATH = "index/faiss.index"
CHUNKS_PATH = "data/processed/chunks.jsonl"
LOG_PATH = "logs/retrieval_traces.jsonl"

QUERY = "impact of air pollution on cardiovascular health"
TOP_K = 5


# -----------------------------
# LOAD MODEL
# -----------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# LOAD FAISS INDEX
# -----------------------------

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)


# -----------------------------
# LOAD CHUNKS
# -----------------------------

print("Loading chunk metadata...")

chunks = []
with open(CHUNKS_PATH, "r") as f:
    for line in f:
        chunks.append(json.loads(line))


# -----------------------------
# ENCODE QUERY
# -----------------------------

print("Encoding query...")

query_embedding = model.encode([QUERY])


# -----------------------------
# SEARCH INDEX
# -----------------------------

print("Searching FAISS index...\n")

distances, indices = index.search(query_embedding, TOP_K)


# -----------------------------
# PRINT RESULTS
# -----------------------------

print("Query:", QUERY)
print("\nTop Results:\n")

results = []

for rank, idx in enumerate(indices[0]):

    chunk = chunks[idx]

    score = float(distances[0][rank])

    print("Result", rank + 1)
    print("Score:", score)
    print("Paper:", chunk["paper_id"])
    print("Section:", chunk["section"])
    print("Text:", chunk["text"][:300])
    print("-" * 60)

    results.append({
        "rank": rank + 1,
        "score": score,
        "paper_id": chunk["paper_id"],
        "section": chunk["section"],
        "chunk_id": chunk["chunk_id"],
        "text_preview": chunk["text"][:200]
    })


# -----------------------------
# SAVE RETRIEVAL TRACE
# -----------------------------

trace = {
    "timestamp": datetime.utcnow().isoformat(),
    "query": QUERY,
    "top_k": TOP_K,
    "results": results
}

os.makedirs("logs", exist_ok=True)

with open(LOG_PATH, "a") as f:
    f.write(json.dumps(trace) + "\n")

print("\nRetrieval trace saved →", LOG_PATH)