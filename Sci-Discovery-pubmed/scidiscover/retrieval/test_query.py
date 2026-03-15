import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("index/faiss.index")

# Load chunk metadata
chunks = []
with open("data/processed/chunks.jsonl", "r") as f:
    for line in f:
        chunks.append(json.loads(line))

# Example query
query = "impact of air pollution on cardiovascular health"

# Encode query
query_embedding = model.encode([query])

# Search FAISS index
top_k = 5
distances, indices = index.search(query_embedding, top_k)

print("\nQuery:", query)
print("\nTop Results:\n")

for rank, idx in enumerate(indices[0]):
    chunk = chunks[idx]

    print("Result", rank + 1)
    print("Score:", distances[0][rank])
    print("Paper:", chunk["paper_id"])
    print("Section:", chunk["section"])
    print("Text:", chunk["text"][:300])
    print("-" * 50)