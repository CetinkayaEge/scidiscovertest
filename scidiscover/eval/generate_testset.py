"""
Generate evaluation queries from the corpus using RAGAS TestsetGenerator.

Usage:
    PYTHONPATH=. python -m scidiscover.eval.generate_testset --config configs/demo.yaml --n-queries 30
"""

import argparse
import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Label helpers (mirrors eval/build_labels.py)
# ---------------------------------------------------------------------------

def _strip_hop_tags(text: str) -> str:
    return re.sub(r"^<\d+-hop>\s*", "", text.strip())


def _build_chunk_index(chunks_path: str) -> dict:
    index = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            text = chunk.get("text", "").strip()
            if not text:
                continue
            key = " ".join(text[:120].split())
            index[key] = chunk["paper_id"]
    return index


def _find_paper_ids(reference_contexts: list, chunk_index: dict) -> list:
    found = []
    for ctx in reference_contexts:
        ctx_clean = _strip_hop_tags(ctx)
        ctx_key = " ".join(ctx_clean[:120].split())
        paper_id = chunk_index.get(ctx_key)
        if paper_id is None:
            for prefix_len in (80, 60, 40):
                short_key = " ".join(ctx_clean[:prefix_len].split())
                if not short_key:
                    continue
                for chunk_key, pid in chunk_index.items():
                    if chunk_key.startswith(short_key):
                        paper_id = pid
                        break
                if paper_id:
                    break
        if paper_id and paper_id not in found:
            found.append(paper_id)
    return found


def build_labels(queries: list, chunks_path: str, labels_path: str) -> None:
    """Write labels.jsonl from a list of query dicts that contain reference_contexts."""
    print(f"Building labels from {chunks_path} ...")
    chunk_index = _build_chunk_index(chunks_path)
    print(f"  {len(chunk_index):,} chunks indexed")

    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    matched = unmatched = 0
    with open(labels_path, "w", encoding="utf-8") as out:
        for q in queries:
            ref_contexts = q.get("reference_contexts", [])
            if isinstance(ref_contexts, str):
                ref_contexts = [ref_contexts]
            paper_ids = _find_paper_ids(ref_contexts, chunk_index)
            out.write(json.dumps({"query_id": q["query_id"], "expected_paper_ids": paper_ids},
                                 ensure_ascii=False) + "\n")
            if paper_ids:
                matched += 1
            else:
                unmatched += 1

    print(f"  Labels written: {matched} matched, {unmatched} unmatched → {labels_path}")

load_dotenv()

HEALTH_KEYWORDS = {
    "health", "disease", "clinical", "medical", "patient", "hospital",
    "cancer", "drug", "therapy", "mental", "mortality", "epidemi",
    "infection", "chronic", "physician", "nursing", "diabetes", "obesity",
}
SUSTAIN_KEYWORDS = {
    "sustainability", "climate", "energy", "renewable", "carbon", "emission",
    "environmental", "ecosystem", "biodiversity", "pollution", "greenhouse",
    "solar", "wind", "deforestation", "agriculture", "water", "ecosystem",
}


def detect_domain(text: str) -> str:
    text_lower = text.lower()
    h = sum(1 for k in HEALTH_KEYWORDS if k in text_lower)
    s = sum(1 for k in SUSTAIN_KEYWORDS if k in text_lower)
    if h > s:
        return "healthcare"
    if s > h:
        return "sustainability"
    return "interdisciplinary"


def load_chunks(chunks_path: str, max_docs: int = 200):
    """Load ABSTRACT chunks, balanced between domains."""
    health_docs, sustain_docs, other_docs = [], [], []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("section") != "ABSTRACT":
                continue
            text = chunk.get("text", "").strip()
            if len(text) < 100:
                continue
            domain = detect_domain(text)
            entry = {"text": text, "paper_id": chunk["paper_id"], "chunk_id": chunk["chunk_id"], "domain": domain}
            if domain == "healthcare":
                health_docs.append(entry)
            elif domain == "sustainability":
                sustain_docs.append(entry)
            else:
                other_docs.append(entry)

    per_bucket = max_docs // 3
    selected = health_docs[:per_bucket] + sustain_docs[:per_bucket] + other_docs[:per_bucket]
    return selected[:max_docs]


def to_langchain_docs(chunks: list):
    from langchain_core.documents import Document
    return [
        Document(
            page_content=ch["text"],
            metadata={"paper_id": ch["paper_id"], "chunk_id": ch["chunk_id"], "domain": ch["domain"]},
        )
        for ch in chunks
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--max-docs", type=int, default=200)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_cfg = config["eval"]
    output_path = eval_cfg["queries_path"]
    ragas_model = eval_cfg.get("ragas_model", "gemini-2.5-flash")

    print(f"Loading chunks from {config['chunking']['output_path']}...")
    chunks = load_chunks(config["chunking"]["output_path"], max_docs=args.max_docs)
    print(f"Loaded {len(chunks)} chunks "
          f"({sum(1 for c in chunks if c['domain']=='healthcare')} healthcare, "
          f"{sum(1 for c in chunks if c['domain']=='sustainability')} sustainability, "
          f"{sum(1 for c in chunks if c['domain']=='interdisciplinary')} interdisciplinary)")

    docs = to_langchain_docs(chunks)

    print(f"Initialising RAGAS generator with model: {ragas_model} ...")
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import GoogleEmbeddings
    from ragas.testset import TestsetGenerator
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=ragas_model))
    emb = GoogleEmbeddings(model="gemini-embedding-001")

    generator = TestsetGenerator(llm=llm, embedding_model=emb)

    print(f"Generating {args.n_queries} queries (this may take a few minutes)...")
    testset = generator.generate_with_langchain_docs(docs, testset_size=args.n_queries)
    df = testset.to_pandas()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    all_records = []
    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            query_text = row.get("user_input", row.get("question", ""))
            reference = row.get("reference", row.get("ground_truth", ""))
            ref_contexts = row.get("reference_contexts", row.get("contexts", []))
            if not isinstance(ref_contexts, list):
                ref_contexts = [ref_contexts]

            domain = detect_domain(query_text)
            record = {
                "query_id": f"q{written + 1:03d}",
                "query": query_text,
                "reference": reference,
                "reference_contexts": ref_contexts,
                "domain": domain,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.append(record)
            written += 1

    print(f"Written {written} queries to {output_path}")

    labels_path = eval_cfg.get("labels_path", "eval/labels.jsonl")
    chunks_path = config["retrieval"]["chunks_input"]
    build_labels(all_records, chunks_path, labels_path)


if __name__ == "__main__":
    main()
