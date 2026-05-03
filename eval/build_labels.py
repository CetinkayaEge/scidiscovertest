"""
Build eval/labels.jsonl from eval/queries.jsonl by matching reference_contexts
text against chunks.jsonl to recover expected paper_ids.

Usage:
    PYTHONPATH=. python eval/build_labels.py
    PYTHONPATH=. python eval/build_labels.py --queries eval/queries.jsonl \
                                              --chunks  data/processed/chunks.jsonl \
                                              --output  eval/labels.jsonl
"""

import argparse
import json
import re
from pathlib import Path


def strip_hop_tags(text: str) -> str:
    """Remove '<N-hop>' prefixes that RAGAS injects into multi-hop contexts."""
    return re.sub(r"^<\d+-hop>\s*", "", text.strip())


def build_chunk_index(chunks_path: str) -> dict[str, str]:
    """Return {normalised_text_prefix: paper_id} for every chunk."""
    index: dict[str, str] = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            text = chunk.get("text", "").strip()
            if not text:
                continue
            key = " ".join(text[:120].split())
            index[key] = chunk["paper_id"]
    return index


def find_paper_ids(reference_contexts: list, chunk_index: dict[str, str]) -> list[str]:
    """
    For each context string, match it to a chunk and return the
    deduplicated list of paper_ids found.
    """
    found: list[str] = []
    for ctx in reference_contexts:
        ctx_clean = strip_hop_tags(ctx)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/queries.jsonl")
    parser.add_argument("--chunks",  default="data/processed/chunks.jsonl")
    parser.add_argument("--output",  default="eval/labels.jsonl")
    args = parser.parse_args()

    print(f"Building chunk index from {args.chunks} ...")
    chunk_index = build_chunk_index(args.chunks)
    print(f"  {len(chunk_index):,} chunks indexed")

    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"  {len(queries)} queries loaded")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    matched = unmatched = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for q in queries:
            ref_contexts = q.get("reference_contexts", [])
            if isinstance(ref_contexts, str):
                ref_contexts = [ref_contexts]

            paper_ids = find_paper_ids(ref_contexts, chunk_index)

            record = {
                "query_id": q["query_id"],
                "expected_paper_ids": paper_ids,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if paper_ids:
                matched += 1
                print(f"  ✓ {q['query_id']} → {paper_ids}")
            else:
                unmatched += 1
                print(f"  ✗ {q['query_id']} → NO MATCH")

    print(f"\nDone: {matched} matched, {unmatched} unmatched → {args.output}")
    if unmatched:
        print(f"  WARNING: {unmatched} queries have no expected_paper_ids — "
              "Recall@k will be None for those.")


if __name__ == "__main__":
    main()
