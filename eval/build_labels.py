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


def build_chunk_index(chunks_path: str) -> tuple[dict, dict]:
    """
    Return two indexes keyed by normalised 120-char prefix:
      full_index  — chunk text as-is        (title-prefixed, new RAGAS format)
      bare_index  — title prefix stripped   (raw abstract, old RAGAS format)
    Each value is {"paper_id": ..., "chunk_id": ...}.
    """
    full_index: dict[str, dict] = {}
    bare_index: dict[str, dict] = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            text = chunk.get("text", "").strip()
            if not text:
                continue
            entry = {"paper_id": chunk["paper_id"], "chunk_id": chunk["chunk_id"]}
            full_index[" ".join(text[:120].split())] = entry
            bare_text = text.split(" | ", 1)[1] if " | " in text else text
            bare_index[" ".join(bare_text[:120].split())] = entry
    return full_index, bare_index


def _lookup(ctx_clean: str, index: dict) -> dict | None:
    """Try exact 120-char match, then progressively shorter prefixes."""
    for prefix_len in (120, 80, 60, 40):
        key = " ".join(ctx_clean[:prefix_len].split())
        if not key:
            continue
        if key in index:
            return index[key]
        for chunk_key, entry in index.items():
            if chunk_key.startswith(key):
                return entry
    return None


def find_ids(reference_contexts: list,
             full_index: dict,
             bare_index: dict) -> tuple[list[str], list[str]]:
    """
    Return (expected_paper_ids, expected_chunk_ids) matched from reference_contexts.
    Tries full index first, falls back to bare index.
    """
    paper_ids: list[str] = []
    chunk_ids: list[str] = []
    for ctx in reference_contexts:
        ctx_clean = strip_hop_tags(ctx)
        entry = _lookup(ctx_clean, full_index) or _lookup(ctx_clean, bare_index)
        if entry:
            if entry["paper_id"] not in paper_ids:
                paper_ids.append(entry["paper_id"])
            if entry["chunk_id"] not in chunk_ids:
                chunk_ids.append(entry["chunk_id"])
    return paper_ids, chunk_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/queries.jsonl")
    parser.add_argument("--chunks",  default="data/processed/chunks.jsonl")
    parser.add_argument("--output",  default="eval/labels.jsonl")
    args = parser.parse_args()

    print(f"Building chunk index from {args.chunks} ...")
    full_index, bare_index = build_chunk_index(args.chunks)
    print(f"  {len(full_index):,} chunks indexed (full + bare prefix)")

    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"  {len(queries)} queries loaded")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    matched = unmatched = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for q in queries:
            qtype = q.get("query_type", "")

            # unanswerable / out_of_domain — correct answer is nothing retrieved
            if qtype in ("unanswerable", "out_of_domain"):
                record = {
                    "query_id":           q["query_id"],
                    "expected_paper_ids": [],
                    "expected_chunk_ids": [],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                matched += 1
                print(f"  – {q['query_id']} [{qtype}] → []")
                continue

            # prefer IDs already embedded in the query record (new generator)
            paper_ids = q.get("expected_paper_ids") or []
            chunk_ids = q.get("expected_chunk_ids") or []

            # fall back to text matching for queries without pre-filled IDs
            if not paper_ids:
                ref_contexts = q.get("reference_contexts", [])
                if isinstance(ref_contexts, str):
                    ref_contexts = [ref_contexts]
                paper_ids, chunk_ids = find_ids(ref_contexts, full_index, bare_index)

            record = {
                "query_id":           q["query_id"],
                "expected_paper_ids": paper_ids,
                "expected_chunk_ids": chunk_ids,
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
