"""
Build data/eval/labels.jsonl from data/eval/queries.jsonl.

The new query format already contains expected_paper_ids and expected_chunk_ids,
so this script simply extracts those fields.

Usage:
    PYTHONPATH=. python data/eval/build_labels.py
    PYTHONPATH=. python data/eval/build_labels.py --queries data/eval/queries.jsonl \
                                                   --output  data/eval/labels.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--output",  default="data/eval/labels.jsonl")
    args = parser.parse_args()

    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"{len(queries)} queries loaded from {args.queries}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for q in queries:
            record = {
                "query_id": q["id"],
                "expected_paper_ids": q.get("expected_paper_ids", []),
                "expected_chunk_ids": q.get("expected_chunk_ids", []),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"labels.jsonl written to {args.output} ({len(queries)} records)")


if __name__ == "__main__":
    main()
