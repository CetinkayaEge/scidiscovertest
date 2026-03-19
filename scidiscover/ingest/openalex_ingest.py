#!/usr/bin/env python3
"""
Build an OpenAlex corpus for SciDiscover-LLM.

Appends to:
  - data/raw/papers.jsonl
  - docs/corpus_manifest.csv

Uses the OpenAlex Works API with cursor-based pagination.

Example:
  python -m scidiscover.ingestion.openalex_ingest \
      --queries "machine learning" "deep learning" \
      --from-year 2023 \
      --max-papers 500 \
      --raw-output data/raw/papers.jsonl \
      --manifest-output docs/corpus_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import requests


SESSION = requests.Session()
OPENALEX_BASE = "https://api.openalex.org/works"


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
    if not inverted_index:
        return ""
    word_map = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_map.append((pos, word))
    word_map.sort(key=lambda x: x[0])
    raw_text = " ".join(w[1] for w in word_map)
    return clean_text(raw_text)


def fetch_papers(
    queries: List[str],
    from_year: int,
    max_papers: int,
    email: str,
    is_oa: bool = True,
    has_abstract: bool = True,
) -> List[Dict]:
    collected = []
    seen_ids = set()

    for query in queries:
        if len(collected) >= max_papers:
            break

        print(f"[*] Querying OpenAlex: '{query}'...", file=sys.stderr)
        cursor = "*"

        while len(collected) < max_papers and cursor:
            params = {
                "search": query,
                "filter": (
                    f"is_oa:{str(is_oa).lower()},"
                    f"has_abstract:{str(has_abstract).lower()},"
                    f"publication_year:>{from_year}"
                ),
                "select": "id,title,abstract_inverted_index,publication_year,authorships,primary_location,doi,type",
                "per-page": 200,
                "cursor": cursor,
                "mailto": email,
            }

            try:
                resp = SESSION.get(OPENALEX_BASE, params=params, timeout=30)

                if resp.status_code == 403:
                    print("[-] Rate limit hit. Sleeping 5s...", file=sys.stderr)
                    time.sleep(5)
                    continue

                if resp.status_code != 200:
                    print(f"[-] Error {resp.status_code}: {resp.text}", file=sys.stderr)
                    break

                data = resp.json()
                results = data.get("results", [])
                cursor = data.get("meta", {}).get("next_cursor")

                if not results or not cursor:
                    print("[*] No more results for this query.", file=sys.stderr)
                    break

                for work in results:
                    if len(collected) >= max_papers:
                        break

                    work_id = work.get("id", "")
                    if work_id in seen_ids:
                        continue
                    seen_ids.add(work_id)

                    abstract_text = reconstruct_abstract(
                        work.get("abstract_inverted_index")
                    )

                    if not work.get("title") or len(abstract_text) < 50:
                        continue

                    primary_loc = work.get("primary_location") or {}
                    source_info = primary_loc.get("source") or {}

                    authors = []
                    for authorship in work.get("authorships") or []:
                        author_obj = authorship.get("author") or {}
                        name = author_obj.get("display_name")
                        if name:
                            authors.append(clean_text(name))

                    year_val = work.get("publication_year")
                    try:
                        year_int = int(year_val) if year_val else None
                    except (ValueError, TypeError):
                        year_int = None

                    doi = work.get("doi") or ""

                    collected.append(
                        {
                            "paper_id": work_id,
                            "title": clean_text(work["title"]),
                            "abstract": abstract_text,
                            "year": year_int,
                            "authors": authors,
                            "venue": source_info.get("display_name") or "",
                            "doi": doi,
                            "url": doi or work_id,
                            "source": "openalex",
                            "retrieved_date": date.today().isoformat(),
                            "license_note": primary_loc.get("license") or "",
                        }
                    )

                print(
                    f"    --> Total collected: {len(collected)}/{max_papers}",
                    file=sys.stderr,
                )
                time.sleep(0.2)

            except Exception as exc:
                import traceback

                print(f"[-] Exception: {exc}", file=sys.stderr)
                traceback.print_exc()
                break

    return collected


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path_str: str, records: List[Dict]) -> None:
    ensure_parent(path_str)
    with open(path_str, "a", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_manifest_csv(path_str: str, records: List[Dict]) -> None:
    ensure_parent(path_str)
    fieldnames = ["paper_id", "source", "url/doi", "retrieved_date", "license_note"]
    write_header = not Path(path_str).exists() or Path(path_str).stat().st_size == 0
    with open(path_str, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "paper_id": rec["paper_id"],
                    "source": "openalex",
                    "url/doi": rec["doi"] or rec["url"],
                    "retrieved_date": date.today().isoformat(),
                    "license_note": rec.get("license_note", ""),
                }
            )


class OpenAlexIngestor:
    """Config-driven OpenAlex ingestor.

    Usage:
        config = yaml.safe_load(open("configs/demo.yaml"))
        ingestor = OpenAlexIngestor(config)
        ingestor.run()
    """

    def __init__(self, config: dict):
        self.config = config
        src = config["sources"]["openalex"]
        self.queries = src["queries"]
        self.from_year = src["from_year"]
        self.max_papers = src["max_papers"]
        self.email = src.get("email", "test@example.com")
        self.is_oa = src.get("is_oa", True)
        self.has_abstract = src.get("has_abstract", True)
        self.raw_output = config["corpus"]["raw_output"]
        self.manifest_output = config["corpus"]["manifest_output"]

    def run(self) -> List[Dict]:
        papers = fetch_papers(
            queries=self.queries,
            from_year=self.from_year,
            max_papers=self.max_papers,
            email=self.email,
            is_oa=self.is_oa,
            has_abstract=self.has_abstract,
        )
        papers.sort(key=lambda r: str(r["paper_id"]))

        print(
            f"[OpenAlex] Appending {len(papers)} papers to {self.raw_output}",
            file=sys.stderr,
        )
        append_jsonl(self.raw_output, papers)

        print(
            f"[OpenAlex] Appending manifest to {self.manifest_output}",
            file=sys.stderr,
        )
        append_manifest_csv(self.manifest_output, papers)

        print("[OpenAlex] Done.", file=sys.stderr)
        return papers


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an OpenAlex corpus.")
    parser.add_argument(
        "--queries", nargs="+", required=True, help="Search queries"
    )
    parser.add_argument("--from-year", type=int, required=True)
    parser.add_argument("--max-papers", type=int, default=500)
    parser.add_argument("--email", default="test@example.com")
    parser.add_argument("--raw-output", default="data/raw/papers.jsonl")
    parser.add_argument("--manifest-output", default="docs/corpus_manifest.csv")
    parser.add_argument("--is-oa", action="store_true", default=True)
    parser.add_argument("--has-abstract", action="store_true", default=True)
    args = parser.parse_args()

    print(
        f"[OpenAlex] Fetching up to {args.max_papers} papers...",
        file=sys.stderr,
    )

    papers = fetch_papers(
        queries=args.queries,
        from_year=args.from_year,
        max_papers=args.max_papers,
        email=args.email,
        is_oa=args.is_oa,
        has_abstract=args.has_abstract,
    )

    papers.sort(key=lambda r: str(r["paper_id"]))

    print(
        f"[OpenAlex] Appending {len(papers)} papers to {args.raw_output}",
        file=sys.stderr,
    )
    append_jsonl(args.raw_output, papers)

    print(
        f"[OpenAlex] Appending manifest to {args.manifest_output}",
        file=sys.stderr,
    )
    append_manifest_csv(args.manifest_output, papers)

    print("[OpenAlex] Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
