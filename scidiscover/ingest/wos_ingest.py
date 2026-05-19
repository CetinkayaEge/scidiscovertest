#!/usr/bin/env python3
"""
Web of Science (Starter API) ingestion for SciDiscover.

Appends to:
  - data/raw/papers.jsonl
  - docs/corpus_manifest.csv

Requires:
  - WOS_API_KEY environment variable (Clarivate Developer Portal)

Example:
  python -m scidiscover.ingest.wos_ingest \
      --queries "sustainability" "healthcare" \
      --from-year 2020 \
      --to-year 2024 \
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


WOS_BASE = "https://api.clarivate.com/apis/wos-starter/v1/documents"
PAGE_SIZE = 50  # max allowed by Starter API


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_abstract(hit: Dict) -> str:
    raw = hit.get("abstract", {})
    if isinstance(raw, dict):
        value = raw.get("value", "")
        if isinstance(value, list):
            return clean_text(" ".join(str(v) for v in value))
        return clean_text(str(value) if value else "")
    if isinstance(raw, str):
        return clean_text(raw)
    return ""


def _extract_authors(hit: Dict) -> List[str]:
    names = hit.get("names", {})
    authors_raw = names.get("authors", [])
    result = []
    for a in authors_raw:
        name = a.get("displayName") or a.get("wosStandard") or ""
        name = clean_text(name)
        if name:
            result.append(name)
    return result


def _extract_doi(hit: Dict) -> str:
    identifiers = hit.get("identifiers", {})
    doi = identifiers.get("doi") or ""
    return clean_text(str(doi))


def fetch_papers(
    queries: List[str],
    from_year: int,
    to_year: int,
    max_papers: int,
    api_key: str,
    db: str = "WOS",
) -> List[Dict]:
    session = requests.Session()
    session.headers.update({"X-ApiKey": api_key, "Accept": "application/json"})

    collected: List[Dict] = []
    seen_ids: set = set()

    for query in queries:
        if len(collected) >= max_papers:
            break

        wos_query = f"TS=({query}) AND PY=({from_year}-{to_year})"
        print(f"[*] Querying WoS: '{wos_query}'...", file=sys.stderr)

        page = 1
        while len(collected) < max_papers:
            params = {
                "q": wos_query,
                "db": db,
                "limit": PAGE_SIZE,
                "page": page,
            }

            try:
                resp = session.get(WOS_BASE, params=params, timeout=30)

                if resp.status_code == 429:
                    print("[-] Rate limit hit. Sleeping 10s...", file=sys.stderr)
                    time.sleep(10)
                    continue

                if resp.status_code == 401:
                    print("[-] Invalid API key (401). Set WOS_API_KEY.", file=sys.stderr)
                    return collected

                if resp.status_code != 200:
                    print(f"[-] Error {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
                    break

                data = resp.json()
                hits = data.get("hits", [])
                metadata = data.get("metadata", {})
                total = metadata.get("total", 0)

                if not hits:
                    print("[*] No more results for this query.", file=sys.stderr)
                    break

                for hit in hits:
                    if len(collected) >= max_papers:
                        break

                    uid = hit.get("uid", "")
                    if not uid or uid in seen_ids:
                        continue
                    seen_ids.add(uid)

                    title = clean_text(str(hit.get("title", {}).get("value", "") or ""))
                    abstract = _extract_abstract(hit)

                    if not title or len(abstract) < 50:
                        continue

                    source_info = hit.get("source", {})
                    year_val = source_info.get("publishYear")
                    try:
                        year_int = int(year_val) if year_val else None
                    except (ValueError, TypeError):
                        year_int = None

                    doi = _extract_doi(hit)
                    authors = _extract_authors(hit)
                    venue = clean_text(str(source_info.get("sourceTitle", "") or ""))

                    collected.append({
                        "paper_id": f"wos_{uid}",
                        "title": title,
                        "abstract": abstract,
                        "year": year_int,
                        "authors": authors,
                        "venue": venue,
                        "doi": doi,
                        "url": f"https://doi.org/{doi}" if doi else f"https://www.webofscience.com/wos/woscc/full-record/{uid}",
                        "source": "wos",
                        "retrieved_date": date.today().isoformat(),
                        "license_note": "",
                    })

                print(
                    f"    --> Page {page} | batch={len(hits)} | total_available={total} | collected={len(collected)}/{max_papers}",
                    file=sys.stderr,
                )

                fetched_so_far = page * PAGE_SIZE
                if fetched_so_far >= total:
                    break

                page += 1
                time.sleep(0.5)

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
            writer.writerow({
                "paper_id": rec["paper_id"],
                "source": "wos",
                "url/doi": rec["doi"] or rec["url"],
                "retrieved_date": date.today().isoformat(),
                "license_note": rec.get("license_note", ""),
            })


class WoSIngestor:
    """Config-driven Web of Science Starter API ingestor.

    Usage:
        config = yaml.safe_load(open("configs/demo.yaml"))
        ingestor = WoSIngestor(config)
        ingestor.run()
    """

    def __init__(self, config: dict):
        self.config = config
        src = config["sources"]["wos"]
        self.queries = src["queries"]
        self.from_year = src["from_year"]
        self.to_year = src.get("to_year", date.today().year)
        self.max_papers = src["max_papers"]
        self.db = src.get("db", "WOS")
        self.api_key = os.getenv("WOS_API_KEY", "")
        self.raw_output = config["corpus"]["raw_output"]
        self.manifest_output = config["corpus"]["manifest_output"]

    def run(self) -> List[Dict]:
        if not self.api_key:
            print("[WoS] WOS_API_KEY not set — skipping WoS ingestion.", file=sys.stderr)
            return []

        papers = fetch_papers(
            queries=self.queries,
            from_year=self.from_year,
            to_year=self.to_year,
            max_papers=self.max_papers,
            api_key=self.api_key,
            db=self.db,
        )
        papers.sort(key=lambda r: str(r["paper_id"]))

        print(f"[WoS] Appending {len(papers)} papers to {self.raw_output}", file=sys.stderr)
        append_jsonl(self.raw_output, papers)

        print(f"[WoS] Appending manifest to {self.manifest_output}", file=sys.stderr)
        append_manifest_csv(self.manifest_output, papers)

        print("[WoS] Done.", file=sys.stderr)
        return papers


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest papers from Web of Science Starter API.")
    parser.add_argument("--queries", nargs="+", required=True, help="Search topics")
    parser.add_argument("--from-year", type=int, required=True)
    parser.add_argument("--to-year", type=int, default=date.today().year)
    parser.add_argument("--max-papers", type=int, default=500)
    parser.add_argument("--db", default="WOS", help="WoS database code (default: WOS)")
    parser.add_argument("--api-key", default="", help="WoS API key (overrides WOS_API_KEY env var)")
    parser.add_argument("--raw-output", default="data/raw/papers.jsonl")
    parser.add_argument("--manifest-output", default="docs/corpus_manifest.csv")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("WOS_API_KEY", "")
    if not api_key:
        print("[-] No API key provided. Set WOS_API_KEY or use --api-key.", file=sys.stderr)
        return 1

    print(f"[WoS] Fetching up to {args.max_papers} papers...", file=sys.stderr)

    papers = fetch_papers(
        queries=args.queries,
        from_year=args.from_year,
        to_year=args.to_year,
        max_papers=args.max_papers,
        api_key=api_key,
        db=args.db,
    )
    papers.sort(key=lambda r: str(r["paper_id"]))

    print(f"[WoS] Appending {len(papers)} papers to {args.raw_output}", file=sys.stderr)
    append_jsonl(args.raw_output, papers)

    print(f"[WoS] Appending manifest to {args.manifest_output}", file=sys.stderr)
    append_manifest_csv(args.manifest_output, papers)

    print("[WoS] Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
