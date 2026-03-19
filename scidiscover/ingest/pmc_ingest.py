#!/usr/bin/env python3
"""
Build a PMC Open Access corpus for SciDiscover-LLM.

Outputs:
  - data/raw/papers.jsonl
  - docs/corpus_manifest.csv

What this script does:
  1) Uses the PMC OA Web Service API to enumerate PMC OA articles
  2) Uses PMC OAI-PMH (pmc_fm) to fetch front-matter metadata
  3) Writes canonical JSONL + corpus manifest CSV

Example:
  python pmc_ingest.py \
      --from-date 2024-01-01 \
      --until-date 2024-12-31 \
      --max-papers 1000 \
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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests


OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
OAI_BASE = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
SESSION = requests.Session()


@dataclass
class OARecord:
    pmcid: str
    license_note: str
    retracted: str
    citation: str
    url: str
    doi: str
    pdf_url: str
    tgz_url: str


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_from_elem(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return clean_text(" ".join(elem.itertext()))


def safe_request(
    url: str,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    sleep_sec: float = 0.34,
) -> requests.Response:
    """
    Small politeness delay + retry loop.
    """
    last_exc = None
    for attempt in range(5):
        try:
            time.sleep(sleep_sec)
            resp = SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"Request failed for {url} params={params}: {last_exc}")


def parse_oa_response(xml_text: str) -> Tuple[List[OARecord], Optional[str]]:
    """
    Parse PMC OA API response.
    """
    root = ET.fromstring(xml_text)
    records: List[OARecord] = []

    records_elem = root.find("records")
    if records_elem is not None:
        for rec in records_elem.findall("record"):
            pmcid = rec.attrib.get("id", "").strip()
            if not pmcid:
                continue

            citation = clean_text(rec.attrib.get("citation", ""))
            license_note = clean_text(rec.attrib.get("license", ""))
            retracted = clean_text(rec.attrib.get("retracted", ""))
            doi = ""
            pdf_url = ""
            tgz_url = ""

            for link in rec.findall("link"):
                fmt = clean_text(link.attrib.get("format", "")).lower()
                href = clean_text(link.attrib.get("href", ""))
                if fmt == "pdf":
                    pdf_url = href
                elif fmt == "tgz":
                    tgz_url = href

            # DOI is not always present in OA API response, so leave blank here.
            # We will try to get it from OAI-PMH metadata.
            url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

            records.append(
                OARecord(
                    pmcid=pmcid,
                    license_note=license_note,
                    retracted=retracted,
                    citation=citation,
                    url=url,
                    doi=doi,
                    pdf_url=pdf_url,
                    tgz_url=tgz_url,
                )
            )

    resumption_token = None
    # Newer docs show <resumption><link token="..."/></resumption>
    resumption = root.find("resumption")
    if resumption is not None:
        link = resumption.find("link")
        if link is not None:
            resumption_token = clean_text(link.attrib.get("token", "")) or None

    # Older variants may expose a direct <resumptionToken> element
    if not resumption_token:
        token_elem = root.find("resumptionToken")
        if token_elem is not None and token_elem.text:
            resumption_token = clean_text(token_elem.text) or None

    return records, resumption_token


from datetime import datetime, timedelta


def list_oa_records(
    from_date: str,
    until_date: Optional[str],
    max_records: int,
    require_pdf: bool = False,
) -> List[OARecord]:

    start = datetime.strptime(from_date, "%Y-%m-%d")

    if until_date:
        end = datetime.strptime(until_date, "%Y-%m-%d")
    else:
        end = datetime.today()

    all_records: List[OARecord] = []
    seen: set[str] = set()

    current = start

    while current < end and len(all_records) < max_records:

        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

        params: Dict[str, str] = {
            "from": current.strftime("%Y-%m-%d"),
            "until": next_month.strftime("%Y-%m-%d"),
        }

        if require_pdf:
            params["format"] = "pdf"

        print(f"\nDEBUG Month Request → {params}", file=sys.stderr)

        resp = safe_request(OA_BASE, params=params)
        batch, _ = parse_oa_response(resp.text)

        print(f"DEBUG Received → {len(batch)} records", file=sys.stderr)

        for rec in batch:

            if rec.pmcid in seen:
                continue

            seen.add(rec.pmcid)
            all_records.append(rec)

            if len(all_records) >= max_records:
                break

        current = next_month

    return all_records[:max_records]


def oai_get_pmc_front_matter_xml(pmc_numeric_id: str) -> ET.Element:
    """
    Fetch PMC OAI-PMH front-matter metadata (pmc_fm) for a PMCID numeric ID.
    Example identifier: oai:pubmedcentral.nih.gov:7840891
    """
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:pubmedcentral.nih.gov:{pmc_numeric_id}",
        "metadataPrefix": "pmc_fm",
    }
    resp = safe_request(OAI_BASE, params=params)
    return ET.fromstring(resp.text)


def find_first(root: ET.Element, names: Iterable[str]) -> Optional[ET.Element]:
    for name in names:
        elem = root.find(name)
        if elem is not None:
            return elem
    return None


def findall_anywhere(root: ET.Element, local_tag: str) -> List[ET.Element]:
    """
    Namespace-agnostic search by local tag name.
    """
    out = []
    for elem in root.iter():
        if elem.tag.split("}")[-1] == local_tag:
            out.append(elem)
    return out


def findfirst_anywhere(root: ET.Element, local_tag: str) -> Optional[ET.Element]:
    elems = findall_anywhere(root, local_tag)
    return elems[0] if elems else None


def parse_pmc_front_matter(root: ET.Element) -> Dict[str, object]:
    """
    Parse useful fields from OAI-PMH pmc_fm XML.
    Tries to be namespace-agnostic.
    """
    title = ""
    abstract = ""
    authors: List[str] = []
    year = ""
    venue = ""
    doi = ""

    # article-title
    title_elem = findfirst_anywhere(root, "article-title")
    if title_elem is not None:
        title = text_from_elem(title_elem)

    # abstract (may have multiple paragraphs)
    abstract_elems = findall_anywhere(root, "abstract")
    if abstract_elems:
        abstract_parts = [text_from_elem(e) for e in abstract_elems]
        abstract = clean_text(" ".join(p for p in abstract_parts if p))

    # journal title / venue
    journal_title_elem = findfirst_anywhere(root, "journal-title")
    if journal_title_elem is not None:
        venue = text_from_elem(journal_title_elem)

    # publication year
    pub_date_elems = findall_anywhere(root, "pub-date")
    for pub in pub_date_elems:
        year_elem = None
        for child in pub.iter():
            if child.tag.split("}")[-1] == "year":
                year_elem = child
                break
        if year_elem is not None and text_from_elem(year_elem):
            year = text_from_elem(year_elem)
            break

    if not year:
        year_elem = findfirst_anywhere(root, "year")
        if year_elem is not None:
            year = text_from_elem(year_elem)

    # DOI
    for aid in findall_anywhere(root, "article-id"):
        if clean_text(aid.attrib.get("pub-id-type", "")).lower() == "doi":
            doi = text_from_elem(aid)
            if doi:
                break

    # Authors
    contribs = findall_anywhere(root, "contrib")
    for contrib in contribs:
        if clean_text(contrib.attrib.get("contrib-type", "")).lower() != "author":
            continue

        surname = ""
        given = ""
        for child in contrib.iter():
            local = child.tag.split("}")[-1]
            if local == "surname":
                surname = text_from_elem(child)
            elif local == "given-names":
                given = text_from_elem(child)

        full = clean_text(f"{given} {surname}")
        if full:
            authors.append(full)

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
    }


def pmcid_to_numeric(pmcid: str) -> str:
    return pmcid.replace("PMC", "").strip()


def build_canonical_record(oa_rec: OARecord, meta: Dict[str, object]) -> Dict[str, object]:
    pmcid = oa_rec.pmcid
    year_val = meta.get("year", "")
    try:
        year_int = int(str(year_val)[:4]) if str(year_val).strip() else None
    except ValueError:
        year_int = None

    doi = clean_text(str(meta.get("doi", "") or oa_rec.doi))
    title = clean_text(str(meta.get("title", "")))
    abstract = clean_text(str(meta.get("abstract", "")))
    venue = clean_text(str(meta.get("venue", "")))
    authors = [clean_text(str(a)) for a in meta.get("authors", []) if clean_text(str(a))]

    return {
        "paper_id": f"pmc_{pmcid}",
        "title": title,
        "abstract": abstract,
        "year": year_int,
        "authors": authors,
        "venue": venue,
        "doi": doi,
        "url": oa_rec.url,
        "source": "pmc",
        "retrieved_date": date.today().isoformat(),
        "license_note": oa_rec.license_note or "PMC OA subset article",
    }


def build_manifest_row(oa_rec: OARecord, canonical: Dict[str, object]) -> Dict[str, str]:
    doi = clean_text(str(canonical.get("doi", "")))
    return {
        "paper_id": str(canonical["paper_id"]),
        "source": "pmc",
        "url/doi": doi if doi else oa_rec.url,
        "retrieved_date": date.today().isoformat(),
        "license_note": oa_rec.license_note or "PMC OA subset article",
    }


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path_str: str, records: List[Dict[str, object]]) -> None:
    ensure_parent(path_str)
    with open(path_str, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_manifest_csv(path_str: str, rows: List[Dict[str, str]]) -> None:
    ensure_parent(path_str)
    fieldnames = ["paper_id", "source", "url/doi", "retrieved_date", "license_note"]
    with open(path_str, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PMC OA corpus.")
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--until-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--max-papers", type=int, default=1000)
    parser.add_argument("--require-pdf", action="store_true", help="Only keep OA records with PDFs")
    parser.add_argument("--raw-output", default="data/raw/papers.jsonl")
    parser.add_argument("--manifest-output", default="docs/corpus_manifest.csv")
    parser.add_argument("--skip-empty-abstract", action="store_true")
    args = parser.parse_args()

    print(f"[1/4] Listing PMC OA records from {args.from_date}...", file=sys.stderr)
    oa_records = list_oa_records(
        from_date=args.from_date,
        until_date=args.until_date,
        max_records=args.max_papers,
        require_pdf=args.require_pdf,
    )
    print(f"  Retrieved {len(oa_records)} OA records", file=sys.stderr)

    canonical_records: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, str]] = []

    print("[2/4] Fetching front-matter metadata via PMC OAI-PMH...", file=sys.stderr)
    for idx, oa_rec in enumerate(oa_records, start=1):
        try:
            numeric_id = pmcid_to_numeric(oa_rec.pmcid)
            root = oai_get_pmc_front_matter_xml(numeric_id)
            meta = parse_pmc_front_matter(root)
            canonical = build_canonical_record(oa_rec, meta)

            if args.skip_empty_abstract and not canonical["abstract"]:
                continue

            # Keep only records with the minimum useful metadata
            if not canonical["title"]:
                continue

            canonical_records.append(canonical)
            manifest_rows.append(build_manifest_row(oa_rec, canonical))

            if idx % 50 == 0:
                print(f"  Processed {idx}/{len(oa_records)}", file=sys.stderr)

        except Exception as exc:
            print(f"  WARN: failed for {oa_rec.pmcid}: {exc}", file=sys.stderr)

    # Deterministic ordering
    canonical_records.sort(key=lambda r: str(r["paper_id"]))
    manifest_rows.sort(key=lambda r: r["paper_id"])

    print(f"[3/4] Writing {len(canonical_records)} papers to {args.raw_output}", file=sys.stderr)
    write_jsonl(args.raw_output, canonical_records)

    print(f"[4/4] Writing manifest to {args.manifest_output}", file=sys.stderr)
    write_manifest_csv(args.manifest_output, manifest_rows)

    print("Done.", file=sys.stderr)
    print(f"papers.jsonl: {args.raw_output}", file=sys.stderr)
    print(f"manifest:     {args.manifest_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())