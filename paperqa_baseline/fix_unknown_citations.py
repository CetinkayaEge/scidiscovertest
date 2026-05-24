"""Post-process paperqa_results.json to fix `unknownauthorsUnknownyear*` docnames.

PaperQA2's LLM citation extraction fails on our `Title: X\nAuthors: Y\n...` .txt
format and falls back to "Unknown, file, year" → docnames like
`unknownauthorsUnknownyearcarbondotsa`.  The fix is purely cosmetic — paper_recall
matches via DOI (unaffected) and RAGAS uses passage text (also unaffected), but
the answer text and evidence_passages[].doc_name fields are ugly.

This script:
  1. Builds DOI → (proper_docname, proper_citation) from data/raw/papers.jsonl
  2. For each query in paperqa_results.json:
     - Rewrites evidence_passages[].doc_name based on its DOI
     - Rewrites the citations list (dedup-preserving order)
     - Replaces every `(olddocname chunk N)` occurrence in answer text
     - Rewrites the "References" section lines so each entry uses proper APA
  3. Writes back in-place (a backup at .before-citation-fix.json is assumed to
     already exist).

Idempotent — running twice produces the same output.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).parent.parent
PAPERS_PATH = REPO / "data" / "raw" / "papers.jsonl"
RESULTS_PATH = REPO / "paperqa_baseline" / "results" / "paperqa_results.json"

_DOI_URL_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


def normalize_doi(d: str) -> str:
    return _DOI_URL_PREFIX.sub("", (d or "").strip()).lower()


def proper_docname(authors: list[str], year, title: str) -> str:
    """Mimic PaperQA2's citation_to_docname: lowercase last-name + year + titleword."""
    if not authors or not year or not title:
        return ""
    last = authors[0].strip().split()[-1].lower()
    last = re.sub(r"[^a-z]", "", last)
    titleword = re.sub(r"[^a-z]", "", title.lower())[:20]
    return f"{last}{year}{titleword}" if last and titleword else ""


def proper_citation(authors: list[str], year, title: str, journal: str, doi: str) -> str:
    if not authors:
        a = "Anonymous"
    elif len(authors) > 3:
        a = ", ".join(authors[:3]) + ", et al."
    else:
        a = ", ".join(authors)
    parts = [f"{a} ({year or 'n.d.'})", title]
    if journal:
        parts.append(journal)
    if doi:
        parts.append(f"doi:{doi}")
    return ". ".join(parts) + "."


def build_doi_map(papers_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(papers_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            doi = normalize_doi(p.get("doi") or "")
            if not doi:
                continue
            authors = p.get("authors") or []
            year = p.get("year")
            title = (p.get("title") or "").strip()
            journal = (p.get("venue") or "").strip()
            dn = proper_docname(authors, year, title)
            if not dn:
                continue
            out[doi] = {
                "docname": dn,
                "citation": proper_citation(authors, year, title, journal, doi),
            }
    return out


def main() -> None:
    doi_map = build_doi_map(PAPERS_PATH)
    print(f"Built DOI→meta map: {len(doi_map):,} papers")

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    n_queries = n_eps_fixed = n_inline_replaced = n_refs_rewritten = 0
    n_no_doi_match = 0

    for q in data["per_query"]:
        n_queries += 1
        # 1. Build per-query old_docname → (new_docname, new_citation) by DOI lookup
        old_to_new: dict[str, dict[str, str]] = {}
        for ep in q.get("evidence_passages") or []:
            doi = normalize_doi(ep.get("doi") or "")
            old = ep.get("doc_name") or ""
            if not doi or not old.startswith("unknownauthors"):
                continue
            meta = doi_map.get(doi)
            if not meta:
                n_no_doi_match += 1
                continue
            new = meta["docname"]
            ep["doc_name"] = new
            n_eps_fixed += 1
            old_to_new[old] = meta  # may overwrite earlier; same DOI → same meta

        if not old_to_new:
            continue

        # 2. Rewrite citations list (preserve order, dedup)
        if "citations" in q and isinstance(q["citations"], list):
            seen: set[str] = set()
            new_list: list[str] = []
            for c in q["citations"]:
                replaced = old_to_new.get(c, {"docname": c})["docname"]
                if replaced not in seen:
                    seen.add(replaced)
                    new_list.append(replaced)
            q["citations"] = new_list

        # 3. Rewrite answer text
        ans = q.get("answer") or ""
        # 3a. Replace inline (old chunk N) → (new chunk N)
        for old, meta in old_to_new.items():
            new_dn = meta["docname"]
            # match "(old chunk N)" or "(old)" patterns
            n_before = ans.count(old)
            ans = ans.replace(old, new_dn)
            n_inline_replaced += n_before

        # 3b. Rewrite References section lines.  Format from PaperQA2:
        #   "1. (newdocname chunk N): Unknown authors. <title>. Unknown journal, Unknown year. URL: ..."
        # Replace whatever comes after "...): " with our proper APA citation.
        for meta in old_to_new.values():
            new_dn = meta["docname"]
            new_cit = meta["citation"]
            pattern = re.compile(
                rf"(^\s*\d+\.\s*\({re.escape(new_dn)}\s+chunk\s+\d+\)):\s*[^\n]*",
                re.MULTILINE,
            )
            ans, n_sub = pattern.subn(rf"\1: {new_cit}", ans)
            n_refs_rewritten += n_sub

        q["answer"] = ans

    RESULTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nQueries processed: {n_queries}")
    print(f"evidence_passages[].doc_name fixed: {n_eps_fixed}")
    print(f"Inline `(unknown... chunk N)` references replaced: {n_inline_replaced}")
    print(f"Reference-section lines rewritten:   {n_refs_rewritten}")
    print(f"Citations with DOI but no map match: {n_no_doi_match}")
    print(f"\nUpdated: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
