"""İndekslenmiş bir paper'ın DocDetails'ini kontrol et.

PaperQA2 docs/<hash>.zip dosyaları aslında zip değil — zlib-compressed pickled
Docs() objesi.  Bu script onu açıp indexlenmiş paper'ın gerçek metadata'sını
gösterir (docname, citation, authors, year, doi, title).

Kullanım:
    python paperqa_baseline/inspect_indexed_paper.py <doi-or-paper_id>

Örnek:
    python paperqa_baseline/inspect_indexed_paper.py 10.3390/recycling7030032
    python paperqa_baseline/inspect_indexed_paper.py https://openalex.org/W4281620845
    python paperqa_baseline/inspect_indexed_paper.py pmc_PMC10004353
"""
import hashlib
import json
import pickle
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).parent.parent
PAPERS = REPO / "data" / "raw" / "papers.jsonl"
INDEX_DIR = REPO / "paperqa_baseline" / "pqa_index" / "full"
TEXTS_DIR = REPO / "paperqa_baseline" / "paper_texts" / "full"

sys.path.insert(0, str(REPO))
from paperqa_baseline.run_paperqa import sanitize_paper_id  # noqa: E402


def find_paper(needle: str) -> dict | None:
    """needle DOI olabilir (10.x/y), full URL olabilir, ya da paper_id olabilir."""
    needle_low = needle.lower().strip()
    with open(PAPERS, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if needle == p.get("paper_id"):
                return p
            doi = (p.get("doi") or "").lower()
            if needle_low and needle_low in doi:
                return p
    return None


def main(needle: str) -> None:
    paper = find_paper(needle)
    if not paper:
        print(f"❌ '{needle}' papers.jsonl'da bulunamadı"); return

    print("=== papers.jsonl ===")
    print(f"  paper_id: {paper['paper_id']}")
    print(f"  title:    {paper['title'][:90]}")
    print(f"  authors:  {(paper.get('authors') or [])[:3]}")
    print(f"  year:     {paper.get('year')}")
    print(f"  doi:      {paper.get('doi')}")

    txt = TEXTS_DIR / f"{sanitize_paper_id(paper['paper_id'])}.txt"
    print(f"\n=== .txt ===")
    print(f"  {txt}")
    print(f"  exists: {txt.exists()}, size: {txt.stat().st_size if txt.exists() else 0} bytes")
    if not txt.exists():
        return
    md5 = hashlib.md5(txt.read_bytes()).hexdigest()
    print(f"  md5:    {md5}")

    zips = list(INDEX_DIR.rglob(f"{md5}.zip"))
    print(f"\n=== indexed .zip ===")
    if not zips:
        print(f"  ❌ {md5}.zip bulunamadı — henüz indekslenmemiş")
        return
    z = zips[0]
    print(f"  {z}")
    print(f"  size: {z.stat().st_size} bytes")

    docs = pickle.loads(zlib.decompress(z.read_bytes()))
    print(f"\n=== DocDetails ===")
    print(f"  internal docs entries: {len(docs.docs)}")
    for dockey, doc in docs.docs.items():
        dn = doc.docname or ""
        marker = "✅" if dn and "unknown" not in dn.lower() else "❌"
        print(f"  docname:  {dn} {marker}")
        print(f"  citation: {(doc.citation or '')[:140]}")
        if hasattr(doc, "authors"): print(f"  authors:  {doc.authors[:3] if doc.authors else None}")
        if hasattr(doc, "year"):    print(f"  year:     {doc.year}")
        if hasattr(doc, "doi"):     print(f"  doi:      {doc.doi}")
        break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
