"""PaperQA2 SOTA baseline runner.

Strategy: write each paper's title + abstract as a .txt file in
paperqa_baseline/paper_texts/, then point PaperQA2 at the directory via
Settings(agent=AgentSettings(index=IndexSettings(paper_directory=...))).
PaperQA2 builds and caches its own vector index under paperqa_baseline/pqa_index/
on the first run; subsequent runs reuse the cached index.

Usage (local, dry-run — first 2 queries, ~2 reference papers):
    PYTHONPATH=. python paperqa_baseline/run_paperqa.py --dry-run

Usage (dry-run + RAGAS evaluation):
    PYTHONPATH=. python paperqa_baseline/run_paperqa.py --dry-run --run-ragas

Usage (full run — all 40 queries, ~22 K papers):
    PYTHONPATH=. python paperqa_baseline/run_paperqa.py

Usage (skip env override if OPENAI_API_KEY is already set correctly):
    PYTHONPATH=. python paperqa_baseline/run_paperqa.py --skip-env-override
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── canonical paths ─────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
REPO_ROOT = _HERE.parent
DEFAULT_PAPERS_PATH = REPO_ROOT / "data" / "raw" / "papers.jsonl"
DEFAULT_QUERIES_PATH = REPO_ROOT / "eval" / "queries.jsonl"
PAPER_TEXTS_ROOT = _HERE / "paper_texts"
PQA_INDEX_ROOT = _HERE / "pqa_index"
RESULTS_DIR = _HERE / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "paperqa_results.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def sanitize_paper_id(paper_id: str) -> str:
    """Return a filesystem-safe version of paper_id (used as .txt filename)."""
    return (
        paper_id
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def load_papers(papers_path: Path) -> dict[str, dict[str, Any]]:
    """Load papers.jsonl into a dict keyed by paper_id."""
    papers: dict[str, dict[str, Any]] = {}
    with open(papers_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            papers[paper["paper_id"]] = paper
    logger.info("Loaded %d papers from %s", len(papers), papers_path)
    return papers


def load_queries(
    queries_path: Path,
    max_queries: int | None = None,
) -> list[dict[str, Any]]:
    """Load eval/queries.jsonl and optionally cap the list."""
    queries: list[dict[str, Any]] = []
    with open(queries_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    if max_queries is not None:
        queries = queries[:max_queries]
    logger.info("Loaded %d queries from %s", len(queries), queries_path)
    return queries


def write_paper_texts(
    papers: dict[str, dict[str, Any]],
    output_dir: Path,
    paper_ids: set[str] | None = None,
) -> int:
    """Write each paper as a plain-text file for PaperQA2 ingestion.

    The filename is ``sanitize_paper_id(paper_id) + ".txt"`` so that
    PaperQA2's docname maps back to the paper_id unambiguously.

    Args:
        papers:     paper_id → paper record loaded from papers.jsonl.
        output_dir: Directory to write .txt files into (created if missing).
        paper_ids:  If given, only write these IDs (used for --dry-run).

    Returns:
        Number of files present in output_dir after the call.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_to_write = paper_ids if paper_ids is not None else set(papers.keys())
    written = skipped = 0

    for paper_id in ids_to_write:
        if paper_id not in papers:
            continue
        paper = papers[paper_id]
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue

        safe_name = sanitize_paper_id(paper_id)
        txt_path = output_dir / f"{safe_name}.txt"
        if txt_path.exists():
            skipped += 1
            continue

        title = paper.get("title", "")
        authors = ", ".join(paper.get("authors") or [])
        year = paper.get("year", "")
        doi = paper.get("doi", "")
        venue = paper.get("venue", "")

        content = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Year: {year}\n"
            f"Venue: {venue}\n"
            f"DOI: {doi}\n"
            f"Paper ID: {paper_id}\n"
            f"\nAbstract:\n{abstract}\n"
        )
        txt_path.write_text(content, encoding="utf-8")
        written += 1

    total = written + skipped
    logger.info(
        "Paper texts: %d newly written, %d already cached → %d total in %s",
        written, skipped, total, output_dir,
    )
    return total


def write_manifest(
    papers: dict[str, dict[str, Any]],
    paper_texts_dir: Path,
    manifest_path: Path,
    paper_ids: set[str] | None = None,
) -> int:
    """Write the manifest CSV PaperQA2 reads to skip external metadata lookups.

    PaperQA2 reads ALL columns in the manifest CSV (despite the IndexSettings
    docstring saying only file_location/doi/title are used) and passes them to
    DocDetails(**row).  When we include `authors`, `year`, and `journal`,
    DocDetails auto-builds a proper APA citation + docname (e.g.
    "ren2024carbondotsa") — so PaperQA2 skips its LLM citation-peek step and
    answer references are no longer "Unknown authors. Unknown year. ...".

    Without this, the LLM peek on our "Title: X / Authors: Y" .txt format
    returns "insufficient information" and PaperQA2 falls back to
    "Unknown, <file>, <year>" → every docname becomes
    `unknownauthorsUnknownyear<titleword>`.

    Authors are JSON-encoded so csv.DictReader → pydantic parses them as a
    list[str].
    """
    ids = paper_ids if paper_ids is not None else set(papers.keys())
    rows: list[dict[str, str]] = []
    for pid in ids:
        paper = papers.get(pid)
        if not paper or not (paper.get("abstract") or "").strip():
            continue
        authors = paper.get("authors") or []
        rows.append({
            "file_location": f"{sanitize_paper_id(pid)}.txt",
            "doi": _normalize_doi(paper.get("doi") or ""),
            "title": (paper.get("title") or "").replace("\n", " ").strip(),
            "authors": json.dumps(authors, ensure_ascii=False) if authors else "",
            "year": str(paper.get("year") or ""),
            "journal": (paper.get("venue") or "").strip(),
        })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["file_location", "doi", "title", "authors", "year", "journal"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Manifest written: %d rows → %s", len(rows), manifest_path)
    return len(rows)


# ── result extraction helpers ─────────────────────────────────────────────────

_DOI_RE = re.compile(
    # bibtex: doi = {10.xxx}  OR  doi = "10.xxx"
    r"doi\s*=\s*[\{\"]([^\"}\s,]+)[\}\"]"
    # plain text: doi:10.xxx  or  doi: 10.xxx
    r"|doi:\s*([^\s,\]]+)"
    # URL: doi.org/10.xxx
    r"|doi\.org/([^\s,\"'<>\]]+)",
    re.IGNORECASE,
)

# OpenAlex stores DOIs as full URLs (https://doi.org/10.xxx); strip the prefix
# so they line up with the bare form _extract_doi returns from PaperQA2 bibtex.
_DOI_URL_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


def _normalize_doi(doi: str) -> str:
    """Return bare lower-case DOI (e.g. '10.xxx/yyy'), stripping any URL prefix."""
    if not doi:
        return ""
    return _DOI_URL_PREFIX_RE.sub("", doi.strip()).lower()


def _extract_doi(citation: str) -> str:
    """Extract DOI from citation strings in bibtex, plain 'doi:' or doi.org URL format."""
    m = _DOI_RE.search(citation or "")
    if not m:
        return ""
    # Return whichever capture group matched, strip trailing punctuation
    raw = next(g for g in m.groups() if g is not None)
    return raw.strip(" ,.]").lower()


def compute_paper_recall(
    evidence_passages: list[dict[str, Any]],
    expected_paper_ids: list[str],
    doi_to_paper_id: dict[str, str],
) -> float | None:
    """Paper-level recall: fraction of expected_paper_ids cited by PaperQA2.

    Matching strategy (in order):
      1. DOI match  — most reliable; many papers have DOIs in the bibtex.
      2. doc_name prefix match — for papers where PaperQA2 generates a citation
         key that starts with 'sanitized_paper_id' (fallback).

    Returns None when expected_paper_ids is empty.
    """
    if not expected_paper_ids:
        return None

    cited_dois = {ep["doi"] for ep in evidence_passages if ep.get("doi")}
    cited_doc_names = {ep["doc_name"] for ep in evidence_passages if ep.get("doc_name")}

    hits = 0
    for pid in expected_paper_ids:
        paper_doi = doi_to_paper_id.get(pid, "")  # inverted: doi → pid; here we want pid → doi
        # doi_to_paper_id maps doi → paper_id; invert temporarily per query
        found = False
        # Check DOI match: find doi for this paper_id
        for doi, mapped_pid in doi_to_paper_id.items():
            if mapped_pid == pid and doi and doi in cited_dois:
                found = True
                break
        if not found:
            # Fallback: check if sanitized paper_id appears in any doc_name
            safe_pid = sanitize_paper_id(pid).lower()
            if any(safe_pid in dn.lower() or dn.lower() in safe_pid for dn in cited_doc_names):
                found = True
        if found:
            hits += 1

    return round(hits / len(expected_paper_ids), 4)


# ── result extraction ────────────────────────────────────────────────────────

def extract_result(
    response: Any,
    query_record: dict[str, Any],
    wall_latency_s: float,
    doi_to_paper_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a PaperQA2 AnswerResponse into our result schema.

    AnswerResponse layout (paperqa 2026.x):
        response.session          → PQASession
        response.session.answer   → str (answer text)
        response.session.contexts → list[Context]
        response.duration         → float (seconds)

    Each Context:
        ctx.context               → LLM-generated passage summary (used to answer)
        ctx.text.text             → raw chunk text (may be empty after indexing)
        ctx.text.doc.docname      → LLM-inferred citation key (e.g. 'huang2014small')
        ctx.text.doc.citation     → bibtex string (contains DOI)
        ctx.score                 → relevance score 0–10

    We use ctx.context (not ctx.text.text) as the context passage because:
      - ctx.text.text may be empty (PaperQA2 drops raw text from the index)
      - ctx.context is the summarised passage the LLM actually used to answer
      - This makes RAGAS faithfulness/context_recall scores meaningful
    """
    # Unwrap AnswerResponse → PQASession
    session: Any = getattr(response, "session", response)

    # Prefer formatted_answer (includes in-text citations); fall back to answer
    answer_text: str = ""
    for attr in ("formatted_answer", "answer", "raw_answer"):
        val = getattr(session, attr, None)
        if val and isinstance(val, str) and len(val.strip()) > 5:
            answer_text = val.strip()
            break

    # Abstention: PaperQA2's AgentStatus (success/truncated/unsure = answered;
    # fail = failed; empty = unknown).  Also fall back to phrase matching.
    # AgentStatus values: "success", "truncated", "unsure", "fail"
    pqa_status = str(getattr(response, "status", "") or "").lower()
    _PQA_ANSWERED_STATUSES = {"success", "truncated", "unsure"}
    _ABSTAIN_PHRASES = (
        "cannot answer", "i don't know", "insufficient information",
        "not enough information", "unable to answer", "no information",
        "cannot find", "not addressed", "i cannot", "no relevant",
        "the question cannot", "i was unable",
    )
    status_abstained = bool(pqa_status) and pqa_status not in _PQA_ANSWERED_STATUSES
    phrase_abstained = bool(answer_text) and any(p in answer_text.lower() for p in _ABSTAIN_PHRASES)
    abstained = not answer_text or status_abstained or phrase_abstained

    # Extract citations and evidence passages from contexts
    citations: list[str] = []
    evidence_passages: list[dict[str, Any]] = []

    for ctx in (getattr(session, "contexts", None) or []):
        try:
            doc_name: str = ""
            doi: str = ""
            score: float = float(getattr(ctx, "score", 0) or 0)

            # ctx.context = LLM-summarised passage used to generate the answer
            passage_text = (getattr(ctx, "context", "") or "").strip()

            text_obj = getattr(ctx, "text", None)
            if text_obj is not None:
                doc_obj = getattr(text_obj, "doc", None)
                if doc_obj is not None:
                    doc_name = getattr(doc_obj, "docname", "") or ""
                    doi = _extract_doi(getattr(doc_obj, "citation", "") or "")

            if doc_name:
                citations.append(doc_name)
            evidence_passages.append({
                "doc_name": doc_name,
                "passage": passage_text[:600],
                "score": round(score, 4),
                "doi": doi,
            })
        except Exception:
            pass

    # Paper-level recall (fraction of expected_paper_ids that PaperQA2 cited)
    paper_recall: float | None = None
    if doi_to_paper_id is not None:
        paper_recall = compute_paper_recall(
            evidence_passages,
            query_record.get("expected_paper_ids") or [],
            doi_to_paper_id,
        )

    # PaperQA2 measures its own duration; use it if available, else wall clock
    pqa_duration = float(getattr(response, "duration", wall_latency_s) or wall_latency_s)

    return {
        "query_id": query_record.get("query_id", ""),
        "query": query_record["query"],
        "reference": query_record.get("reference", ""),
        "answer": answer_text,
        "abstained": abstained,
        "citations": list(dict.fromkeys(citations)),   # deduplicated, order-preserving
        "n_citations": len(set(citations)),
        "evidence_passages": evidence_passages,
        "paper_recall": paper_recall,
        "latency_s": round(min(wall_latency_s, pqa_duration * 1.05), 2),
        "pqa_duration_s": round(pqa_duration, 2),
        "domain": query_record.get("domain", ""),
        "query_type": query_record.get("query_type", ""),
    }


# ── main query loop ──────────────────────────────────────────────────────────

def build_doi_to_paper_id(papers: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Build a doi (bare, lower-case) → paper_id reverse-lookup from papers.jsonl."""
    mapping: dict[str, str] = {}
    for pid, paper in papers.items():
        doi = _normalize_doi(paper.get("doi") or "")
        if doi:
            mapping[doi] = pid
    return mapping


def run_queries(
    queries: list[dict[str, Any]],
    paper_texts_dir: Path,
    pqa_index_dir: Path,
    model: str,
    papers: dict[str, dict[str, Any]] | None = None,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run each query through PaperQA2 and return per-query result dicts.

    Args:
        queries:         List of query records from queries.jsonl.
        paper_texts_dir: Directory containing .txt paper files.
        pqa_index_dir:   Directory where PaperQA2 stores its vector index.
        model:           OpenAI model name (e.g. "gpt-5.4-mini").
        papers:          Optional papers dict (paper_id → record) for paper-recall.
        manifest_path:   Optional manifest CSV (file_location, doi, title) so
                         PaperQA2 skips Semantic Scholar metadata lookups.

    Returns:
        List of result dicts, one per query.
    """
    import os as _os

    import asyncio

    try:
        import paperqa.clients  # type: ignore[import]
        import paperqa.docs  # type: ignore[import]
        from paperqa import Settings, ask  # type: ignore[import]
        from paperqa.settings import AgentSettings, IndexSettings  # type: ignore[import]
        from paperqa.agents.search import get_directory_index  # type: ignore[import]
        from paperqa.clients.client_models import MetadataProvider  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "paper-qa is not installed.\n"
            "Install it with:  pip install paper-qa"
        ) from exc

    # Disable all external metadata providers (Crossref, Semantic Scholar, etc).
    # PaperQA2's defaults make a network call per paper during indexing to enrich
    # title/year/doi from S2 — even when a manifest is provided.  On 22k papers
    # without SEMANTIC_SCHOLAR_API_KEY the public endpoint rate-limits at 429 and
    # crashes the build.  We supply all metadata via the manifest CSV, so a NoOp
    # provider is the safe default for a controlled benchmark run.
    #
    # paperqa.docs binds DEFAULT_CLIENTS by name at import time, so patching the
    # original module alone is not enough — we must rebind paperqa.docs too.
    class _NoOpProvider(MetadataProvider):  # type: ignore[misc]
        async def _query(self, query):  # noqa: D401
            return None
        def query_factory(self, query):
            return query
    paperqa.clients.DEFAULT_CLIENTS = (_NoOpProvider,)
    paperqa.docs.DEFAULT_CLIENTS = (_NoOpProvider,)
    logger.info("Metadata providers disabled (manifest is sole source of paper metadata).")


    # Read the key that sota_openai_env() already placed in the environment and
    # inject it directly into every LiteLLM config dict.  We also pin api_base
    # to the canonical OpenAI URL so LiteLLM never falls back to OPENAI_BASE_URL
    # (which would route to the HPC vLLM server and cause a 401).
    api_key = _os.environ.get("OPENAI_API_KEY", "")
    _OPENAI_BASE = "https://api.openai.com/v1"
    _llm_cfg = {
        "model_list": [{
            "model_name": model,
            "litellm_params": {
                "model": model,
                "api_key": api_key,
                "api_base": _OPENAI_BASE,
            },
        }]
    }
    # Embedding model uses a flat kwargs dict, not a model_list router.
    _emb_cfg: dict[str, Any] = {
        "kwargs": {"api_key": api_key, "api_base": _OPENAI_BASE},
    }

    settings = Settings(
        llm=model,
        llm_config=_llm_cfg,
        summary_llm=model,
        summary_llm_config=_llm_cfg,
        embedding_config=_emb_cfg,
        agent=AgentSettings(
            agent_llm=model,
            agent_llm_config=_llm_cfg,
            index=IndexSettings(
                paper_directory=str(paper_texts_dir),
                index_directory=str(pqa_index_dir),
                manifest_file=str(manifest_path) if manifest_path else None,
                # On 22k papers the default concurrency=5 races on docs/<hash>.zip
                # writes → FileNotFoundError mid-indexing.  Serial writes plus
                # batched commits avoid the race (slower but deterministic).
                concurrency=1,
                batch_size=50,
            ),
        ),
        verbosity=0,
    )

    # Pre-build the Tantivy/embedding index before any queries run.  Without
    # this, ask() launches queries concurrently with first-time indexing on
    # large corpora and races against files.zip being written → zlib decode
    # errors and every query aborts in ~2s.
    logger.info("Pre-building PaperQA2 index from %s (this may take 20-30 min for 22k papers)...", paper_texts_dir)
    _t_idx = time.time()
    asyncio.run(get_directory_index(settings, build=True))
    logger.info("Index build complete in %.0fs", time.time() - _t_idx)

    doi_to_pid: dict[str, str] = build_doi_to_paper_id(papers) if papers else {}

    results: list[dict[str, Any]] = []
    n = len(queries)

    for i, q in enumerate(queries, 1):
        qid = q.get("query_id", f"q{i:03d}")
        logger.info("[%d/%d] %s — %s...", i, n, qid, q["query"][:70])

        t0 = time.time()
        try:
            response = ask(q["query"], settings=settings)
            wall = time.time() - t0
            row = extract_result(response, q, wall, doi_to_paper_id=doi_to_pid or None)
            logger.info(
                "  abstained=%-5s  n_citations=%d  paper_recall=%s  latency=%.1fs",
                row["abstained"], row["n_citations"],
                row.get("paper_recall"), row["latency_s"],
            )
        except Exception as exc:
            wall = time.time() - t0
            logger.error("  ERROR after %.1fs: %s", wall, exc)
            row = {
                "query_id": qid,
                "query": q["query"],
                "reference": q.get("reference", ""),
                "answer": "",
                "abstained": True,
                "citations": [],
                "n_citations": 0,
                "evidence_passages": [],
                "paper_recall": None,
                "latency_s": round(wall, 2),
                "pqa_duration_s": round(wall, 2),
                "domain": q.get("domain", ""),
                "query_type": q.get("query_type", ""),
                "error": str(exc),
            }
        results.append(row)

    return results


# ── summary builder ──────────────────────────────────────────────────────────

def run_ragas_on_pqa(
    results: list[dict[str, Any]],
    ragas_model: str,
) -> dict[str, float]:
    """Run RAGAS evaluation on PaperQA2 results.

    Uses ctx.context passages (stored in evidence_passages[*].passage) as
    the 'contexts' for RAGAS — these are the LLM-generated summaries of
    retrieved passages that PaperQA2 used when generating its answer.

    Requires OPENROUTER_API_KEY (for 'openai/gpt-5.4-mini' etc.)
    or GOOGLE_API_KEY (for 'gemini-2.5-flash').

    Returns a dict with ragas_faithfulness, ragas_answer_relevancy,
    ragas_context_recall keys.
    """
    import warnings
    try:
        from datasets import Dataset
        from ragas import evaluate, RunConfig
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_huggingface import HuggingFaceEmbeddings
        from utils.eval_llm import make_eval_llm
    except ImportError as exc:
        logger.error("RAGAS dependencies not available: %s", exc)
        return {}

    rows = []
    for r in results:
        answer = r.get("answer") or ""
        if not answer or r.get("abstained"):
            continue
        contexts = [
            ep["passage"] for ep in (r.get("evidence_passages") or [])
            if ep.get("passage")
        ]
        if not contexts:
            contexts = [""]
        rows.append({
            "question": r["query"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": r.get("reference", ""),
        })

    if not rows:
        logger.warning("No non-abstained results for RAGAS evaluation.")
        return {}

    logger.info("Running RAGAS on %d PaperQA2 results with %s ...", len(rows), ragas_model)
    dataset = Dataset.from_list(rows)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        llm = LangchainLLMWrapper(make_eval_llm(ragas_model))
        emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        faithfulness.llm = llm
        context_recall.llm = llm
        answer_relevancy.llm = llm
        answer_relevancy.embeddings = emb

    run_config = RunConfig(timeout=180, max_retries=3, max_wait=90, max_workers=2)
    score = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm, embeddings=emb,
        run_config=run_config,
        raise_exceptions=False,
    )
    df = score.to_pandas()
    out = {
        "ragas_faithfulness": round(float(df["faithfulness"].mean(skipna=True)), 4),
        "ragas_answer_relevancy": round(float(df["answer_relevancy"].mean(skipna=True)), 4),
        "ragas_context_recall": round(float(df["context_recall"].mean(skipna=True)), 4),
    }
    logger.info("RAGAS scores: %s", out)
    return out


def build_summary(
    results: list[dict[str, Any]],
    model: str,
    ragas_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate per-query results into a top-level summary report."""

    def _avg(key: str) -> float | None:
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    _EXPECTED_ABSTAIN = {"unanswerable", "out_of_domain"}
    answerable = [r for r in results if r.get("query_type") not in _EXPECTED_ABSTAIN]
    expect_abs = [r for r in results if r.get("query_type") in _EXPECTED_ABSTAIN]

    def _abstention_rate(subset: list[dict[str, Any]]) -> float | None:
        if not subset:
            return None
        return round(sum(1 for r in subset if r.get("abstained")) / len(subset), 4)

    report: dict[str, Any] = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "n_queries": len(results),
        "avg_latency_s": _avg("latency_s"),
        "avg_n_citations": _avg("n_citations"),
        "avg_paper_recall": _avg("paper_recall"),
        "abstention_rate_answerable": _abstention_rate(answerable),
        "correct_abstention_rate": _abstention_rate(expect_abs),
    }
    if ragas_scores:
        report.update(ragas_scores)
    report["per_query"] = results
    return report


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the PaperQA2 baseline runner."""
    parser = argparse.ArgumentParser(
        description="Run PaperQA2 SOTA baseline against our benchmark queries."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only the first 2 queries using only their referenced papers "
             "(fast sanity check before a full HPC run).",
    )
    parser.add_argument(
        "--queries-path", default=str(DEFAULT_QUERIES_PATH),
        help="Path to eval/queries.jsonl  (default: %(default)s)",
    )
    parser.add_argument(
        "--papers-path", default=str(DEFAULT_PAPERS_PATH),
        help="Path to data/raw/papers.jsonl  (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Output path for results JSON  (default: %(default)s)",
    )
    parser.add_argument(
        "--model", default="gpt-5.4-mini",
        help="OpenAI model for PaperQA2  (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-env-override", action="store_true",
        help="Do NOT swap OPENAI_API_KEY/OPENAI_BASE_URL.  Use this when the "
             "current environment already points at real OpenAI.",
    )
    parser.add_argument(
        "--run-ragas", action="store_true",
        help="After running PaperQA2, evaluate with RAGAS (faithfulness, "
             "answer_relevancy, context_recall).  Requires OPENROUTER_API_KEY "
             "or GOOGLE_API_KEY for the eval LLM.",
    )
    parser.add_argument(
        "--ragas-model", default="gemini-2.5-flash",
        help="Model for RAGAS evaluation  (default: %(default)s).  "
             "Use 'gemini-2.5-flash' with GOOGLE_API_KEY or an OpenRouter "
             "model slug with OPENROUTER_API_KEY.",
    )
    args = parser.parse_args()

    mode_label = "DRY RUN (2 queries)" if args.dry_run else f"FULL RUN ({args.model})"
    logger.info("=== PaperQA2 baseline — %s ===", mode_label)

    # ── load corpus and queries ─────────────────────────────────────────────
    papers = load_papers(Path(args.papers_path))
    queries = load_queries(
        Path(args.queries_path),
        max_queries=2 if args.dry_run else None,
    )

    # ── decide which papers to export ──────────────────────────────────────
    # For --dry-run: only the papers explicitly expected by those 2 queries.
    # For full run:  all papers in papers.jsonl.
    if args.dry_run:
        dry_ids: set[str] = set()
        for q in queries:
            dry_ids.update(q.get("expected_paper_ids") or [])
        paper_ids_to_write: set[str] | None = dry_ids or None
        logger.info("Dry-run: exporting %d reference papers", len(dry_ids))
    else:
        paper_ids_to_write = None  # export everything

    # Use separate sub-directories so dry-run index never contaminates full run
    subdir = "dry_run" if args.dry_run else "full"
    paper_texts_dir = PAPER_TEXTS_ROOT / subdir
    pqa_index_dir = PQA_INDEX_ROOT / subdir

    # ── write paper text files ──────────────────────────────────────────────
    n_files = write_paper_texts(papers, paper_texts_dir, paper_ids=paper_ids_to_write)
    if n_files == 0:
        logger.error("No paper text files were written — aborting.")
        return

    # ── write manifest so PaperQA2 skips Semantic Scholar lookups ───────────
    manifest_path = paper_texts_dir / "manifest.csv"
    write_manifest(papers, paper_texts_dir, manifest_path, paper_ids=paper_ids_to_write)

    # ── set up env override context ─────────────────────────────────────────
    if args.skip_env_override:
        env_ctx: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
    else:
        from paperqa_baseline.env_utils import sota_openai_env
        env_ctx = sota_openai_env()

    # ── run PaperQA2 ────────────────────────────────────────────────────────
    with env_ctx:
        results = run_queries(
            queries, paper_texts_dir, pqa_index_dir,
            model=args.model, papers=papers,
            manifest_path=manifest_path,
        )

    # ── optional RAGAS evaluation ────────────────────────────────────────────
    ragas_scores: dict[str, float] | None = None
    if args.run_ragas:
        ragas_scores = run_ragas_on_pqa(results, ragas_model=args.ragas_model) or None

    # ── save results ────────────────────────────────────────────────────────
    report = build_summary(results, model=args.model, ragas_scores=ragas_scores)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    logger.info("Results saved to %s", output_path)
    logger.info(
        "Summary: n=%d  avg_latency=%.1fs  abstention_answerable=%s  correct_abstention=%s",
        report["n_queries"],
        report["avg_latency_s"] or 0,
        report["abstention_rate_answerable"],
        report["correct_abstention_rate"],
    )


if __name__ == "__main__":
    main()
