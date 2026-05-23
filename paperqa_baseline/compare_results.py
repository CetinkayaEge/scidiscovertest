"""Compare SciDiscover pipeline results against PaperQA2 baseline.

Loads both result files, computes shared metrics, and prints + saves a
Markdown comparison table.

Usage:
    # Use default paths (reports/eval_results.json vs paperqa_baseline/results/paperqa_results.json)
    PYTHONPATH=. python paperqa_baseline/compare_results.py

    # Override the SciDiscover results path
    PYTHONPATH=. python paperqa_baseline/compare_results.py \\
        --our-results reports/eval_full.json

    # Compare against a specific PaperQA2 run
    PYTHONPATH=. python paperqa_baseline/compare_results.py \\
        --our-results reports/eval_full.json \\
        --pqa-results paperqa_baseline/results/paperqa_results.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure stdout handles Unicode on Windows (cp1252 default breaks ↑↓ arrows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
REPO_ROOT = _HERE.parent

DEFAULT_OUR_RESULTS = REPO_ROOT / "reports" / "eval_results.json"
DEFAULT_PQA_RESULTS = _HERE / "results" / "paperqa_results.json"
DEFAULT_OUTPUT = _HERE / "results" / "comparison_table.md"


# ── ROUGE-L ──────────────────────────────────────────────────────────────────

def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """Compute token-level ROUGE-L F1 between hypothesis and reference.

    Uses the standard longest-common-subsequence formulation:
        P = LCS / |hypothesis|,  R = LCS / |reference|,  F1 = 2PR / (P+R)
    """
    h = hypothesis.lower().split()
    r = reference.lower().split()
    if not h or not r:
        return 0.0

    m, n = len(r), len(h)
    # dp[i][j] = LCS length for r[:i] and h[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0.0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 4)


# ── helpers ──────────────────────────────────────────────────────────────────

def _avg(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def _fmt(val: Any, decimals: int = 3) -> str:
    """Format a metric value for the Markdown table."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _abstention_rate(rows: list[dict[str, Any]], key: str = "abstained") -> float | None:
    if not rows:
        return None
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 4)


def _split_answerable(
    per_q: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (answerable, expected-to-abstain) groups."""
    _SKIP = {"unanswerable", "out_of_domain"}
    return (
        [r for r in per_q if r.get("query_type") not in _SKIP],
        [r for r in per_q if r.get("query_type") in _SKIP],
    )


def compute_rouge_l_mean(
    per_q: list[dict[str, Any]],
    answer_key: str,
) -> float | None:
    """Mean ROUGE-L across queries that have a non-empty reference."""
    scores = [
        rouge_l_f1(r.get(answer_key, ""), r.get("reference", ""))
        for r in per_q
        if r.get("reference") and r.get(answer_key)
    ]
    return _avg(scores)


# ── comparison table builder ─────────────────────────────────────────────────

def build_comparison_table(
    our_data: dict[str, Any] | None,
    pqa_data: dict[str, Any],
    our_path: Path,
    pqa_path: Path,
) -> str:
    """Return the full Markdown comparison table as a string."""

    our_pq: list[dict[str, Any]] = (our_data or {}).get("per_query", [])
    pqa_pq: list[dict[str, Any]] = pqa_data.get("per_query", [])

    our_ans, our_exp = _split_answerable(our_pq)
    pqa_ans, pqa_exp = _split_answerable(pqa_pq)

    # ── metric rows: (label, our_value, pqa_value, notes) ──────────────────
    rows: list[tuple[str, Any, Any, str]] = []

    # 1. Abstention rates
    rows.append((
        "Abstention rate — answerable queries (↓ better)",
        _abstention_rate(our_ans) if our_pq else (our_data or {}).get("abstention_rate_answerable"),
        _abstention_rate(pqa_ans),
        "Fraction of answerable queries where the system refused to answer",
    ))
    rows.append((
        "Correct abstention rate — unanswerable queries (↑ better)",
        _abstention_rate(our_exp) if our_pq else (our_data or {}).get("correct_abstention_rate"),
        _abstention_rate(pqa_exp),
        "Fraction of unanswerable/OOD queries where the system correctly declined",
    ))

    # 2. Claims / citations
    our_n_claims = _avg([r.get("n_claims") for r in our_pq]) if our_pq else None
    pqa_n_cit = pqa_data.get("avg_n_citations") or _avg([r.get("n_citations") for r in pqa_pq])
    rows.append((
        "Avg claims / citations per query",
        our_n_claims,
        pqa_n_cit,
        "SciDiscover: verified key_claims; PaperQA2: retrieved context passages",
    ))

    # 3. Latency
    our_lat = (our_data or {}).get("avg_latency_s") or _avg([r.get("latency_s") for r in our_pq])
    pqa_lat = pqa_data.get("avg_latency_s") or _avg([r.get("latency_s") for r in pqa_pq])
    rows.append((
        "Avg latency (s) per query (↓ better)",
        our_lat,
        pqa_lat,
        "Mean wall-clock seconds from query submission to final answer",
    ))

    # 4. ROUGE-L vs ground-truth reference
    our_rouge = compute_rouge_l_mean(our_pq, "final_answer") if our_pq else None
    pqa_rouge = compute_rouge_l_mean(pqa_pq, "answer")
    rows.append((
        "ROUGE-L vs reference (↑ better)",
        our_rouge,
        pqa_rouge,
        "Token-level F1 LCS overlap with ground-truth answer from eval/queries.jsonl",
    ))

    # 5. RAGAS — both systems if available
    def _pqa(key: str) -> Any:
        return pqa_data.get(key) or _avg([r.get(key) for r in pqa_pq])

    def _our(key: str) -> Any:
        if not our_data:
            return None
        return our_data.get(key) or _avg([r.get(key) for r in our_pq])

    ragas_keys = [
        ("ragas_faithfulness", "RAGAS Faithfulness (↑ better)",
         "Fraction of answer claims supported by retrieved context (NLI-based)"),
        ("ragas_answer_relevancy", "RAGAS Answer Relevancy (↑ better)",
         "Cosine similarity of answer embeddings to the question"),
        ("ragas_context_recall", "RAGAS Context Recall (↑ better)",
         "Fraction of ground-truth statements attributable to the retrieved context"),
    ]
    for rk, label, note in ragas_keys:
        our_v = _our(rk)
        pqa_v = _pqa(rk)
        if our_v is not None or pqa_v is not None:
            rows.append((label, our_v, pqa_v, note))

    # 6. Paper-level recall — comparable cross-system metric
    # SciDiscover: avg_recall_at_k (fraction of ground-truth chunk IDs in top-k FAISS)
    # PaperQA2:    avg_paper_recall (fraction of expected paper IDs cited in answer)
    # Both measure "did the system find the relevant source?" but at different granularities.
    our_recall = _our("avg_recall_at_k")
    pqa_recall = _pqa("avg_paper_recall")
    rows.append((
        "Paper/chunk recall (↑ better)",
        our_recall,
        pqa_recall,
        "SciDiscover: chunk-level recall@k in FAISS; PaperQA2: paper-level recall vs expected_paper_ids",
    ))

    # 7. SciDiscover-only pipeline metrics
    rows.append((
        "Citation coverage (↑ better)",
        _our("avg_citation_coverage"),
        None,
        "Fraction of key_claims citing ≥1 chunk [SciDiscover only]",
    ))
    rows.append((
        "Claim support rate (↑ better)",
        _our("avg_support_rate"),
        None,
        "Fraction of cited claims verified SUPPORTED by VerifierAgent NLI [SciDiscover only]",
    ))
    rows.append((
        "Synthesizer hallucination rate (↓ better)",
        _our("avg_synthesizer_hallucination_rate"),
        None,
        "Fraction of cited chunk_ids absent from retrieved evidence set [SciDiscover only]",
    ))

    # ── format Markdown ─────────────────────────────────────────────────────
    pqa_model = pqa_data.get("model", "unknown")
    our_mode = (our_data or {}).get("mode", "unknown") if our_data else "not available"
    pqa_ts = pqa_data.get("run_timestamp", "unknown")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        "# SOTA Baseline Comparison: SciDiscover vs PaperQA2",
        "",
        f"**Generated:** {now_str}",
        f"**SciDiscover mode:** `{our_mode}`  —  results from `{our_path}`",
        f"**PaperQA2 model:** `{pqa_model}`  —  run timestamp `{pqa_ts}`  —  results from `{pqa_path}`",
        "",
        "> **Note on LLM fairness:** SciDiscover runs on a local HPC vLLM server (Qwen2.5-32B).",
        "> PaperQA2 uses the real OpenAI API (gpt-4o-mini by default).  Direct numeric comparisons",
        "> are therefore indicative only; for a fully controlled comparison both systems should use",
        "> the same LLM backend.",
        "",
        "## Metric Comparison",
        "",
        "| Metric | SciDiscover | PaperQA2 | Notes |",
        "| ------ | :---------: | :------: | ----- |",
    ]

    for label, our_v, pqa_v, note in rows:
        our_str = _fmt(our_v) if our_data else "*(no results)*"
        pqa_str = _fmt(pqa_v)
        lines.append(f"| {label} | {our_str} | {pqa_str} | {note} |")

    # Flag whether PaperQA2 has RAGAS scores
    pqa_has_ragas = any(pqa_data.get(f"ragas_{k}") is not None for k in ("faithfulness", "answer_relevancy", "context_recall"))
    pqa_has_paper_recall = pqa_data.get("avg_paper_recall") is not None

    ragas_note = (
        "RAGAS computed for both systems (run with `--run-ragas`)."
        if pqa_has_ragas
        else "RAGAS **not yet computed** for PaperQA2.  Re-run with `--run-ragas` flag."
    )
    paper_recall_note = (
        "Paper-level recall computed for PaperQA2."
        if pqa_has_paper_recall
        else "Paper-level recall **not yet computed** for PaperQA2 (requires re-run with updated code)."
    )

    lines += [
        "",
        "## Metric Availability Notes",
        "",
        f"- **RAGAS**: {ragas_note}",
        f"- **Paper/chunk recall**: {paper_recall_note}",
        "- **Citation coverage**: PaperQA2 does not expose per-claim citation-coverage in a structured field.",
        "- **Claim support rate**: No equivalent of SciDiscover's VerifierAgent NLI checking in PaperQA2.",
        "- **Synthesizer hallucination rate**: Internal metric tied to SciDiscover's chunk-ID citation format.",
        "",
        "## Query-Level Statistics",
        "",
        f"| | SciDiscover | PaperQA2 |",
        f"| - | - | - |",
        f"| Total queries evaluated | {len(our_pq) if our_pq else '—'} | {len(pqa_pq)} |",
        f"| Answerable queries | {len(our_ans) if our_pq else '—'} | {len(pqa_ans)} |",
        f"| Unanswerable / OOD queries | {len(our_exp) if our_pq else '—'} | {len(pqa_exp)} |",
        "",
        "## How to Read This Table",
        "",
        "- **Abstention rate (answerable)**: lower is better — a high rate means the system often",
        "  refused to answer questions it should have been able to answer.",
        "- **Correct abstention rate**: higher is better — the system should decline unanswerable",
        "  and out-of-domain queries.",
        "- **ROUGE-L**: higher is better; computed only for queries with a non-empty reference.",
        "- **Paper/chunk recall**: SciDiscover uses chunk-level recall@k in its FAISS index;",
        "  PaperQA2 uses paper-level recall vs expected_paper_ids from queries.jsonl.",
        "  Both measure 'did the system retrieve the relevant source?' at different granularities.",
        "- **—** in a column: metric is not available for that system.",
    ]

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the comparison evaluator."""
    parser = argparse.ArgumentParser(
        description="Compare SciDiscover pipeline results vs PaperQA2 baseline."
    )
    parser.add_argument(
        "--our-results", default=str(DEFAULT_OUR_RESULTS),
        help="Path to SciDiscover eval results JSON  (default: %(default)s)",
    )
    parser.add_argument(
        "--pqa-results", default=str(DEFAULT_PQA_RESULTS),
        help="Path to PaperQA2 results JSON  (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Output path for comparison_table.md  (default: %(default)s)",
    )
    args = parser.parse_args()

    our_path = Path(args.our_results)
    pqa_path = Path(args.pqa_results)
    output_path = Path(args.output)

    # ── load ────────────────────────────────────────────────────────────────
    our_data: dict[str, Any] | None = None
    if our_path.exists():
        with open(our_path, "r", encoding="utf-8") as fh:
            our_data = json.load(fh)
        print(f"Loaded SciDiscover results from {our_path}")
    else:
        print(f"WARNING: SciDiscover results not found at {our_path}")
        print("         SciDiscover columns will show 'no results'.")
        print("         Run the eval pipeline first:  "
              "PYTHONPATH=. python -m scidiscover.eval.run --config configs/demo.yaml")

    if not pqa_path.exists():
        print(f"ERROR: PaperQA2 results not found at {pqa_path}")
        print("       Run the baseline first:  "
              "PYTHONPATH=. python paperqa_baseline/run_paperqa.py")
        return

    with open(pqa_path, "r", encoding="utf-8") as fh:
        pqa_data: dict[str, Any] = json.load(fh)
    print(f"Loaded PaperQA2 results from {pqa_path}")

    # ── build and save table ─────────────────────────────────────────────────
    table = build_comparison_table(our_data, pqa_data, our_path, pqa_path)

    print("\n" + "=" * 70)
    print(table)
    print("=" * 70 + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")
    print(f"Comparison table saved to {output_path}")


if __name__ == "__main__":
    main()
