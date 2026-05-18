"""
Run evaluation pipeline over data/eval/queries.jsonl and report metrics.

Usage:
    PYTHONPATH=. python -m scidiscover.eval.run --config configs/demo.yaml
    PYTHONPATH=. python -m scidiscover.eval.run --config configs/demo.yaml --skip-ragas
    PYTHONPATH=. python -m scidiscover.eval.run --config configs/demo.yaml --max-queries 5
    PYTHONPATH=. python -m scidiscover.eval.run --config configs/demo.yaml --skip-verifier --output reports/eval_no_verifier_hpc.json

Ablation study run commands:
    # 1. Full pipeline — HPC/deepseek (reranker.enabled: true in config)
    python -m scidiscover.eval.run --config configs/demo.yaml --output reports/eval_full_hpc.json

    # 2. Full pipeline — Gemini (change llm.model to gemini-2.5-flash in config)
    python -m scidiscover.eval.run --config configs/demo.yaml --output reports/eval_full_gemini.json

    # 3. No verifier — HPC/deepseek (reranker.enabled: true)
    python -m scidiscover.eval.run --config configs/demo.yaml --skip-verifier --output reports/eval_no_verifier_hpc.json

    # 4. No verifier, no reranker — HPC/deepseek (set reranker.enabled: false in config)
    python -m scidiscover.eval.run --config configs/demo.yaml --skip-verifier --output reports/eval_no_verifier_no_reranker_hpc.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_ABSTAIN_MSG = "Insufficient evidence in retrieved corpus to answer reliably."


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_recall_at_k(evidence_pack: dict, expected_chunk_ids: list) -> float | None:
    """Fraction of expected chunk_ids found in the retrieved chunks."""
    if not expected_chunk_ids:
        return None
    retrieved = {ch["chunk_id"] for ch in evidence_pack["chunks"]}
    hits = sum(1 for cid in expected_chunk_ids if cid in retrieved)
    return round(hits / len(expected_chunk_ids), 4)


def compute_summarizer_hallucination_rate(summaries: list, evidence_pack: dict) -> float | None:
    """
    Fraction of [chunk_id] citations in summarizer output that do NOT belong to
    that paper's own retrieved chunks.

    Catches two failure modes:
      - Completely fabricated chunk IDs (not in evidence_pack at all)
      - Cross-paper citations (chunk from paper Y cited in paper X's summary)

    Returns None when no citations were produced, to distinguish from a clean
    run where citations were present and all were valid.
    """
    # Per-paper chunk sets so cross-paper citations are flagged
    paper_chunks: dict = {}
    for ch in evidence_pack["chunks"]:
        paper_chunks.setdefault(ch["paper_id"], set()).add(ch["chunk_id"])

    pattern = re.compile(r'\[([^\]]+\|\|[^\]]+)\]')
    total = hallucinated = 0
    for ps in summaries:
        own_ids = paper_chunks.get(ps.get("paper_id", ""), set())
        for cid in pattern.findall(ps.get("summary_text", "")):
            total += 1
            if cid not in own_ids:
                hallucinated += 1

    return round(hallucinated / total, 4) if total > 0 else None


def compute_synthesizer_hallucination_rate(synthesis: dict, evidence_pack: dict) -> float | None:
    """
    Fraction of citation_ids in synthesizer key_claims that do NOT exist in the
    retrieved evidence set.

    Catches synthesizer-level hallucinations that bypass the summarizer metric.
    Returns None when no citations were produced.
    """
    valid_ids = {ch["chunk_id"] for ch in evidence_pack["chunks"]}
    total = hallucinated = 0
    for claim in synthesis.get("key_claims", []):
        for cid in claim.get("citation_ids", []):
            total += 1
            if cid not in valid_ids:
                hallucinated += 1

    return round(hallucinated / total, 4) if total > 0 else None


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _citation_coverage_from_claims(key_claims: list) -> float:
    """Compute citation coverage directly from synthesis key_claims (no verifier)."""
    if not key_claims:
        return 0.0
    cited = sum(1 for c in key_claims if c.get("citation_ids"))
    return round(cited / len(key_claims), 4)


def run_pipeline(query_record: dict, agents: dict, top_k: int,
                 skip_verifier: bool = False, retrieval_k: int | None = None) -> dict:
    query = query_record["query"]
    qid = query_record.get("query_id") or query_record.get("id", "")
    rk = retrieval_k if retrieval_k is not None else top_k

    t0 = time.time()

    if "query_decomposer" in agents:
        sub_queries = agents["query_decomposer"].run(query)
        merged: dict = {}
        for sq in sub_queries:
            ep = agents["retriever_agent"].run(sq, k=rk)
            for ch in ep["chunks"]:
                cid = ch["chunk_id"]
                if cid not in merged or ch["score"] > merged[cid]["score"]:
                    merged[cid] = ch
        evidence_pack = {"query": query, "chunks": list(merged.values())}
    elif "hyde_agent" in agents:
        hypothetical = agents["hyde_agent"].run(query)
        evidence_pack = agents["retriever_agent"].run(hypothetical, k=rk)
        evidence_pack["query"] = query
    else:
        evidence_pack = agents["retriever_agent"].run(query, k=rk)

    if "reranker_agent" in agents:
        evidence_pack = agents["reranker_agent"].run(evidence_pack)

    # Single-agent baseline: one LLM call replaces the full multi-agent chain
    if "single_agent" in agents:
        recall_at_k = compute_recall_at_k(evidence_pack, query_record.get("expected_chunk_ids", []))
        retrieved_contexts = [ch["text"] for ch in evidence_pack["chunks"]]
        result = agents["single_agent"].run(evidence_pack)
        latency_s = time.time() - t0
        return {
            "query_id": qid,
            "query": query,
            "reference": query_record.get("ground_truth", query_record.get("reference", "")),
            "retrieved_contexts": retrieved_contexts,
            "final_answer": result["final_answer"],
            "abstained": result["abstained"],
            "citation_coverage": _citation_coverage_from_claims(result["key_claims"]),
            "support_rate": None,
            "n_claims": len(result["key_claims"]),
            "n_supported": None,
            "n_unsupported": None,
            "n_conflict": None,
            "n_unknown": None,
            "recall_at_k": recall_at_k,
            "summarizer_hallucination_rate": None,
            "synthesizer_hallucination_rate": compute_synthesizer_hallucination_rate(
                {"key_claims": result["key_claims"]}, evidence_pack
            ),
            "latency_s": round(latency_s, 2),
            "domain": query_record.get("domain", ""),
            "query_type": query_record.get("query_type", ""),
        }

    summaries = agents["summarizer_agent"].run(evidence_pack)

    synthesis = agents["synthesizer_agent"].run({
        "query": query,
        "paper_summaries": summaries,
    })

    recall_at_k = compute_recall_at_k(evidence_pack, query_record.get("expected_chunk_ids", []))

    # Hallucination rates (both None when model produced no citations)
    summarizer_hal = compute_summarizer_hallucination_rate(summaries, evidence_pack)
    synthesizer_hal = compute_synthesizer_hallucination_rate(synthesis, evidence_pack)

    # Store retrieved texts for RAGAS
    retrieved_contexts = [ch["text"] for ch in evidence_pack["chunks"]]

    if skip_verifier:
        draft = synthesis.get("draft_answer", "")
        key_claims = synthesis.get("key_claims", [])
        abstained = (
            not draft
            or draft.startswith("ABSTAIN")
            or draft.startswith("[")
            or "paper summaries" in draft.lower()   # synthesizer early-return fallback
        )
        final_answer = _ABSTAIN_MSG if abstained else draft
        citation_coverage = _citation_coverage_from_claims(key_claims)
        latency_s = time.time() - t0
        return {
            "query_id": qid,
            "query": query,
            "reference": query_record.get("ground_truth", query_record.get("reference", "")),
            "retrieved_contexts": retrieved_contexts,
            "final_answer": final_answer,
            "abstained": abstained,
            "citation_coverage": citation_coverage,
            "support_rate": None,       # verifier not run
            "n_claims": len(key_claims),
            "n_supported": None,
            "n_unsupported": None,
            "n_conflict": None,
            "n_unknown": None,
            "recall_at_k": recall_at_k,
            "summarizer_hallucination_rate": summarizer_hal,
            "synthesizer_hallucination_rate": synthesizer_hal,
            "latency_s": round(latency_s, 2),
            "domain": query_record.get("domain", ""),
            "query_type": query_record.get("query_type", ""),
        }

    # Full pipeline — run verifier
    verified = agents["verifier_agent"].run({
        "synthesis": synthesis,
        "evidence_pack": evidence_pack,
    })
    latency_s = time.time() - t0
    vs = verified["verification_summary"]

    return {
        "query_id": qid,
        "query": query,
        "reference": query_record.get("ground_truth", query_record.get("reference", "")),
        "retrieved_contexts": retrieved_contexts,
        "final_answer": verified["final_answer"],
        "abstained": verified["final_answer"] == _ABSTAIN_MSG,
        "citation_coverage": round(vs["citation_coverage"], 4),
        "support_rate": round(vs["support_rate"], 4),
        "n_claims": vs["total_claims"],
        "n_supported": vs["supported"],
        "n_unsupported": vs["unsupported"],
        "n_conflict": vs["conflict"],
        "n_unknown": vs["unknown"],
        "recall_at_k": recall_at_k,
        "summarizer_hallucination_rate": summarizer_hal,
        "synthesizer_hallucination_rate": synthesizer_hal,
        "latency_s": round(latency_s, 2),
        "domain": query_record.get("domain", ""),
        "query_type": query_record.get("query_type", ""),
    }


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------

def run_ragas(results: list, ragas_model: str) -> dict:
    import warnings
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from utils.eval_llm import make_eval_llm

    rows = []
    for r in results:
        answer = r.get("final_answer") or ""
        if not answer or answer == _ABSTAIN_MSG:
            continue
        rows.append({
            "question": r["query"],
            "answer": answer,
            "contexts": r.get("retrieved_contexts") or [""],
            "ground_truth": r.get("reference", ""),
        })

    if not rows:
        return {}

    dataset = Dataset.from_list(rows)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        llm = LangchainLLMWrapper(make_eval_llm(ragas_model))
        emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        # Explicitly inject LLM/embeddings into singletons to ensure they are used
        faithfulness.llm = llm
        context_recall.llm = llm
        answer_relevancy.llm = llm
        answer_relevancy.embeddings = emb

    run_config = RunConfig(timeout=180, max_retries=3, max_wait=90, max_workers=2)
    score = evaluate(dataset=dataset,
                     metrics=[faithfulness, answer_relevancy, context_recall],
                     llm=llm, embeddings=emb,
                     run_config=run_config,
                     raise_exceptions=False)
    df = score.to_pandas()
    return {
        "ragas_faithfulness": round(float(df["faithfulness"].mean(skipna=True)), 4),
        "ragas_answer_relevancy": round(float(df["answer_relevancy"].mean(skipna=True)), 4),
        "ragas_context_recall": round(float(df["context_recall"].mean(skipna=True)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--skip-verifier", action="store_true",
                        help="Ablation: skip VerifierAgent, use synthesis output directly")
    parser.add_argument("--skip-reranker", action="store_true",
                        help="Ablation: skip RerankerAgent regardless of config")
    parser.add_argument("--output", default=None,
                        help="Override output path (default: eval.results_output in config)")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--baseline", choices=["single_agent"], default=None,
                        help="Run the single-agent baseline: one LLM call over all retrieved chunks, "
                             "replacing the full Summarizer + Synthesizer + Verifier chain")
    parser.add_argument("--query-transformer", choices=["none", "decompose", "hyde"], default=None,
                        help="Override query_transformer.strategy from config")
    parser.add_argument("--retrieval-k", type=int, default=None,
                        help="Override retriever pool size (default: eval.top_k_recall). "
                             "Use a larger value than top_k_recall when reranker is enabled "
                             "so the reranker has a wider pool to filter from.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_cfg = config["eval"]
    queries_path = eval_cfg["queries_path"]
    results_output = args.output or eval_cfg["results_output"]
    top_k = eval_cfg.get("top_k_recall", 20)
    max_queries = args.max_queries or eval_cfg.get("max_queries")
    ragas_model = eval_cfg.get("ragas_model", "gemini-2.5-flash")

    # Derive a human-readable mode tag for the report
    reranker_on = config.get("reranker", {}).get("enabled", False) and not args.skip_reranker
    llm_model = config["llm"]["model"]
    qt_strategy_for_tag = args.query_transformer or config.get("query_transformer", {}).get("strategy", "none")
    rk_tag = f"rk{args.retrieval_k}" if args.retrieval_k else ""
    if args.baseline:
        reranker_str = "reranker" if reranker_on else "no_reranker"
        mode_tag = f"baseline_{args.baseline}_{reranker_str}_{llm_model.split('-')[0]}"
        if rk_tag:
            mode_tag += f"_{rk_tag}"
    else:
        mode_tag = (
            f"{'no_verifier' if args.skip_verifier else 'full'}"
            f"_{'no_reranker' if not reranker_on else 'reranker'}"
            f"_{qt_strategy_for_tag}"
            f"_{llm_model.split('-')[0]}"
        )
        if rk_tag:
            mode_tag += f"_{rk_tag}"

    # Agents must be imported before utils.llm_client on Windows to avoid
    # a silent crash caused by google.genai and sentence_transformers both
    # trying to register the google namespace package in the same process.
    from scidiscover.agents.retriever import Retriever
    from scidiscover.agents.retriever_agent import RetrieverAgent
    from scidiscover.agents.reranker import RerankerAgent
    from scidiscover.agents.query_decomposer import QueryDecomposerAgent
    from scidiscover.agents.hyde_agent import HyDEAgent

    if not args.baseline:
        from scidiscover.agents.summarizer import SummarizerAgent
        from scidiscover.agents.synthesizer import SynthesizerAgent
        from scidiscover.agents.verifier import VerifierAgent
    else:
        from scidiscover.agents.single_agent_baseline import SingleAgentBaseline

    from utils.llm_client import configure_llm
    configure_llm(config["llm"]["model"], config["llm"]["max_tokens"])

    retriever = Retriever(
        chunks_input=config["retrieval"]["chunks_input"],
        index_output=config["retrieval"]["index_output"],
        chunk_ids_output=config["retrieval"]["chunk_ids_output"],
        traces_output=config["retrieval"]["traces_output"],
        model_name=config["retrieval"]["model_name"],
        papers_input=config["retrieval"]["papers_input"],
    )

    agents = {"retriever_agent": RetrieverAgent(retriever)}

    if not args.baseline:
        verifier_cfg = config["verifier"]
        agents["summarizer_agent"] = SummarizerAgent(
            prompt_path=config["summarizer"]["prompt_path"],
            traces_output=config["summarizer"]["traces_output"],
            output_path=config["summarizer"]["output_path"],
            max_workers=config["summarizer"].get("max_workers", 4),
        )
        agents["synthesizer_agent"] = SynthesizerAgent(
            prompt_path=config["synthesizer"]["prompt_path"],
            traces_output=config["synthesizer"]["traces_output"],
            output_path=config["synthesizer"]["output_path"],
        )
        agents["verifier_agent"] = VerifierAgent(
            prompt_path=verifier_cfg["prompt_path"],
            traces_output=verifier_cfg["traces_output"],
            verification_output=verifier_cfg["verification_output"],
            answers_output=verifier_cfg["answers_output"],
            min_citation_coverage=verifier_cfg.get("min_citation_coverage", 0.95),
            min_support_rate=verifier_cfg.get("min_support_rate", 0.50),
        )
    else:  # single_agent
        baseline_cfg = config.get("baselines", {}).get("single_agent", {})
        agents["single_agent"] = SingleAgentBaseline(
            prompt_path=baseline_cfg.get("prompt_path", "prompts/single_agent_baseline.txt"),
            traces_output=baseline_cfg.get("traces_output", "logs/single_agent_baseline_traces.jsonl"),
        )

    qt_cfg = config.get("query_transformer", {})
    qt_strategy = args.query_transformer or qt_cfg.get("strategy", "none")
    if qt_strategy == "decompose":
        decomp_cfg = qt_cfg.get("decomposer", {})
        agents["query_decomposer"] = QueryDecomposerAgent(
            prompt_path=decomp_cfg.get("prompt_path", "prompts/query_decomposer.txt"),
            max_sub_queries=decomp_cfg.get("max_sub_queries", 3),
            traces_output=decomp_cfg.get("traces_output", "logs/query_decomposer_traces.jsonl"),
        )
    elif qt_strategy == "hyde":
        hyde_cfg = qt_cfg.get("hyde", {})
        agents["hyde_agent"] = HyDEAgent(
            prompt_path=hyde_cfg.get("prompt_path", "prompts/hyde.txt"),
            traces_output=hyde_cfg.get("traces_output", "logs/hyde_traces.jsonl"),
        )

    reranker_cfg = config.get("reranker", {})
    if reranker_cfg.get("enabled", False) and not args.skip_reranker:
        agents["reranker_agent"] = RerankerAgent(
            model_name=reranker_cfg["model_name"],
            top_k=reranker_cfg.get("top_k", top_k),
            min_score=reranker_cfg.get("min_score", 0.0),
        )

    if not Path(queries_path).exists():
        raise FileNotFoundError(
            f"{queries_path} not found. Run generate_testset.py first:\n"
            f"  PYTHONPATH=. python -m scidiscover.eval.generate_testset --config {args.config}"
        )

    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    if max_queries:
        queries = queries[:max_queries]

    print(f"Running eval on {len(queries)} queries (top_k={top_k})...")

    results = []
    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q.get('query_id', '')} — {q['query'][:70]}...")
        try:
            row = run_pipeline(q, agents, top_k,
                               skip_verifier=args.skip_verifier,
                               retrieval_k=args.retrieval_k)
            rec = row["recall_at_k"]
            hal_sum = row["summarizer_hallucination_rate"]
            hal_syn = row["synthesizer_hallucination_rate"]
            sup = row["support_rate"]
            print(f"    cov={row['citation_coverage']:.2f}  "
                  f"support={f'{sup:.2f}' if sup is not None else 'N/A'}  "
                  f"rec@k={rec if rec is not None else 'N/A'}  "
                  f"hal_sum={f'{hal_sum:.2f}' if hal_sum is not None else 'N/A'}  "
                  f"hal_syn={f'{hal_syn:.2f}' if hal_syn is not None else 'N/A'}  "
                  f"abstain={row['abstained']}  "
                  f"{row['latency_s']}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            row = {
                "query_id": q.get("query_id", ""),
                "query": q["query"],
                "reference": q.get("ground_truth", q.get("reference", "")),
                "retrieved_contexts": [],
                "final_answer": "",
                "abstained": True,
                "citation_coverage": 0.0,
                "support_rate": None,
                "n_claims": 0,
                "n_supported": None,
                "n_unsupported": None,
                "n_conflict": None,
                "n_unknown": None,
                "recall_at_k": None,
                "summarizer_hallucination_rate": None,
                "synthesizer_hallucination_rate": None,
                "latency_s": 0.0,
                "domain": q.get("domain", ""),
                "query_type": q.get("query_type", ""),
                "error": str(e),
            }
        results.append(row)

    n = len(results)

    def _avg(key, subset=None):
        rows = subset if subset is not None else results
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _domain_summary(subset):
        m = len(subset)
        if m == 0:
            return {}
        return {
            "n_queries": m,
            "avg_citation_coverage": _avg("citation_coverage", subset),
            "avg_support_rate": _avg("support_rate", subset),
            "abstention_rate": round(sum(1 for r in subset if r.get("abstained")) / m, 4),
            "avg_recall_at_k": _avg("recall_at_k", subset),
            "avg_summarizer_hallucination_rate": _avg("summarizer_hallucination_rate", subset),
            "avg_synthesizer_hallucination_rate": _avg("synthesizer_hallucination_rate", subset),
            "avg_latency_s": _avg("latency_s", subset),
        }

    _EXPECTED_ABSTAIN = {"unanswerable", "out_of_domain"}
    answerable_results     = [r for r in results if r.get("query_type") not in _EXPECTED_ABSTAIN]
    expect_abstain_results = [r for r in results if r.get("query_type") in _EXPECTED_ABSTAIN]

    def _abstention_rate(subset):
        return round(sum(1 for r in subset if r.get("abstained")) / len(subset), 4) if subset else None

    # Group results by domain
    from collections import defaultdict
    domain_groups: dict = defaultdict(list)
    for r in results:
        domain_groups[r.get("domain") or "unknown"].append(r)
    by_domain = {d: _domain_summary(rows) for d, rows in sorted(domain_groups.items())}

    # Recall@k split by query type
    qtype_groups: dict = defaultdict(list)
    for r in results:
        qtype_groups[r.get("query_type") or "unknown"].append(r)
    recall_by_query_type = {
        qt: _avg("recall_at_k", rows)
        for qt, rows in sorted(qtype_groups.items())
    }

    ragas_scores = {}
    if not args.skip_ragas:
        print("\nRunning RAGAS evaluation...")
        try:
            ragas_scores = run_ragas(results, ragas_model)
            print(f"  context_recall={ragas_scores.get('ragas_context_recall')}  "
                  f"faithfulness={ragas_scores.get('ragas_faithfulness')}  "
                  f"answer_relevancy={ragas_scores.get('ragas_answer_relevancy')}")
        except Exception as e:
            print(f"  RAGAS evaluation failed: {e}")

    report = {
        "mode": mode_tag,
        "n_queries": n,
        "avg_citation_coverage": _avg("citation_coverage"),
        "avg_support_rate": _avg("support_rate"),
        "avg_latency_s": _avg("latency_s"),
        "abstention_rate_answerable": _abstention_rate(answerable_results),
        "correct_abstention_rate": _abstention_rate(expect_abstain_results),
        "avg_recall_at_k": _avg("recall_at_k"),
        "recall_at_k_by_query_type": recall_by_query_type,
        "avg_summarizer_hallucination_rate": _avg("summarizer_hallucination_rate"),
        "avg_synthesizer_hallucination_rate": _avg("synthesizer_hallucination_rate"),
        **ragas_scores,
        "by_domain": by_domain,
        "per_query": results,
    }

    Path(results_output).parent.mkdir(parents=True, exist_ok=True)
    with open(results_output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Eval Results  ({n} queries)  mode={mode_tag}")
    print(f"{'='*60}")
    print(f"  Citation Coverage        {report['avg_citation_coverage']}")
    print(f"  Support Rate             {report['avg_support_rate']}")
    print(f"  Abstention Rate (answerable)  {report['abstention_rate_answerable']}")
    print(f"  Correct Abstention Rate       {report['correct_abstention_rate']}")
    print(f"  Avg Recall@k             {report['avg_recall_at_k']}")
    for qt, val in report["recall_at_k_by_query_type"].items():
        print(f"    [{qt}] {val}")
    print(f"  Summarizer Hal. Rate     {report['avg_summarizer_hallucination_rate']}")
    print(f"  Synthesizer Hal. Rate    {report['avg_synthesizer_hallucination_rate']}")
    print(f"  Avg Latency              {report['avg_latency_s']}s")
    if ragas_scores:
        print(f"  RAGAS Context Recall     {ragas_scores.get('ragas_context_recall')}")
        print(f"  RAGAS Faithfulness       {ragas_scores.get('ragas_faithfulness')}")
        print(f"  RAGAS Answer Relevancy   {ragas_scores.get('ragas_answer_relevancy')}")
    if by_domain:
        print(f"\n  By Domain:")
        for domain, stats in by_domain.items():
            print(f"    [{domain}] n={stats['n_queries']}  "
                  f"cov={stats['avg_citation_coverage']}  "
                  f"sup={stats['avg_support_rate']}  "
                  f"abstain={stats['abstention_rate']}  "
                  f"rec@k={stats['avg_recall_at_k']}  "
                  f"hal_sum={stats['avg_summarizer_hallucination_rate']}  "
                  f"hal_syn={stats['avg_synthesizer_hallucination_rate']}")
    print(f"{'='*60}")
    print(f"Full report → {results_output}")


if __name__ == "__main__":
    main()
