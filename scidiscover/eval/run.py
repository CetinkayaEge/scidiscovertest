"""
Run evaluation pipeline over eval/queries.jsonl and report metrics.

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

def compute_recall_at_k(evidence_pack: dict, expected_paper_ids: list) -> float | None:
    """Fraction of expected paper_ids found in the retrieved chunks."""
    if not expected_paper_ids:
        return None
    retrieved = {ch["paper_id"] for ch in evidence_pack["chunks"]}
    hits = sum(1 for pid in expected_paper_ids if pid in retrieved)
    return round(hits / len(expected_paper_ids), 4)


def compute_hallucination_rate(summaries: list, evidence_pack: dict) -> float:
    """Fraction of [chunk_id] citations in summaries that are NOT in the retrieved evidence set."""
    valid_ids = {ch["chunk_id"] for ch in evidence_pack["chunks"]}
    pattern = re.compile(r'\[([^\]]+\|\|[^\]]+)\]')
    total = hallucinated = 0
    for ps in summaries:
        for cid in pattern.findall(ps.get("summary_text", "")):
            total += 1
            if cid not in valid_ids:
                hallucinated += 1
    return round(hallucinated / total, 4) if total > 0 else 0.0


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
                 labels_lookup: dict, skip_verifier: bool = False) -> dict:
    query = query_record["query"]
    qid = query_record.get("query_id", "")

    t0 = time.time()

    evidence_pack = agents["retriever_agent"].run(query, k=top_k)

    if "reranker_agent" in agents:
        evidence_pack = agents["reranker_agent"].run(evidence_pack)

    summaries = agents["summarizer_agent"].run(evidence_pack)

    synthesis = agents["synthesizer_agent"].run({
        "query": query,
        "paper_summaries": summaries,
    })

    latency_s = time.time() - t0

    # Recall@k — uses labels if available for this query
    expected = labels_lookup.get(qid, {}).get("expected_paper_ids", [])
    recall_at_k = compute_recall_at_k(evidence_pack, expected)

    # Summarizer hallucination rate
    hallucination_rate = compute_hallucination_rate(summaries, evidence_pack)

    # Store retrieved texts for RAGAS
    retrieved_contexts = [ch["text"] for ch in evidence_pack["chunks"]]

    if skip_verifier:
        draft = synthesis.get("draft_answer", "")
        key_claims = synthesis.get("key_claims", [])
        abstained = not draft or draft.startswith("ABSTAIN") or draft.startswith("[")
        final_answer = _ABSTAIN_MSG if abstained else draft
        citation_coverage = _citation_coverage_from_claims(key_claims)
        return {
            "query_id": qid,
            "query": query,
            "reference": query_record.get("reference", ""),
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
            "hallucination_rate": hallucination_rate,
            "latency_s": round(latency_s, 2),
            "domain": query_record.get("domain", ""),
        }

    # Full pipeline — run verifier
    verified = agents["verifier_agent"].run({
        "synthesis": synthesis,
        "evidence_pack": evidence_pack,
    })
    vs = verified["verification_summary"]

    return {
        "query_id": qid,
        "query": query,
        "reference": query_record.get("reference", ""),
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
        "hallucination_rate": hallucination_rate,
        "latency_s": round(latency_s, 2),
        "domain": query_record.get("domain", ""),
    }


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------

def run_ragas(results: list, ragas_model: str) -> dict:
    import warnings
    from datasets import Dataset
    from ragas import evaluate, RunConfig

    # ragas.metrics.collections only supports OpenAI InstructorLLM — use the
    # legacy ragas.metrics singletons which still work with LangchainLLMWrapper
    # (Gemini). Suppress the resulting deprecation warnings explicitly.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings

    from langchain_google_genai import ChatGoogleGenerativeAI

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
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=ragas_model))
    # Use local sentence-transformers for embeddings — avoids Google embedding
    # API version issues. The same model is already loaded for retrieval.
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    # Reduce concurrency and raise timeout to avoid rate-limit TimeoutErrors.
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
    parser.add_argument("--output", default=None,
                        help="Override output path (default: eval.results_output in config)")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_cfg = config["eval"]
    queries_path = eval_cfg["queries_path"]
    labels_path = eval_cfg.get("labels_path", "eval/labels.jsonl")
    results_output = args.output or eval_cfg["results_output"]
    top_k = eval_cfg.get("top_k_recall", 20)
    max_queries = args.max_queries or eval_cfg.get("max_queries")
    ragas_model = eval_cfg.get("ragas_model", "gemini-2.5-flash")

    # Derive a human-readable mode tag for the report
    reranker_on = config.get("reranker", {}).get("enabled", False)
    llm_model = config["llm"]["model"]
    mode_tag = f"{'no_verifier' if args.skip_verifier else 'full'}_{'no_reranker' if not reranker_on else 'reranker'}_{llm_model.split('-')[0]}"

    # Agents must be imported before utils.llm_client on Windows to avoid
    # a silent crash caused by google.genai and sentence_transformers both
    # trying to register the google namespace package in the same process.
    from scidiscover.agents.retriever import Retriever
    from scidiscover.agents.retriever_agent import RetrieverAgent
    from scidiscover.agents.summarizer import SummarizerAgent
    from scidiscover.agents.synthesizer import SynthesizerAgent
    from scidiscover.agents.reranker import RerankerAgent
    from scidiscover.agents.verifier import VerifierAgent

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

    verifier_cfg = config["verifier"]
    agents = {
        "retriever_agent": RetrieverAgent(retriever),
        "summarizer_agent": SummarizerAgent(
            prompt_path=config["summarizer"]["prompt_path"],
            traces_output=config["summarizer"]["traces_output"],
            output_path=config["summarizer"]["output_path"],
        ),
        "synthesizer_agent": SynthesizerAgent(
            prompt_path=config["synthesizer"]["prompt_path"],
            traces_output=config["synthesizer"]["traces_output"],
            output_path=config["synthesizer"]["output_path"],
        ),
        "verifier_agent": VerifierAgent(
            prompt_path=verifier_cfg["prompt_path"],
            traces_output=verifier_cfg["traces_output"],
            verification_output=verifier_cfg["verification_output"],
            answers_output=verifier_cfg["answers_output"],
            min_citation_coverage=verifier_cfg.get("min_citation_coverage", 0.95),
            min_support_rate=verifier_cfg.get("min_support_rate", 0.50),
        ),
    }

    reranker_cfg = config.get("reranker", {})
    if reranker_cfg.get("enabled", False):
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

    # Load labels for Recall@k (optional)
    labels_lookup: dict = {}
    if Path(labels_path).exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                labels_lookup[entry["query_id"]] = entry
        print(f"Loaded {len(labels_lookup)} labeled queries for Recall@k.")
    else:
        print(f"No labels file found at {labels_path} — Recall@k will be None.")

    print(f"Running eval on {len(queries)} queries (top_k={top_k})...")

    results = []
    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q.get('query_id', '')} — {q['query'][:70]}...")
        try:
            row = run_pipeline(q, agents, top_k, labels_lookup,
                               skip_verifier=args.skip_verifier)
            rec = row["recall_at_k"]
            hal = row["hallucination_rate"]
            print(f"    cov={row['citation_coverage']:.2f}  "
                  f"support={row['support_rate']:.2f}  "
                  f"rec@k={rec if rec is not None else 'N/A'}  "
                  f"hal={hal:.2f}  "
                  f"abstain={row['abstained']}  "
                  f"{row['latency_s']}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            row = {
                "query_id": q.get("query_id", ""),
                "query": q["query"],
                "reference": q.get("reference", ""),
                "retrieved_contexts": [],
                "final_answer": "",
                "abstained": True,
                "citation_coverage": 0.0,
                "support_rate": 0.0,
                "n_claims": 0,
                "n_supported": 0,
                "n_unsupported": 0,
                "n_conflict": 0,
                "n_unknown": 0,
                "recall_at_k": None,
                "hallucination_rate": None,
                "latency_s": 0.0,
                "domain": q.get("domain", ""),
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
            "avg_hallucination_rate": _avg("hallucination_rate", subset),
            "avg_latency_s": _avg("latency_s", subset),
        }

    # Group results by domain
    from collections import defaultdict
    domain_groups: dict = defaultdict(list)
    for r in results:
        domain_groups[r.get("domain") or "unknown"].append(r)
    by_domain = {d: _domain_summary(rows) for d, rows in sorted(domain_groups.items())}

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
        "abstention_rate": round(sum(1 for r in results if r.get("abstained")) / n, 4) if n else 0.0,
        "avg_recall_at_k": _avg("recall_at_k"),
        "avg_hallucination_rate": _avg("hallucination_rate"),
        "n_labeled_queries": len(labels_lookup),
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
    print(f"  Abstention Rate          {report['abstention_rate']}")
    print(f"  Avg Recall@k             {report['avg_recall_at_k']}")
    print(f"  Avg Hallucination Rate   {report['avg_hallucination_rate']}")
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
                  f"hal={stats['avg_hallucination_rate']}")
    print(f"{'='*60}")
    print(f"Full report → {results_output}")


if __name__ == "__main__":
    main()
