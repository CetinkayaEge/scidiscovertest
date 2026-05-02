"""
Run evaluation pipeline over eval/queries.jsonl and report metrics.

Usage:
    PYTHONPATH=. python -m scidiscover.eval.run --config configs/demo.yaml
    PYTHONPATH=. python -m scidiscover.eval.run --config configs/demo.yaml --skip-ragas
    PYTHONPATH=. python -m scidiscover.eval.run --config configs/demo.yaml --max-queries 5
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def run_pipeline(query_record: dict, agents: dict, top_k: int) -> dict:
    query = query_record["query"]

    t0 = time.time()

    evidence_pack = agents["retriever_agent"].run(query, k=top_k)

    if "reranker_agent" in agents:
        evidence_pack = agents["reranker_agent"].run(evidence_pack)

    summaries = agents["summarizer_agent"].run(evidence_pack)

    synthesis = agents["synthesizer_agent"].run({
        "query": query,
        "paper_summaries": summaries,
    })

    verified = agents["verifier_agent"].run({
        "synthesis": synthesis,
        "evidence_pack": evidence_pack,
    })

    latency_s = time.time() - t0
    vs = verified["verification_summary"]

    # Store actual retrieved texts so RAGAS context_recall can judge them vs. ground truth
    retrieved_contexts = [ch["text"] for ch in evidence_pack["chunks"]]

    return {
        "query_id": query_record.get("query_id", ""),
        "query": query,
        "reference": query_record.get("reference", ""),
        "retrieved_contexts": retrieved_contexts,
        "final_answer": verified["final_answer"],
        "citation_coverage": round(vs["citation_coverage"], 4),
        "support_rate": round(vs["support_rate"], 4),
        "latency_s": round(latency_s, 2),
        "domain": query_record.get("domain", ""),
    }


def run_ragas(results: list, ragas_model: str) -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import GoogleEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI

    rows = []
    for r in results:
        # Use what the system actually retrieved — not the testset's ideal reference_contexts
        contexts = r.get("retrieved_contexts") or [""]
        rows.append({
            "question": r["query"],
            "answer": r["final_answer"] or "",
            "contexts": contexts,
            "ground_truth": r.get("reference", ""),
        })

    dataset = Dataset.from_list(rows)
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=ragas_model))
    emb = GoogleEmbeddings(model="gemini-embedding-001")

    score = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=emb,
    )
    df = score.to_pandas()
    return {
        "ragas_faithfulness": round(float(df["faithfulness"].mean()), 4),
        "ragas_answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
        "ragas_context_recall": round(float(df["context_recall"].mean()), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_cfg = config["eval"]
    queries_path = eval_cfg["queries_path"]
    results_output = eval_cfg["results_output"]
    top_k_recall = eval_cfg.get("top_k_recall", 20)
    max_queries = args.max_queries or eval_cfg.get("max_queries")
    ragas_model = eval_cfg.get("ragas_model", "gemini-2.5-flash")

    from utils.llm_client import configure_llm
    configure_llm(config["llm"]["model"], config["llm"]["max_tokens"])

    from scidiscover.agents.retriever import Retriever
    from scidiscover.agents.retriever_agent import RetrieverAgent
    from scidiscover.agents.summarizer import SummarizerAgent
    from scidiscover.agents.synthesizer import SynthesizerAgent
    from scidiscover.agents.reranker import RerankerAgent
    from scidiscover.agents.verifier import VerifierAgent

    retriever = Retriever(
        chunks_input=config["retrieval"]["chunks_input"],
        index_output=config["retrieval"]["index_output"],
        chunk_ids_output=config["retrieval"]["chunk_ids_output"],
        traces_output=config["retrieval"]["traces_output"],
        model_name=config["retrieval"]["model_name"],
        papers_input=config["retrieval"]["papers_input"],
    )

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
            prompt_path=config["verifier"]["prompt_path"],
            traces_output=config["verifier"]["traces_output"],
            verification_output=config["verifier"]["verification_output"],
            answers_output=config["verifier"]["answers_output"],
        ),
    }

    reranker_cfg = config.get("reranker", {})
    if reranker_cfg.get("enabled", False):
        agents["reranker_agent"] = RerankerAgent(
            model_name=reranker_cfg["model_name"],
            top_k=reranker_cfg.get("top_k", top_k_recall),
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

    print(f"Running eval on {len(queries)} queries (top_k={top_k_recall})...")

    results = []
    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q.get('query_id', '')} — {q['query'][:70]}...")
        try:
            row = run_pipeline(q, agents, top_k_recall)
            print(f"    cov={row['citation_coverage']:.2f}  support={row['support_rate']:.2f}  {row['latency_s']}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            row = {
                "query_id": q.get("query_id", ""),
                "query": q["query"],
                "reference": q.get("reference", ""),
                "retrieved_contexts": [],
                "final_answer": "",
                "citation_coverage": 0.0,
                "support_rate": 0.0,
                "latency_s": 0.0,
                "domain": q.get("domain", ""),
                "error": str(e),
            }
        results.append(row)

    n = len(results)
    avg_cov = sum(r["citation_coverage"] for r in results) / n if n else 0.0
    avg_support = sum(r["support_rate"] for r in results) / n if n else 0.0
    avg_latency = sum(r["latency_s"] for r in results) / n if n else 0.0

    ragas_scores = {}
    if not args.skip_ragas:
        print("\nRunning RAGAS evaluation (context_recall + faithfulness + answer_relevancy)...")
        try:
            ragas_scores = run_ragas(results, ragas_model)
            print(f"  context_recall={ragas_scores['ragas_context_recall']:.4f}  "
                  f"faithfulness={ragas_scores['ragas_faithfulness']:.4f}  "
                  f"answer_relevancy={ragas_scores['ragas_answer_relevancy']:.4f}")
        except Exception as e:
            print(f"  RAGAS evaluation failed: {e}")

    report = {
        "n_queries": n,
        "avg_citation_coverage": round(avg_cov, 4),
        "avg_support_rate": round(avg_support, 4),
        "avg_latency_s": round(avg_latency, 2),
        **ragas_scores,
        "per_query": results,
    }

    Path(results_output).parent.mkdir(parents=True, exist_ok=True)
    with open(results_output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"Eval Results  ({n} queries)")
    print(f"{'='*55}")
    print(f"  Citation Coverage       {avg_cov:.4f}")
    print(f"  Support Rate            {avg_support:.4f}")
    print(f"  Avg Latency             {avg_latency:.2f}s")
    if ragas_scores:
        print(f"  RAGAS Context Recall    {ragas_scores['ragas_context_recall']:.4f}")
        print(f"  RAGAS Faithfulness      {ragas_scores['ragas_faithfulness']:.4f}")
        print(f"  RAGAS Answer Relevancy  {ragas_scores['ragas_answer_relevancy']:.4f}")
    print(f"{'='*55}")
    print(f"Full report → {results_output}")


if __name__ == "__main__":
    main()
