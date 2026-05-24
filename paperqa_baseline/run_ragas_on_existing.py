"""Run RAGAS on the existing paperqa_results.json without re-querying.

Uses OpenAI gpt-5.4-mini (via OPENAI_API_KEY_SOTA) as the RAGAS eval LLM.
The Gemini free tier's 20-RPD quota is too low for 74 results × 3 metrics.
"""
from __future__ import annotations

# RAGAS imports langchain_community.chat_models.vertexai + .llms.vertexai at
# module load time, but we use OpenAI for evaluation.  Recent langchain_community
# versions either lack the modules or fail with a metaclass conflict; stub them
# so the imports resolve cleanly without ever instantiating VertexAI.
import sys as _sys
import types as _types
class _VertexStub:  # pragma: no cover - stub, never instantiated
    pass
for _modname, _attr in (
    ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
    ("langchain_community.llms.vertexai", "VertexAI"),
):
    _m = _types.ModuleType(_modname)
    setattr(_m, _attr, _VertexStub)
    _sys.modules[_modname] = _m

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS = Path(__file__).parent / "results" / "paperqa_results.json"


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY_SOTA", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY_SOTA missing from .env")

    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    logger.info("Building eval LLM (gpt-5.4-mini)...")
    llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-5.4-mini",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        temperature=0,
        max_retries=2,
        timeout=60,
    ))

    logger.info("Loading embeddings (all-MiniLM-L6-v2)...")
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    logger.info("Loading PaperQA2 results from %s", RESULTS)
    report = json.loads(RESULTS.read_text(encoding="utf-8"))

    rows = []
    for q in report["per_query"]:
        ans = q.get("answer") or ""
        if not ans or q.get("abstained"):
            continue
        rows.append({
            "question": q["query"],
            "answer": ans,
            "contexts": [
                ep["passage"] for ep in (q.get("evidence_passages") or [])
                if ep.get("passage")
            ] or [""],
            "ground_truth": q.get("reference", ""),
        })

    if not rows:
        logger.error("No non-abstained rows — nothing to evaluate.")
        return

    logger.info("Evaluating %d rows with RAGAS (faithfulness, answer_relevancy, context_recall)...", len(rows))
    faithfulness.llm = llm
    context_recall.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = emb

    ds = Dataset.from_list(rows)
    t0 = time.time()
    score = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(timeout=90, max_retries=2, max_wait=60, max_workers=2),
        raise_exceptions=False,
    )
    logger.info("RAGAS finished in %.0fs", time.time() - t0)

    df = score.to_pandas()
    scores = {
        "ragas_faithfulness": round(float(df["faithfulness"].mean(skipna=True)), 4),
        "ragas_answer_relevancy": round(float(df["answer_relevancy"].mean(skipna=True)), 4),
        "ragas_context_recall": round(float(df["context_recall"].mean(skipna=True)), 4),
        "ragas_eval_model": "gpt-5.4-mini",
        "ragas_n_rows": len(rows),
    }
    report.update(scores)
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Merged RAGAS scores into %s:", RESULTS)
    for k, v in scores.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
