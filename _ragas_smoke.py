"""Throwaway RAGAS smoke test — 3 rows, verbose, tight timeouts."""
import sys, types, time, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Stub VertexAI imports that RAGAS pulls in but we don't use.
class _S: pass
for m,a in (("langchain_community.chat_models.vertexai","ChatVertexAI"),
            ("langchain_community.llms.vertexai","VertexAI")):
    mod = types.ModuleType(m); setattr(mod,a,_S); sys.modules[m]=mod

print("[1] imports", flush=True)
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from utils.eval_llm import make_eval_llm

print("[2] eval LLM (OpenAI gpt-4o-mini via OPENAI_API_KEY_SOTA)", flush=True)
import os
from langchain_openai import ChatOpenAI
llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY_SOTA"],
    base_url="https://api.openai.com/v1",
    temperature=0,
    max_retries=2,
    timeout=60,
))

print("[3] embeddings (first run downloads model)", flush=True)
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

print("[4] load 3 sample rows", flush=True)
r = json.load(open("paperqa_baseline/results/paperqa_results.json"))
rows = []
for q in r["per_query"]:
    if q.get("answer") and not q.get("abstained"):
        rows.append({
            "question": q["query"],
            "answer": q["answer"],
            "contexts": [ep["passage"] for ep in q.get("evidence_passages", []) if ep.get("passage")] or [""],
            "ground_truth": q.get("reference", ""),
        })
        if len(rows) >= 3: break
print(f"  rows={len(rows)}", flush=True)

faithfulness.llm = llm
context_recall.llm = llm
answer_relevancy.llm = llm
answer_relevancy.embeddings = emb

ds = Dataset.from_list(rows)
print("[5] evaluate (3 rows, 60s timeout, 1 worker)", flush=True)
t0 = time.time()
score = evaluate(
    dataset=ds,
    metrics=[faithfulness, answer_relevancy, context_recall],
    llm=llm, embeddings=emb,
    run_config=RunConfig(timeout=60, max_retries=1, max_wait=30, max_workers=1),
    raise_exceptions=False,
)
print(f"[6] DONE in {time.time()-t0:.1f}s", flush=True)
print(score)
