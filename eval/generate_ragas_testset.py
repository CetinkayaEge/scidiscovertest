"""
Generate eval/queries.jsonl using RAGAS TestsetGenerator from the current corpus,
supplemented with unanswerable and out-of-domain queries.

Query types produced:
  ragas_single_hop  — RAGAS SingleHopSpecificQuerySynthesizer  (difficulty: easy)
  ragas_multi_hop   — RAGAS MultiHop*Synthesizer               (difficulty: hard)
  unanswerable      — in-domain questions the abstract cannot answer
  out_of_domain     — fixed questions completely outside the corpus

Usage:
    PYTHONPATH=. python eval/generate_ragas_testset.py
    PYTHONPATH=. python eval/generate_ragas_testset.py \
        --n-chunks 300 --testset-size 40 --unanswerable 5 \
        --chunks data/processed/chunks.jsonl \
        --output eval/queries.jsonl \
        --seed 42
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "healthcare": [
        "patient", "clinical", "disease", "treatment", "medical",
        "drug", "hospital", "diagnosis", "surgery", "vaccine",
        "cancer", "tumor", "epidemi", "placebo",
    ],
    "sustainability": [
        "sustainability", "sustainable", "carbon", "renewable energy",
        "climate change", "biodiversity", "waste", "circular economy",
        "environmental", "green", "net zero", "emissions",
    ],
    "interdisciplinary": [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "natural language", "robotics", "blockchain",
        "digital", "internet of things",
    ],
}

OUT_OF_DOMAIN_QUERIES = [
    {"query": "In what year did the Voyager 1 spacecraft cross the heliopause and enter interstellar space?", "difficulty": "easy"},
    {"query": "What is the boiling point of liquid nitrogen at standard atmospheric pressure?", "difficulty": "easy"},
    {"query": "What were the primary economic causes of the fall of the Western Roman Empire?", "difficulty": "medium"},
    {"query": "What tectonic processes are responsible for the formation of the Himalayan mountain range?", "difficulty": "medium"},
    {"query": "What are the main principles of quantum error correction in fault-tolerant quantum computing?", "difficulty": "hard"},
]

# RAGAS synthesizer name → difficulty
SYNTHESIZER_DIFFICULTY = {
    "singlehopspecificquery": "easy",
    "multihopspecificquery":  "medium",
    "multihopabstractquery":  "hard",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_domain(text: str) -> str:
    lower = text.lower()
    scores = {d: sum(1 for kw in kws if kw in lower) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "interdisciplinary"


def synthesizer_to_difficulty(name: str) -> str:
    key = re.sub(r"[^a-z]", "", name.lower())
    for pattern, diff in SYNTHESIZER_DIFFICULTY.items():
        if pattern in key:
            return diff
    return "medium"


def synthesizer_to_query_type(name: str) -> str:
    key = name.lower()
    if "singlehop" in key or "single_hop" in key:
        return "ragas_single_hop"
    if "multihop" in key or "multi_hop" in key:
        return "ragas_multi_hop"
    return "ragas"


def _is_retryable(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("503", "502", "429", "quota", "rate limit",
                                   "unavailable", "overloaded", "resource_exhausted"))


def _retry(fn, retries: int = 4, base_wait: float = 15.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1 and _is_retryable(e):
                wait = base_wait * (2 ** attempt)
                print(f"    Retry {attempt + 1}/{retries}: {type(e).__name__} — waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise


def _parse_json_response(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------

def load_sample_chunks(chunks_path: str, n: int, seed: int) -> list[dict]:
    """Sample n ABSTRACT chunks with at least 100 tokens."""
    pool = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("section") == "ABSTRACT" and c.get("token_len", 0) >= 100:
                pool.append(c)
    random.seed(seed)
    random.shuffle(pool)
    return pool[:n]


# ---------------------------------------------------------------------------
# Unanswerable query generation (same approach as data/eval generator)
# ---------------------------------------------------------------------------

def make_unanswerable_query(chunk: dict, llm) -> dict:
    prompt = f"""You are creating evaluation questions for a scientific RAG system.

Below is the abstract of a scientific paper:

{chunk['text'][:2000]}

Write ONE question that:
- Is clearly about the topic of this paper
- Asks for specific experimental details, numeric results, statistical values, or mechanistic steps that are NOT stated in this abstract (they would only appear in the full paper body)
- Sounds like a natural scientific question a researcher would ask
- Is self-contained — no mention of "abstract", "paper", or "text"
- Is 10–25 words long

Return ONLY valid JSON: {{"question": "..."}}"""

    response = llm.invoke(prompt)
    return _parse_json_response(response.content)


def generate_unanswerable(chunks: list, n_total: int, used_ids: set,
                           q_index: int, llm, seed: int) -> list[dict]:
    random.seed(seed)
    pool = [c for c in chunks if c["chunk_id"] not in used_ids]
    random.shuffle(pool)

    rows = []
    for chunk in pool:
        if len(rows) >= n_total:
            break
        try:
            result = _retry(lambda ch=chunk: make_unanswerable_query(ch, llm))
            q = (result.get("question") or "").strip()
            if len(q.split()) < 8:
                continue
            rows.append({
                "query_id": f"q{q_index:03d}",
                "query": q,
                "reference": "",
                "reference_contexts": [],
                "domain": classify_domain(chunk["text"]),
                "difficulty": "unanswerable",
                "query_type": "unanswerable",
                "expected_paper_ids": [],
                "expected_chunk_ids": [],
            })
            used_ids.add(chunk["chunk_id"])
            q_index += 1
            print(f"  [unanswerable] {q[:70]}...")
        except Exception as e:
            print(f"  [unanswerable] skipped: {e}")
        time.sleep(1)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks",       default="data/processed/chunks.jsonl")
    parser.add_argument("--output",       default="eval/queries.jsonl")
    parser.add_argument("--n-chunks",     type=int, default=300,
                        help="Chunks fed to RAGAS knowledge graph builder")
    parser.add_argument("--testset-size", type=int, default=40,
                        help="RAGAS queries to generate")
    parser.add_argument("--unanswerable", type=int, default=5,
                        help="Unanswerable queries to append")
    parser.add_argument("--no-out-of-domain", action="store_true",
                        help="Skip out-of-domain queries")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--config",     default="configs/demo.yaml",
                        help="Path to demo.yaml — ragas_model is read from eval.ragas_model")
    parser.add_argument("--eval-model", default=None,
                        help="Override eval.ragas_model from config")
    args = parser.parse_args()

    # resolve eval model: CLI flag > config file > hardcoded fallback
    import yaml
    with open(args.config, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
    eval_model = args.eval_model or _cfg.get("eval", {}).get("ragas_model", "gemini-2.5-flash")

    # ── LLM / embeddings ────────────────────────────────────────────────────
    print(f"Initialising models (eval-model={eval_model})...")
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.testset import TestsetGenerator
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import RunConfig
    from langchain_core.documents import Document
    from utils.eval_llm import make_eval_llm

    lc_llm = make_eval_llm(eval_model, temperature=0.3)
    llm    = LangchainLLMWrapper(lc_llm)
    emb    = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # ── Load chunks ──────────────────────────────────────────────────────────
    print(f"Sampling {args.n_chunks} chunks from {args.chunks}...")
    chunks = load_sample_chunks(args.chunks, args.n_chunks, args.seed)
    print(f"  {len(chunks)} chunks loaded")

    # Build a lookup: normalised 120-char prefix → chunk metadata
    chunk_lookup: dict[str, dict] = {}
    for c in chunks:
        key = " ".join(c["text"][:120].split())
        chunk_lookup[key] = c

    # ── RAGAS generation ─────────────────────────────────────────────────────
    docs = [
        Document(
            page_content=c["text"],
            metadata={"chunk_id": c["chunk_id"], "paper_id": c["paper_id"]},
        )
        for c in chunks
    ]

    generator  = TestsetGenerator(llm=llm, embedding_model=emb)
    run_config = RunConfig(timeout=180, max_retries=3, max_wait=90, max_workers=2)

    print(f"Generating {args.testset_size} RAGAS queries...")
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=args.testset_size,
        run_config=run_config,
        raise_exceptions=False,
    )

    rows     = testset.to_list()
    used_ids: set[str] = set()
    all_rows: list[dict] = []
    q_index  = 1

    print(f"  {len(rows)} RAGAS samples generated")

    for row in rows:
        query       = (row.get("user_input") or row.get("question") or "").strip()
        reference   = (row.get("reference") or "").strip()
        synthesizer = row.get("synthesizer_name", "")

        ref_contexts = row.get("reference_contexts") or []
        if isinstance(ref_contexts, str):
            ref_contexts = [ref_contexts]
        ref_contexts = [re.sub(r"^<\d+-hop>\s*", "", c.strip()) for c in ref_contexts]

        # Resolve expected IDs from reference_contexts
        expected_paper_ids: list[str] = []
        expected_chunk_ids: list[str] = []
        for ctx in ref_contexts:
            key = " ".join(ctx[:120].split())
            match = chunk_lookup.get(key)
            if match is None:
                # try shorter prefix
                for plen in (80, 60, 40):
                    short = " ".join(ctx[:plen].split())
                    for ck, cv in chunk_lookup.items():
                        if ck.startswith(short):
                            match = cv
                            break
                    if match:
                        break
            if match:
                if match["paper_id"] not in expected_paper_ids:
                    expected_paper_ids.append(match["paper_id"])
                if match["chunk_id"] not in expected_chunk_ids:
                    expected_chunk_ids.append(match["chunk_id"])
                used_ids.add(match["chunk_id"])

        domain     = classify_domain(" ".join(ref_contexts))
        difficulty = synthesizer_to_difficulty(synthesizer)
        qtype      = synthesizer_to_query_type(synthesizer)

        all_rows.append({
            "query_id":            f"q{q_index:03d}",
            "query":               query,
            "reference":           reference,
            "reference_contexts":  ref_contexts,
            "domain":              domain,
            "difficulty":          difficulty,
            "query_type":          qtype,
            "synthesizer_name":    synthesizer,
            "expected_paper_ids":  expected_paper_ids,
            "expected_chunk_ids":  expected_chunk_ids,
        })
        q_index += 1

    # ── Out-of-domain ────────────────────────────────────────────────────────
    if not args.no_out_of_domain:
        print(f"\nAdding {len(OUT_OF_DOMAIN_QUERIES)} out-of-domain queries...")
        for ood in OUT_OF_DOMAIN_QUERIES:
            all_rows.append({
                "query_id":           f"q{q_index:03d}",
                "query":              ood["query"],
                "reference":          "",
                "reference_contexts": [],
                "domain":             "out_of_domain",
                "difficulty":         ood["difficulty"],
                "query_type":         "out_of_domain",
                "synthesizer_name":   "",
                "expected_paper_ids": [],
                "expected_chunk_ids": [],
            })
            print(f"  {ood['difficulty']:6s}: {ood['query'][:70]}")
            q_index += 1

    # ── Unanswerable ─────────────────────────────────────────────────────────
    if args.unanswerable > 0:
        print(f"\nGenerating {args.unanswerable} unanswerable queries...")
        unans = generate_unanswerable(
            chunks, args.unanswerable, used_ids, q_index, lc_llm, args.seed + 99
        )
        all_rows.extend(unans)
        q_index += len(unans)

    # ── Write output ─────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    from collections import Counter
    types = Counter(r["query_type"]  for r in all_rows)
    diffs = Counter(r["difficulty"]  for r in all_rows)
    doms  = Counter(r["domain"]      for r in all_rows)
    has_paper = sum(1 for r in all_rows if r["expected_paper_ids"])
    has_chunk = sum(1 for r in all_rows if r["expected_chunk_ids"])

    print(f"\n{'='*55}")
    print(f"Saved {len(all_rows)} queries → {args.output}")
    print(f"  Query types:         {dict(types)}")
    print(f"  Difficulties:        {dict(diffs)}")
    print(f"  Domains:             {dict(doms)}")
    print(f"  Has expected papers: {has_paper}/{len(all_rows)}")
    print(f"  Has expected chunks: {has_chunk}/{len(all_rows)}")
    print(f"\nNext step: PYTHONPATH=. python eval/build_labels.py")


if __name__ == "__main__":
    main()
