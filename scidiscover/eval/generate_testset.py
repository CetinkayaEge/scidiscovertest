"""
Generate the full evaluation test set and write data/eval/queries.jsonl
and data/eval/labels.jsonl from scratch.

Query types produced
--------------------
standard    — single-chunk, easy/medium/hard per domain
out_of_domain — fixed questions outside the corpus (no expected papers)
multi_source  — synthesis questions requiring 2 papers
unanswerable  — in-domain questions whose answers are not in the abstract

Usage
-----
    PYTHONPATH=. python scidiscover/eval/generate_testset.py
    PYTHONPATH=. python scidiscover/eval/generate_testset.py \
        --n-queries 30 --multi 6 --unanswerable 5 \
        --chunks data/processed/chunks.jsonl \
        --output data/eval/queries.jsonl \
        --labels data/eval/labels.jsonl \
        --seed 42
"""

import argparse
import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "healthcare": [
        "patient", "clinical", "disease", "treatment", "medical",
        "covid", "drug therapy", "hospital", "diagnosis",
        "surgery", "pharmacol", "epidemi", "vaccine", "cancer", "tumor",
        "randomized controlled", "placebo", "adverse effect",
    ],
    "sustainability": [
        "sustainability", "sustainable development", "esg",
        "carbon emission", "carbon footprint", "renewable energy",
        "green building", "climate change", "net zero",
        "biodiversity", "waste management", "circular economy",
        "environmental impact", "life cycle assessment",
        "construction waste", "urban planning",
    ],
    "interdisciplinary": [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "natural language processing",
        "additive manufacturing", "robotics", "data science",
        "internet of things", "blockchain", "digital twin",
        "cross-disciplinary", "multidisciplinary", "human-computer",
    ],
}

OUT_OF_DOMAIN_QUERIES = [
    {"query": "In what year did the Voyager 1 spacecraft cross the heliopause and enter interstellar space?", "difficulty": "easy"},
    {"query": "What is the boiling point of liquid nitrogen at standard atmospheric pressure?", "difficulty": "easy"},
    {"query": "What were the primary economic causes of the fall of the Western Roman Empire?", "difficulty": "medium"},
    {"query": "What tectonic processes are responsible for the formation of the Himalayan mountain range?", "difficulty": "medium"},
    {"query": "What are the main principles of quantum error correction in fault-tolerant quantum computing?", "difficulty": "hard"},
]

DIFFICULTY_PROMPTS = {
    "easy": (
        "Write an EASY question — ask for a single specific fact, number, name, or term "
        "that is stated almost verbatim in the passage (e.g. 'What percentage...', "
        "'How many...', 'What is the name of...')."
    ),
    "medium": (
        "Write a MEDIUM question — ask about the main finding, purpose, or methodology "
        "of the passage using different vocabulary than the source text. "
        "The answer requires understanding the text, not just copying a phrase."
    ),
    "hard": (
        "Write a HARD question — ask about a mechanism, cause-effect relationship, "
        "comparison, or multi-step inference that requires deep comprehension "
        "(e.g. 'Why does...', 'How does X lead to Y...', 'What is the relationship between...')."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_domain(text: str) -> str:
    lower = text.lower()
    scores = {d: sum(1 for kw in kws if kw in lower) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "interdisciplinary"


def _parse_json_response(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _retry(fn, retries: int = 2, wait: int = 10):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt + 1}: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------

def load_chunks_by_domain(chunks_path: str, max_per_domain: int) -> dict:
    buckets: dict[str, list] = {d: [] for d in DOMAIN_KEYWORDS}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("section") not in ("ABSTRACT", "BODY"):
                continue
            token_len = c.get("token_len", len(c.get("text", "").split()))
            if token_len < 120 or not c.get("text", "").strip():
                continue
            domain = classify_domain(c["text"])
            if len(buckets[domain]) < max_per_domain:
                buckets[domain].append(c)
    return buckets


def load_abstract_chunks_by_domain(chunks_path: str, min_tokens: int = 120) -> dict:
    """Separate loader for abstract-only chunks used by multi-source / unanswerable."""
    buckets: dict[str, list] = {d: [] for d in DOMAIN_KEYWORDS}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("section") != "ABSTRACT":
                continue
            if c.get("token_len", len(c.get("text", "").split())) < min_tokens:
                continue
            domain = classify_domain(c["text"])
            buckets[domain].append(c)
    return buckets


# ---------------------------------------------------------------------------
# Standard query generation (single-source)
# ---------------------------------------------------------------------------

def make_questions_batch(chunks: list, difficulty: str) -> list[str]:
    passages = "\n\n".join(f"[{i+1}] {c['text'][:2000]}" for i, c in enumerate(chunks))
    n = len(chunks)
    prompt = f"""You are creating evaluation questions for a scientific RAG system.

Below are {n} scientific passages numbered [1] to [{n}].
For each passage write exactly ONE question at the specified difficulty level.

Difficulty level: {difficulty.upper()}
{DIFFICULTY_PROMPTS[difficulty]}

Rules:
- Output a JSON array of {n} strings, in the same order as the passages.
- Do NOT mention "the text", "the passage", or "the chunk".
- Do NOT answer the questions.
- Each question must be self-contained and specific.
- Minimum 6 words per question.

Passages:
{passages}

Return ONLY a JSON array, no extra text.
"""
    response = llm.invoke(prompt)
    return _parse_json_response(response.content)


def make_ground_truths_batch(chunks: list, questions: list[str]) -> list[str]:
    pairs = "\n\n".join(
        f"[{i+1}] Passage: {c['text'][:2000]}\nQuestion: {q}"
        for i, (c, q) in enumerate(zip(chunks, questions))
    )
    n = len(chunks)
    prompt = f"""You are building a reference answer set for evaluating a scientific RAG system.

Below are {n} passage–question pairs numbered [1] to [{n}].
For each pair write a concise factual answer (1–3 sentences) that:
- Is fully grounded in the passage (do not add outside knowledge)
- Directly and completely answers the question
- Uses clear, precise language

Pairs:
{pairs}

Return ONLY a JSON array of {n} strings (one answer per pair), no extra text.
"""
    response = llm.invoke(prompt)
    return _parse_json_response(response.content)


def generate_standard(domain: str, chunks: list, difficulty_split: dict,
                      batch_size: int, q_index: int, used_ids: set) -> list[dict]:
    rows = []
    pool = list(chunks)
    offset = 0

    for difficulty, count in difficulty_split.items():
        if count == 0:
            continue

        target_chunks = []
        while len(target_chunks) < count and offset < len(pool):
            c = pool[offset]; offset += 1
            if c["chunk_id"] not in used_ids:
                target_chunks.append(c)

        if len(target_chunks) < count:
            print(f"  Warning: only {len(target_chunks)} chunks for {domain}/{difficulty}, need {count}")

        generated = []
        for batch_start in range(0, len(target_chunks), batch_size):
            batch = target_chunks[batch_start: batch_start + batch_size][: count - len(generated)]
            if not batch:
                break

            print(f"  [{domain}][{difficulty}] {len(batch)} questions ({len(generated)}/{count})...")

            try:
                questions = _retry(lambda b=batch, d=difficulty: make_questions_batch(b, d))
            except Exception as e:
                print(f"    Skipped batch: {e}")
                questions = []

            valid_pairs = [
                (chunk, q.strip().strip('"'))
                for chunk, q in zip(batch, questions)
                if len((q or "").strip().split()) >= 6
            ]
            if len(valid_pairs) < len(batch):
                print(f"    Skipped {len(batch) - len(valid_pairs)} weak question(s).")

            ground_truths = [""] * len(valid_pairs)
            if valid_pairs:
                vchunks, vquestions = zip(*valid_pairs)
                try:
                    ground_truths = _retry(
                        lambda vc=list(vchunks), vq=list(vquestions): make_ground_truths_batch(vc, vq)
                    )
                except Exception as e:
                    print(f"    Ground truth failed: {e}. Using empty strings.")

            for (chunk, question), ground_truth in zip(valid_pairs, ground_truths):
                rows.append({
                    "id": f"q{q_index:03d}",
                    "query": question,
                    "ground_truth": (ground_truth or "").strip(),
                    "domain": domain,
                    "difficulty": difficulty,
                    "query_type": "standard",
                    "expected_paper_ids": [chunk["paper_id"]],
                    "expected_chunk_ids": [chunk["chunk_id"]],
                })
                used_ids.add(chunk["chunk_id"])
                q_index += 1
                generated.append(question)

            time.sleep(2)

    return rows


# ---------------------------------------------------------------------------
# Multi-source query generation
# ---------------------------------------------------------------------------

def make_multi_source_query(chunk_a: dict, chunk_b: dict) -> dict:
    prompt = f"""You are creating evaluation questions for a scientific RAG system.

Below are two scientific abstracts from DIFFERENT papers on related topics.

Paper A:
{chunk_a['text'][:1500]}

Paper B:
{chunk_b['text'][:1500]}

Write ONE synthesis question that:
- Requires information from BOTH papers to answer completely
- Asks about a comparison, connection, or combined implication
- Is self-contained — no mention of "passage", "text", or "paper"
- Is 10–25 words long

Then write a concise ground truth answer (2–4 sentences) grounded only in what both abstracts state.

Return ONLY valid JSON: {{"question": "...", "ground_truth": "..."}}"""

    response = llm.invoke(prompt)
    return _parse_json_response(response.content)


def generate_multi_source(buckets: dict, n_per_domain: int,
                          used_ids: set, q_index: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows = []
    for domain in list(DOMAIN_KEYWORDS):
        pool = [c for c in buckets.get(domain, []) if c["chunk_id"] not in used_ids]
        random.shuffle(pool)
        generated = 0
        idx = 0
        while generated < n_per_domain and idx + 1 < len(pool):
            a, b = pool[idx], pool[idx + 1]; idx += 2
            if a["paper_id"] == b["paper_id"]:
                continue
            try:
                result = _retry(lambda ca=a, cb=b: make_multi_source_query(ca, cb))
                q = (result.get("question") or "").strip()
                gt = (result.get("ground_truth") or "").strip()
                if len(q.split()) < 8 or not gt:
                    continue
                rows.append({
                    "id": f"q{q_index:03d}",
                    "query": q,
                    "ground_truth": gt,
                    "domain": domain,
                    "difficulty": "medium",
                    "query_type": "multi_source",
                    "expected_paper_ids": [a["paper_id"], b["paper_id"]],
                    "expected_chunk_ids": [a["chunk_id"], b["chunk_id"]],
                })
                used_ids.update([a["chunk_id"], b["chunk_id"]])
                q_index += 1
                generated += 1
                print(f"  [{domain}] multi-source: {q[:70]}...")
            except Exception as e:
                print(f"  [{domain}] multi-source skipped: {e}")
            time.sleep(2)
    return rows


# ---------------------------------------------------------------------------
# Unanswerable query generation
# ---------------------------------------------------------------------------

def make_unanswerable_query(chunk: dict) -> dict:
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


def generate_unanswerable(buckets: dict, n_total: int,
                          used_ids: set, q_index: int, seed: int) -> list[dict]:
    random.seed(seed + 1)
    domains = list(DOMAIN_KEYWORDS)
    pool = []
    for d in domains:
        candidates = [c for c in buckets.get(d, []) if c["chunk_id"] not in used_ids]
        random.shuffle(candidates)
        pool.extend((d, c) for c in candidates[:n_total * 2])
    random.shuffle(pool)

    rows = []
    for domain, chunk in pool:
        if len(rows) >= n_total:
            break
        try:
            result = _retry(lambda ch=chunk: make_unanswerable_query(ch))
            q = (result.get("question") or "").strip()
            if len(q.split()) < 8:
                continue
            rows.append({
                "id": f"q{q_index:03d}",
                "query": q,
                "ground_truth": "",
                "domain": domain,
                "difficulty": "unanswerable",
                "query_type": "unanswerable",
                "expected_paper_ids": [],
                "expected_chunk_ids": [],
            })
            used_ids.add(chunk["chunk_id"])
            q_index += 1
            print(f"  [{domain}] unanswerable: {q[:70]}...")
        except Exception as e:
            print(f"  [{domain}] unanswerable skipped: {e}")
        time.sleep(2)
    return rows


# ---------------------------------------------------------------------------
# Labels builder
# ---------------------------------------------------------------------------

def build_labels(queries: list[dict], labels_path: str) -> None:
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        for q in queries:
            row = {
                "query_id": q["id"],
                "expected_paper_ids": q.get("expected_paper_ids", []),
                "expected_chunk_ids": q.get("expected_chunk_ids", []),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"labels.jsonl written → {labels_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--output", default="data/eval/queries.jsonl")
    parser.add_argument("--labels", default="data/eval/labels.jsonl")
    parser.add_argument("--n-queries", type=int, default=30,
                        help="Total standard in-domain queries (split across 3 domains)")
    parser.add_argument("--multi", type=int, default=6,
                        help="Multi-source queries (2 per domain)")
    parser.add_argument("--unanswerable", type=int, default=5,
                        help="Unanswerable in-domain queries")
    parser.add_argument("--max-per-domain", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- load chunks -------------------------------------------------------
    print("Loading chunks for standard queries...")
    std_buckets = load_chunks_by_domain(args.chunks, max_per_domain=args.max_per_domain)
    for d, cs in std_buckets.items():
        random.shuffle(cs)
        print(f"  {d}: {len(cs)} usable chunks")

    print("\nLoading abstract chunks for multi-source / unanswerable...")
    abs_buckets = load_abstract_chunks_by_domain(args.chunks)
    for d, cs in abs_buckets.items():
        print(f"  {d}: {len(cs)} abstract chunks")

    used_chunk_ids: set[str] = set()
    all_rows: list[dict] = []
    q_index = 1

    # ---- standard queries --------------------------------------------------
    per_domain = args.n_queries // 3
    remainder  = args.n_queries % 3

    def difficulty_split(total: int) -> dict:
        each_side = total // 3
        medium    = total - 2 * each_side
        return {"easy": each_side, "medium": medium, "hard": each_side}

    for i, domain in enumerate(DOMAIN_KEYWORDS):
        target = per_domain + (1 if i < remainder else 0)
        pool   = std_buckets[domain][: target + 10]
        split  = difficulty_split(target)
        print(f"\n=== [standard] {domain}  target={target}  split={split} ===")
        rows = generate_standard(domain, pool, split, args.batch_size, q_index, used_chunk_ids)
        all_rows.extend(rows)
        q_index += len(rows)
        print(f"  → {len(rows)} queries")

    # ---- out-of-domain -----------------------------------------------------
    print("\n=== [out_of_domain] ===")
    for ood in OUT_OF_DOMAIN_QUERIES:
        all_rows.append({
            "id": f"q{q_index:03d}",
            "query": ood["query"],
            "ground_truth": "",
            "domain": "out_of_domain",
            "difficulty": ood["difficulty"],
            "query_type": "out_of_domain",
            "expected_paper_ids": [],
            "expected_chunk_ids": [],
        })
        print(f"  {ood['difficulty']:6s}: {ood['query'][:70]}")
        q_index += 1

    # ---- multi-source ------------------------------------------------------
    n_per_domain = max(1, args.multi // 3)
    print(f"\n=== [multi_source] {args.multi} queries ({n_per_domain}/domain) ===")
    multi_rows = generate_multi_source(abs_buckets, n_per_domain, used_chunk_ids, q_index, args.seed + 10)
    all_rows.extend(multi_rows)
    q_index += len(multi_rows)

    # ---- unanswerable ------------------------------------------------------
    print(f"\n=== [unanswerable] {args.unanswerable} queries ===")
    unans_rows = generate_unanswerable(abs_buckets, args.unanswerable, used_chunk_ids, q_index, args.seed + 20)
    all_rows.extend(unans_rows)

    # ---- write outputs -----------------------------------------------------
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    build_labels(all_rows, args.labels)

    # ---- summary -----------------------------------------------------------
    from collections import Counter
    types = Counter(r.get("query_type", "standard") for r in all_rows)
    diffs = Counter(r["difficulty"] for r in all_rows)
    print(f"\n{'='*55}")
    print(f"Saved {len(all_rows)} queries → {args.output}")
    print(f"\nQuery types : {dict(types)}")
    print(f"Difficulties: {dict(diffs)}")
    print()
    for domain in list(DOMAIN_KEYWORDS) + ["out_of_domain"]:
        dr = [r for r in all_rows if r["domain"] == domain]
        if not dr:
            continue
        dc = Counter(r["difficulty"] for r in dr)
        print(f"  {domain:22s}  " + "  ".join(f"{k}={v}" for k, v in dc.items()))


if __name__ == "__main__":
    main()
