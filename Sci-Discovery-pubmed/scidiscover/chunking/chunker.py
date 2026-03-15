import json
from pathlib import Path

INPUT_FILE = "data/raw/papers.jsonl"
OUTPUT_FILE = "data/processed/chunks.jsonl"

CHUNK_SIZE = 200
OVERLAP = 40


def tokenize(text):
    return text.split()


def detokenize(tokens):
    return " ".join(tokens)


def chunk_text(text):

    tokens = tokenize(text)

    start = 0
    offset = 0

    while start < len(tokens):

        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        yield offset, detokenize(chunk_tokens), len(chunk_tokens)

        start += CHUNK_SIZE - OVERLAP
        offset += 1


def run():

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE) as f, open(OUTPUT_FILE, "w") as out:

        for line in f:

            paper = json.loads(line)

            paper_id = paper["paper_id"]

            sections = {
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", "")
            }

            for section, text in sections.items():

                if not text:
                    continue

                for offset, chunk_text_value, tok_len in chunk_text(text):

                    chunk_id = f"{paper_id}_{section}_{offset:03d}"

                    chunk = {
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "section": section,
                        "text": chunk_text_value,
                        "token_len": tok_len
                    }

                    out.write(json.dumps(chunk) + "\n")


if __name__ == "__main__":
    run()