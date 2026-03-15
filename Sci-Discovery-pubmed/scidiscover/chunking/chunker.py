import json
from pathlib import Path

import yaml


def load_config():
    with open("configs/demo.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize(text):
    return text.split()


def detokenize(tokens):
    return " ".join(tokens)


def chunk_text(text, chunk_size, overlap):
    tokens = tokenize(text)

    start = 0
    offset = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        yield offset, detokenize(chunk_tokens), len(chunk_tokens)

        if end >= len(tokens):
            break

        start += chunk_size - overlap
        offset += 1


def run():
    config = load_config()

    input_file = config["chunking"]["input_path"]
    output_file = config["chunking"]["output_path"]
    chunk_size = config["chunking"]["chunk_size"]
    overlap = config["chunking"]["overlap"]

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            paper = json.loads(line)

            paper_id = paper["paper_id"]

            sections = {
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
            }

            for section, text in sections.items():
                if not text:
                    continue

                for offset, chunk_text_value, tok_len in chunk_text(text, chunk_size, overlap):
                    chunk_id = f"{paper_id}_{section}_{offset:03d}"

                    chunk = {
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "section": section,
                        "text": chunk_text_value,
                        "token_len": tok_len,
                    }

                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total_chunks += 1

    print(f"Chunks written to {output_file}")
    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    run()