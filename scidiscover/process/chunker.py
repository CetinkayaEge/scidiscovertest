import argparse
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
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
    args = parse_args()
    config = load_config(args.config)

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
                "TITLE": paper.get("title", ""),
                "ABSTRACT": paper.get("abstract", ""),
                "BODY_UNK": paper.get("full_text", paper.get("body", "")),
            }

            for section, text in sections.items():
                if not text:
                    continue

                for offset, chunk_text_value, tok_len in chunk_text(text, chunk_size, overlap):
                    chunk_id = f"{paper_id}||{section}||{offset:03d}"

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
