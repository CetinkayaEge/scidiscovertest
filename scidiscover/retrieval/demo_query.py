import argparse
import yaml

from scidiscover.retrieval.retrieve import Retriever


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    retriever = Retriever(
        chunks_input=config["retrieval"]["chunks_input"],
        index_output=config["retrieval"]["index_output"],
        chunk_ids_output=config["retrieval"]["chunk_ids_output"],
        traces_output=config["retrieval"]["traces_output"],
        model_name=config["retrieval"]["model_name"],
    )

    top_k = args.top_k or config["retrieval"]["top_k"]
    results = retriever.retrieve(args.query, k=top_k)

    for i, r in enumerate(results, start=1):
        print(f"\nRank {i}")
        print(f"Score: {r['score']:.4f}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Paper ID: {r['paper_id']}")
        print(f"Section: {r['section']}")
        print(f"Text: {r['text'][:500]}")
        print("-" * 80)


if __name__ == "__main__":
    main()