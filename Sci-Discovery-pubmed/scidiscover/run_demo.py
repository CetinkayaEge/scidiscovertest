import argparse
import subprocess
import sys

import yaml

from scidiscover.retrieval.build_index import build_faiss_index


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-chunking", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    pmc = config["sources"]["pmc"]

    if not args.skip_ingestion:
        if not pmc["enabled"]:
            print("PMC source disabled")
        else:
            cmd = [
                sys.executable,
                "-m",
                "scidiscover.ingestion.pmc_ingest",
                "--from-date",
                pmc["from_date"],
                "--max-papers",
                str(pmc["max_papers"]),
                "--raw-output",
                config["corpus"]["raw_output"],
                "--manifest-output",
                config["corpus"]["manifest_output"],
            ]

            if pmc.get("until_date"):
                cmd += ["--until-date", pmc["until_date"]]

            if pmc.get("require_pdf"):
                cmd.append("--require-pdf")

            if pmc.get("skip_empty_abstract"):
                cmd.append("--skip-empty-abstract")

            print("STEP 1: Running ingestion pipeline...\n")
            subprocess.run(cmd, check=True)

    if not args.skip_chunking:
        print("\nSTEP 2: Running chunking pipeline...\n")
        subprocess.run(
            [sys.executable, "-m", "scidiscover.chunking.chunker"],
            check=True,
        )

    if not args.skip_index:
        print("\nSTEP 3: Building embeddings and FAISS index...\n")
        build_faiss_index(
            chunks_input=config["retrieval"]["chunks_input"],
            embeddings_output=config["retrieval"]["embeddings_output"],
            index_output=config["retrieval"]["index_output"],
            chunk_ids_output=config["retrieval"]["chunk_ids_output"],
            model_name=config["retrieval"]["model_name"],
        )

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()