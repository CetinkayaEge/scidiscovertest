import argparse
import yaml
import subprocess
import sys


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    pmc = config["sources"]["pmc"]

    if not pmc["enabled"]:
        print("PMC source disabled")
        return

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

    print("Running ingestion pipeline...\n")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()