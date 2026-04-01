import argparse
import yaml
from dotenv import load_dotenv

load_dotenv()

from scidiscover.agents.retriever import Retriever
from scidiscover.agents.retriever_agent import RetrieverAgent
from scidiscover.agents.summarizer import SummarizerAgent
from scidiscover.agents.synthesizer import SynthesizerAgent
from utils.llm_client import configure_llm


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

    # ---- INIT BASE RETRIEVER ----
    retriever = Retriever(
        chunks_input=config["retrieval"]["chunks_input"],
        index_output=config["retrieval"]["index_output"],
        chunk_ids_output=config["retrieval"]["chunk_ids_output"],
        traces_output=config["retrieval"]["traces_output"],
        model_name=config["retrieval"]["model_name"],
        papers_input=config["retrieval"]["papers_input"],
    )

    # ---- INIT AGENTS ----
    configure_llm(config["llm"]["model"], config["llm"]["max_tokens"])
    retriever_agent = RetrieverAgent(retriever)
    summarizer_agent = SummarizerAgent(
        prompt_path=config["summarizer"]["prompt_path"],
        traces_output=config["summarizer"]["traces_output"],
        output_path=config["summarizer"]["output_path"],
    )
    synthesizer_agent = SynthesizerAgent(
        prompt_path=config["synthesizer"]["prompt_path"],
        traces_output=config["synthesizer"]["traces_output"],
        output_path=config["synthesizer"]["output_path"],
    )

    # ---- RETRIEVE (AGENT) ----
    top_k = args.top_k or config["retrieval"]["top_k"]
    evidence_pack = retriever_agent.run(args.query, k=top_k)

    # ---- DEBUG: RAW RETRIEVAL OUTPUT ----
    print("\n===== RAW RETRIEVAL RESULTS =====\n")

    for i, r in enumerate(evidence_pack["chunks"], start=1):
        print("=" * 80)
        print(f"Rank: {i}")
        print(f"Score: {r['score']:.4f}")
        print(f"Paper ID: {r['paper_id']}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Section: {r['section']}")

        # ---- CONTENT ----
        text = r["text"].strip()

        if r["section"] == "TITLE":
            print(f"\n Title:\n{text}")
        elif r["section"] == "ABSTRACT":
            print(f"\n Abstract:\n{text[:300]}...")
        else:
            print(f"\n Content:\n{text[:300]}...")

    print("=" * 80)
    # ---- SUMMARIZER (AGENT) ----
    print("\n\n===== PER-PAPER SUMMARIES =====\n")

    summaries = summarizer_agent.run(evidence_pack)

    for item in summaries:
        print("=" * 80)
        print(f"Paper: {item['paper_id']}")
        print(item["summary_text"])
        citations = [e["chunk_id"] for e in item["evidence"]]
        print("Citations:", citations)
        print()

    # ---- SYNTHESIZER (AGENT) ----
    print("\n\n===== SYNTHESIS =====\n")

    synthesis = synthesizer_agent.run({
        "query": args.query,
        "paper_summaries": summaries,
    })

    print("Draft Answer:")
    print(synthesis["draft_answer"])
    print()
    print("Key Claims:")
    for i, kc in enumerate(synthesis["key_claims"], start=1):
        print(f"  {i}. {kc['claim']}")
        print(f"     Citations: {kc['citation_ids']}")
    print()
    print("Limitations & Uncertainty:")
    for lim in synthesis["limitations_and_uncertainty"]:
        print(f"  - {lim}")
    print()
    print(f"Evidence chunks used: {len(synthesis['evidence'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()