import argparse
import yaml
from dotenv import load_dotenv

load_dotenv()

from scidiscover.agents.retriever import Retriever
from scidiscover.agents.retriever_agent import RetrieverAgent
from scidiscover.agents.summarizer import SummarizerAgent
from scidiscover.agents.synthesizer import SynthesizerAgent
from scidiscover.agents.reranker import RerankerAgent
from scidiscover.agents.verifier import VerifierAgent
from scidiscover.agents.critique_loop import CritiqueLoopAgent
from utils.llm_client import configure_llm


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--critique-loop", action="store_true",
                        help="Use CritiqueLoopAgent instead of plain Verifier")
    parser.add_argument("--force-critique", action="store_true",
                        help="Set min_support_rate=0.99 to force the loop to always trigger")
    args = parser.parse_args()

    config = load_config(args.config)

    configure_llm(config["llm"]["model"], config["llm"]["max_tokens"])

    retriever = Retriever(
        chunks_input=config["retrieval"]["chunks_input"],
        index_output=config["retrieval"]["index_output"],
        chunk_ids_output=config["retrieval"]["chunk_ids_output"],
        traces_output=config["retrieval"]["traces_output"],
        model_name=config["retrieval"]["model_name"],
        papers_input=config["retrieval"]["papers_input"],
    )

    retriever_agent = RetrieverAgent(retriever)
    summarizer_agent = SummarizerAgent(
        prompt_path=config["summarizer"]["prompt_path"],
        traces_output=config["summarizer"]["traces_output"],
        output_path=config["summarizer"]["output_path"],
    )
    cl_cfg = config.get("critique_loop", {})
    synthesizer_agent = SynthesizerAgent(
        prompt_path=config["synthesizer"]["prompt_path"],
        traces_output=config["synthesizer"]["traces_output"],
        output_path=config["synthesizer"]["output_path"],
        critique_prompt_path=cl_cfg.get("critique_prompt_path"),
    )

    verifier_cfg = config["verifier"]
    min_support_rate = 0.99 if args.force_critique else verifier_cfg.get("min_support_rate", 0.50)
    verifier_agent = VerifierAgent(
        prompt_path=verifier_cfg["prompt_path"],
        traces_output=verifier_cfg["traces_output"],
        verification_output=verifier_cfg["verification_output"],
        answers_output=verifier_cfg["answers_output"],
        min_citation_coverage=verifier_cfg.get("min_citation_coverage", 0.95),
        min_support_rate=min_support_rate,
    )

    reranker_cfg = config.get("reranker", {})
    reranker_agent = None
    if reranker_cfg.get("enabled", False):
        reranker_agent = RerankerAgent(
            model_name=reranker_cfg["model_name"],
            top_k=reranker_cfg.get("top_k", 10),
            min_score=reranker_cfg.get("min_score", 0.0),
        )

    top_k = args.top_k or config["retrieval"]["top_k"]
    evidence_pack = retriever_agent.run(args.query, k=top_k)

    if reranker_agent is not None:
        evidence_pack = reranker_agent.run(evidence_pack)

    # ---- RAW RETRIEVAL ----
    print("\n===== RAW RETRIEVAL RESULTS =====\n")
    for i, r in enumerate(evidence_pack["chunks"], start=1):
        print("=" * 80)
        print(f"Rank: {i}  Score: {r['score']:.4f}  Paper: {r['paper_id']}")
        print(f"Chunk: {r['chunk_id']}  Section: {r['section']}")
        print(r["text"].strip()[:300] + "...")
    print("=" * 80)

    # ---- SUMMARIES ----
    print("\n\n===== PER-PAPER SUMMARIES =====\n")
    summaries = summarizer_agent.run(evidence_pack)
    for item in summaries:
        print("=" * 80)
        print(f"Paper: {item['paper_id']}")
        print(item["summary_text"])
        print("Citations:", [e["chunk_id"] for e in item["evidence"]])

    summary_pack = {"query": args.query, "paper_summaries": summaries}

    # ---- CRITIQUE LOOP or plain Synthesis+Verification ----
    if args.critique_loop:
        if args.force_critique:
            print(f"\n[NOTE] --force-critique active: min_support_rate set to 0.99 "
                  f"so the loop always triggers regardless of actual support.\n")

        critique_loop_agent = CritiqueLoopAgent(
            synthesizer=synthesizer_agent,
            verifier=verifier_agent,
            max_iterations=cl_cfg.get("max_iterations", 2),
            traces_output=cl_cfg.get("traces_output", "logs/critique_loop_traces.jsonl"),
        )

        print("\n\n===== CRITIQUE LOOP =====\n")
        loop_result = critique_loop_agent.run(summary_pack, evidence_pack)
        synthesis = loop_result["synthesis"]
        verified = loop_result["verified"]
        n_iters = loop_result["n_iterations"]

        print(f"Critique iterations run: {n_iters}")
        _print_synthesis(synthesis)
        _print_verification(verified)

    else:
        print("\n\n===== SYNTHESIS =====\n")
        synthesis = synthesizer_agent.run(summary_pack)
        _print_synthesis(synthesis)

        print("\n\n===== VERIFICATION =====\n")
        verified = verifier_agent.run({
            "synthesis": synthesis,
            "evidence_pack": evidence_pack,
        })
        _print_verification(verified)


def _print_synthesis(synthesis: dict) -> None:
    print("Draft Answer:")
    print(synthesis["draft_answer"])
    print()
    print("Key Claims:")
    for i, kc in enumerate(synthesis["key_claims"], start=1):
        print(f"  {i}. {kc['claim']}")
        print(f"     Citations: {kc['citation_ids']}")
    print()
    print("Limitations:")
    for lim in synthesis["limitations_and_uncertainty"]:
        print(f"  - {lim}")
    print(f"\nEvidence chunks used: {len(synthesis['evidence'])}")
    print("=" * 80)


def _print_verification(verified: dict) -> None:
    print("Final Answer:")
    print(verified["final_answer"])
    print()
    print("Verified Claims:")
    for i, kc in enumerate(verified["key_claims"], start=1):
        print(f"  {i}. [{kc['status']}] {kc['claim']}")
        print(f"     Notes: {kc['notes']}")
        print(f"     Citations: {kc['citation_ids']}")
    print()
    vs = verified["verification_summary"]
    print(f"Citation Coverage : {vs['citation_coverage']:.2f}")
    print(f"Support Rate      : {vs['support_rate']:.2f}")
    print(f"SUPPORTED={vs['supported']}  UNSUPPORTED={vs['unsupported']}  "
          f"CONFLICT={vs['conflict']}  UNKNOWN={vs['unknown']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
