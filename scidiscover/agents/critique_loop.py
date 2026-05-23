from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scidiscover.agents.synthesizer import SynthesizerAgent
from scidiscover.agents.verifier import VerifierAgent


class CritiqueLoopAgent:
    """Verifier→Synthesizer iterative refinement loop.

    Runs synthesis, then checks support_rate. If it falls below the verifier's
    threshold, feeds the failing claims back to the synthesizer as a critique
    and re-verifies. Repeats up to max_iterations times, then writes the final
    verification result to disk via verifier.run().
    """

    def __init__(
        self,
        synthesizer: SynthesizerAgent,
        verifier: VerifierAgent,
        max_iterations: int = 2,
        traces_output: str = "logs/critique_loop_traces.jsonl",
    ) -> None:
        self.synthesizer = synthesizer
        self.verifier = verifier
        self.max_iterations = max_iterations
        self.traces_output = traces_output

    def run(self, summary_pack: dict, evidence_pack: dict) -> dict:
        """
        Input:
          summary_pack  — {"query": str, "paper_summaries": [...]}
          evidence_pack — {"query": str, "chunks": [...]}

        Output:
          {
            "synthesis":     final synthesis dict (same shape as SynthesizerAgent.run()),
            "verified":      final verification dict (same shape as VerifierAgent.run()),
            "n_iterations":  number of critique iterations that were actually run,
          }
        """
        query = summary_pack["query"]

        # Initial synthesis
        synthesis = self.synthesizer.run(summary_pack)

        iterations_log: list[dict] = []
        final_synthesis = synthesis

        for iteration in range(self.max_iterations):
            # Early exit if synthesis itself abstained or errored
            draft = synthesis.get("draft_answer", "")
            if not draft or draft.startswith(("ABSTAIN", "[LLM ERROR", "[PARSE ERROR")):
                break

            # Internal verification — no disk I/O
            verification = self.verifier._compute_verification(synthesis, evidence_pack)
            vs = verification["verification_summary"]

            iterations_log.append({
                "iteration": iteration,
                "support_rate": vs["support_rate"],
                "citation_coverage": vs["citation_coverage"],
                "n_supported": vs["supported"],
                "n_unsupported": vs["unsupported"],
                "n_conflict": vs["conflict"],
                "n_claims": vs["total_claims"],
            })

            # Support is good enough — stop early
            if not verification["needs_revision"]:
                final_synthesis = synthesis
                break

            failing = verification["failing_claims"]
            if not failing:
                final_synthesis = synthesis
                break

            # Revise synthesis with critique
            revised = self.synthesizer.revise(
                summary_pack=summary_pack,
                failing_claims=failing,
                previous_draft=draft,
            )

            # If revision degraded, keep previous synthesis and stop
            revised_draft = revised.get("draft_answer", "")
            if not revised_draft or revised_draft.startswith(("ABSTAIN", "[LLM ERROR", "[PARSE ERROR")):
                final_synthesis = synthesis
                break

            synthesis = revised
            final_synthesis = revised

        # Final verification with disk I/O
        final_verified = self.verifier.run({
            "synthesis": final_synthesis,
            "evidence_pack": evidence_pack,
        })

        self._log_trace(query, iterations_log, final_verified)

        return {
            "synthesis": final_synthesis,
            "verified": final_verified,
            "n_iterations": len(iterations_log),
        }

    def _log_trace(self, query: str, iterations_log: list[dict], final_verified: dict) -> None:
        vs = final_verified.get("verification_summary", {})
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "n_iterations": len(iterations_log),
            "iterations": iterations_log,
            "final_support_rate": vs.get("support_rate"),
            "final_citation_coverage": vs.get("citation_coverage"),
            "final_answer_preview": final_verified.get("final_answer", "")[:200],
        }
        Path(self.traces_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
