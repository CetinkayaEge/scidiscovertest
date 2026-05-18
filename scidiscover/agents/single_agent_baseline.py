"""Baseline-1: single LLM call over all retrieved chunks.

Replaces the entire Summarizer → Synthesizer → Verifier chain with
one LLM call. Evidence chunks are concatenated and fed directly.
This is the most aggressive simplification of the multi-agent pipeline.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from utils.llm_client import call_llm
from utils.models import SingleAgentLLMOutput

_ABSTAIN_MSG = "Insufficient evidence in retrieved corpus to answer reliably."


class SingleAgentBaseline:
    def __init__(
        self,
        prompt_path: str,
        traces_output: str = "logs/single_agent_baseline_traces.jsonl",
    ) -> None:
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        self.traces_output = traces_output

    def build_context(self, chunks: list) -> str:
        parts = []
        for ch in chunks:
            parts.append(f"[{ch['chunk_id']}]\n{ch['text'].strip()}")
        return "\n\n".join(parts)

    def _parse_response(self, response: str) -> SingleAgentLLMOutput | None:
        """Parse and validate the LLM's JSON response using Pydantic.

        Strips markdown fences if present, then validates with SingleAgentLLMOutput.
        Returns None on any parse or validation failure.

        Compared to the previous json.loads approach this also validates that
        key_claims is actually a list and that citation_ids are lists, and
        silently drops any extra top-level keys the LLM adds.
        """
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return SingleAgentLLMOutput.model_validate_json(cleaned)
        except ValidationError:
            return None

    def run(self, evidence_pack: dict) -> dict:
        """
        Input:  evidence_pack = {query, chunks: [...]}

        Output: {
            query, final_answer, key_claims, limitations_and_uncertainty, abstained
        }

        key_claims follow the same schema as SynthesizerAgent:
            [{claim: str, citation_ids: [str]}]
        so that compute_synthesizer_hallucination_rate() works unchanged.
        """
        query = evidence_pack["query"]
        chunks = evidence_pack["chunks"]

        if not chunks:
            self._log_trace(query, True, [])
            return {
                "query": query,
                "final_answer": _ABSTAIN_MSG,
                "key_claims": [],
                "limitations_and_uncertainty": ["No evidence retrieved."],
                "abstained": True,
            }

        context = self.build_context(chunks)
        prompt = self.prompt_template.format(query=query, context=context)

        try:
            response = call_llm(system=prompt, user="", json_mode=True)
        except Exception as e:
            try:
                err_msg = str(e)
            except Exception:
                err_msg = type(e).__name__
            self._log_trace(query, True, [])
            return {
                "query": query,
                "final_answer": _ABSTAIN_MSG,
                "key_claims": [],
                "limitations_and_uncertainty": ["LLM error: " + err_msg],
                "abstained": True,
            }

        if response.strip().startswith("ABSTAIN:"):
            self._log_trace(query, True, [])
            return {
                "query": query,
                "final_answer": _ABSTAIN_MSG,
                "key_claims": [],
                "limitations_and_uncertainty": ["Insufficient evidence to answer the query."],
                "abstained": True,
            }

        parsed = self._parse_response(response)
        if parsed is None:
            self._log_trace(query, True, [])
            return {
                "query": query,
                "final_answer": _ABSTAIN_MSG,
                "key_claims": [],
                "limitations_and_uncertainty": ["Parse error in LLM response."],
                "abstained": True,
            }

        # Convert Pydantic models back to plain dicts so the rest of the
        # eval pipeline continues to work with the existing dict-based interface.
        final_answer = parsed.final_answer
        key_claims = [c.model_dump() for c in parsed.key_claims]
        limitations = parsed.limitations_and_uncertainty

        abstained = (
            not final_answer
            or final_answer.strip().startswith("ABSTAIN")
            or final_answer.strip().startswith("[")
        )
        if abstained:
            final_answer = _ABSTAIN_MSG

        self._log_trace(query, abstained, key_claims)
        return {
            "query": query,
            "final_answer": final_answer,
            "key_claims": key_claims,
            "limitations_and_uncertainty": limitations,
            "abstained": abstained,
        }

    def _log_trace(self, query: str, abstained: bool, key_claims: list) -> None:
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "abstained": abstained,
            "n_claims": len(key_claims),
        }
        Path(self.traces_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
