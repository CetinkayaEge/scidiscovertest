from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from utils.llm_client import call_llm
from utils.schemas import validate_verification_pack

_ABSTAIN_MSG = "Insufficient evidence in retrieved corpus to answer reliably."


class VerifierAgent:
    def __init__(
        self,
        prompt_path: str,
        traces_output: str = "logs/verifier_traces.jsonl",
        verification_output: str = "outputs/verification.jsonl",
        answers_output: str = "outputs/answers.jsonl",
        min_citation_coverage: float = 0.95,
        min_support_rate: float = 0.50,
    ) -> None:
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        self.traces_output = traces_output
        self.verification_output = verification_output
        self.answers_output = answers_output
        self.min_citation_coverage = min_citation_coverage
        self.min_support_rate = min_support_rate

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, pack: dict) -> dict:
        """
        Input:
          {
            "synthesis":     { query, draft_answer, key_claims, evidence, limitations_and_uncertainty }
            "evidence_pack": { query, chunks: [{chunk_id, text, paper_id, section, score, ...}] }
          }

        Output:
          { query, final_answer, key_claims (with status/notes), evidence,
            limitations_and_uncertainty, verification_summary }
        """
        synthesis = pack["synthesis"]
        evidence_pack = pack["evidence_pack"]

        query = synthesis["query"]
        draft_answer = synthesis.get("draft_answer", "")
        key_claims = synthesis.get("key_claims", [])
        evidence = synthesis.get("evidence", [])
        limitations = synthesis.get("limitations_and_uncertainty", [])

        run_id = str(uuid.uuid4())[:8]

        # Step 1 — No-evidence fallback
        if not evidence_pack.get("chunks"):
            result = self._abstain_result(query, key_claims, evidence, limitations)
            self._write_verification_jsonl(run_id, query, result["key_claims"])
            self._write_answers_jsonl(run_id, result)
            self._log_trace(run_id, query, result)
            return result

        # Step 2 — Build chunk text lookup (normalise chunk IDs)
        chunk_lookup = {
            self._normalize_id(ch["chunk_id"]): ch["text"]
            for ch in evidence_pack["chunks"]
        }

        # Step 3 — Pre-screen claims with no citations (rule-based)
        prescreened: list[dict] = []
        llm_claims: list[tuple[int, dict]] = []  # (original_index, claim)

        for idx, claim in enumerate(key_claims):
            citation_ids = claim.get("citation_ids", [])
            if not citation_ids:
                prescreened.append({
                    "original_idx": idx,
                    "claim": claim["claim"],
                    "citation_ids": [],
                    "status": "UNSUPPORTED",
                    "notes": "No citations provided.",
                })
            else:
                llm_claims.append((idx, claim))

        # Step 4 — LLM verification for claims that have citations
        llm_verified: dict[int, dict] = {}
        if llm_claims:
            llm_verified = self._verify_with_llm(
                query, draft_answer, llm_claims, chunk_lookup
            )

        # Step 5 — Merge results in original order
        verified_claims = []
        prescreened_by_idx = {p["original_idx"]: p for p in prescreened}

        for idx, original_claim in enumerate(key_claims):
            if idx in prescreened_by_idx:
                p = prescreened_by_idx[idx]
                verified_claims.append({
                    "claim": p["claim"],
                    "citation_ids": p["citation_ids"],
                    "status": p["status"],
                    "notes": p["notes"],
                })
            elif idx in llm_verified:
                v = llm_verified[idx]
                verified_claims.append({
                    "claim": original_claim["claim"],
                    "citation_ids": original_claim.get("citation_ids", []),
                    "status": v["status"],
                    "notes": v["notes"],
                })
            else:
                verified_claims.append({
                    "claim": original_claim["claim"],
                    "citation_ids": original_claim.get("citation_ids", []),
                    "status": "UNKNOWN",
                    "notes": "Verification did not return a result for this claim.",
                })

        # Step 6 — Compute summary
        summary = self._compute_summary(verified_claims)

        # Step 7 — Decide final_answer
        if (summary["citation_coverage"] >= self.min_citation_coverage
                and summary["support_rate"] >= self.min_support_rate):
            final_answer = draft_answer
        else:
            final_answer = _ABSTAIN_MSG

        result = {
            "query": query,
            "final_answer": final_answer,
            "key_claims": verified_claims,
            "evidence": evidence,
            "limitations_and_uncertainty": limitations,
            "verification_summary": summary,
        }

        validate_verification_pack(result)

        self._write_verification_jsonl(run_id, query, verified_claims)
        self._write_answers_jsonl(run_id, result)
        self._log_trace(run_id, query, result)

        return result

    # ------------------------------------------------------------------
    # LLM verification
    # ------------------------------------------------------------------

    def _verify_with_llm(
        self,
        query: str,
        draft_answer: str,
        llm_claims: list[tuple[int, dict]],
        chunk_lookup: dict[str, str],
    ) -> dict[int, dict]:
        claims_and_evidence = self._build_claims_and_evidence(llm_claims, chunk_lookup)

        prompt = self.prompt_template.format(
            query=query,
            draft_answer=draft_answer,
            claims_and_evidence=claims_and_evidence,
        )

        try:
            response = call_llm(system=prompt, user="", json_mode=True)
        except Exception as e:
            return {
                idx: {"status": "UNKNOWN", "notes": f"LLM error: {e}"}
                for idx, _ in llm_claims
            }

        parsed = self._parse_llm_response(response)
        if not isinstance(parsed, dict):
            return {
                idx: {"status": "UNKNOWN", "notes": "LLM parse error."}
                for idx, _ in llm_claims
            }

        # Map LLM output back to original claim indices
        result: dict[int, dict] = {}
        verified_list = parsed.get("verified_claims", [])

        for position, (original_idx, _) in enumerate(llm_claims):
            if position < len(verified_list):
                vc = verified_list[position]
                status = vc.get("status", "UNKNOWN").upper()
                if status not in {"SUPPORTED", "UNSUPPORTED", "CONFLICT", "UNKNOWN"}:
                    status = "UNKNOWN"
                result[original_idx] = {
                    "status": status,
                    "notes": vc.get("notes", ""),
                }
            else:
                result[original_idx] = {
                    "status": "UNKNOWN",
                    "notes": "LLM did not return a result for this claim.",
                }

        return result

    def _build_claims_and_evidence(
        self,
        llm_claims: list[tuple[int, dict]],
        chunk_lookup: dict[str, str],
    ) -> str:
        parts = []
        for position, (_, claim) in enumerate(llm_claims):
            citation_ids = claim.get("citation_ids", [])
            chunk_blocks = []
            for cid in citation_ids:
                norm_cid = self._normalize_id(cid)
                text = chunk_lookup.get(norm_cid, "[Chunk not found in evidence set]")
                chunk_blocks.append(f"  [{cid}]\n  {text}")

            chunks_text = "\n".join(chunk_blocks) if chunk_blocks else "  (no citations)"
            parts.append(
                f"Claim {position} (0-based index):\n"
                f"{claim['claim']}\n"
                f"Cited chunks:\n{chunks_text}"
            )

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_id(cid: str) -> str:
        """Fix single-pipe separators → double-pipe (LLM output bug mitigation)."""
        # Replace single | that are not already part of ||
        return re.sub(r'(?<!\|)\|(?!\|)', '||', cid)

    @staticmethod
    def _parse_llm_response(response: str) -> dict | None:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _compute_summary(verified_claims: list[dict]) -> dict:
        total = len(verified_claims)
        if total == 0:
            return {
                "citation_coverage": 0.0,
                "support_rate": 0.0,
                "total_claims": 0,
                "supported": 0,
                "unsupported": 0,
                "conflict": 0,
                "unknown": 0,
            }

        with_citations = sum(1 for c in verified_claims if c.get("citation_ids"))
        supported = sum(1 for c in verified_claims if c["status"] == "SUPPORTED")
        unsupported = sum(1 for c in verified_claims if c["status"] == "UNSUPPORTED")
        conflict = sum(1 for c in verified_claims if c["status"] == "CONFLICT")
        unknown = sum(1 for c in verified_claims if c["status"] == "UNKNOWN")

        decided = supported + unsupported + conflict
        return {
            "citation_coverage": with_citations / total,
            "support_rate": supported / decided if decided > 0 else 0.0,
            "total_claims": total,
            "supported": supported,
            "unsupported": unsupported,
            "conflict": conflict,
            "unknown": unknown,
        }

    def _abstain_result(
        self,
        query: str,
        key_claims: list[dict],
        evidence: list,
        limitations: list,
    ) -> dict:
        verified_claims = [
            {
                "claim": c["claim"],
                "citation_ids": c.get("citation_ids", []),
                "status": "UNSUPPORTED",
                "notes": "No evidence was retrieved.",
            }
            for c in key_claims
        ]
        return {
            "query": query,
            "final_answer": _ABSTAIN_MSG,
            "key_claims": verified_claims,
            "evidence": evidence,
            "limitations_and_uncertainty": limitations + ["No evidence retrieved; all claims unverified."],
            "verification_summary": self._compute_summary(verified_claims),
        }

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_verification_jsonl(
        self, run_id: str, query: str, verified_claims: list[dict]
    ) -> None:
        Path(self.verification_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.verification_output, "a", encoding="utf-8") as f:
            for i, vc in enumerate(verified_claims):
                record = {
                    "run_id": run_id,
                    "query": query,
                    "claim_id": i,
                    "claim": vc["claim"],
                    "status": vc["status"],
                    "cited_chunks": vc.get("citation_ids", []),
                    "notes": vc.get("notes", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_answers_jsonl(self, run_id: str, result: dict) -> None:
        Path(self.answers_output).parent.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": run_id,
            "query": result["query"],
            "final_answer": result["final_answer"],
            "key_claims": result["key_claims"],
            "evidence": result["evidence"],
            "limitations_and_uncertainty": result["limitations_and_uncertainty"],
        }
        with open(self.answers_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_trace(self, run_id: str, query: str, result: dict) -> None:
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "query": query,
            "verification_summary": result["verification_summary"],
            "final_answer_preview": result["final_answer"][:200],
            "claims": [
                {"claim_id": i, "status": vc["status"], "notes": vc["notes"]}
                for i, vc in enumerate(result["key_claims"])
            ],
        }
        Path(self.traces_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
