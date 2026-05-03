import json
import re
from datetime import datetime, timezone
from pathlib import Path


from utils.llm_client import call_llm
from utils.schemas import validate_synthesis_pack


def _is_bad_summary(text: str) -> bool:
    """Return True for summaries that should be excluded from synthesis."""
    t = text.strip().lower()
    return t.startswith("insufficient evidence") or t.startswith("[llm error")


class SynthesizerAgent:
    def __init__(
        self,
        prompt_path: str,
        traces_output: str = "logs/synthesizer_traces.jsonl",
        output_path: str = "outputs/synthesis_result.json",
    ):
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        self.traces_output = traces_output
        self.output_path = output_path

    def build_summaries_context(self, paper_summaries):
        """Format paper summaries into a text block for the LLM prompt."""
        parts = []
        for ps in paper_summaries:
            paper_id = ps["paper_id"]
            summary_text = ps["summary_text"]
            evidence_ids = ", ".join(
                e["chunk_id"] for e in ps.get("evidence", [])
            )
            parts.append(
                f"Paper: {paper_id}\n"
                f"Evidence chunks: {evidence_ids}\n"
                f"Summary:\n{summary_text}"
            )
        return "\n\n---\n\n".join(parts)

    def parse_llm_response(self, response):
        """Parse the LLM JSON response into a dict. Returns None on failure."""
        cleaned = response.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    def extract_cited_ids(self, key_claims):
        """Collect all cited chunk_ids from key_claims citation_ids lists."""
        cited = set()
        for claim in key_claims:
            for cid in claim.get("citation_ids", []):
                cited.add(cid)
        return cited

    def build_evidence_list(self, paper_summaries, cited_ids):
        """Build the evidence lookup table in Python from summarizer output.

        Citations are chunk_ids (format: paper_id||SECTION||offset).
        paper_id is always the first ||-separated segment of chunk_id.
        Chunks not found in the lookup (LLM hallucinations) are skipped.
        """
        # Build lookup keyed by chunk_id — this is the sole citation key
        chunk_lookup = {}
        for ps in paper_summaries:
            for ev in ps.get("evidence", []):
                chunk_id = ev["chunk_id"]
                chunk_lookup[chunk_id] = {
                    "paper_id": chunk_id.split("||")[0],
                    "chunk_id": chunk_id,
                    "score": ev.get("score", 0.0),
                    "section": ev.get("section", ""),
                    "text_preview": ev.get("text_preview", ""),
                    "url": ps.get("url", ""),
                }

        evidence = []
        for cid in cited_ids:
            if cid in chunk_lookup:
                evidence.append(chunk_lookup[cid])
            # If not found, the LLM cited something outside our retrieval set.
            # Skip it — the Verifier will mark those claims UNKNOWN via its
            # own chunk_lookup built from the original evidence_pack.

        return evidence

    def synthesize(self, query, paper_summaries):
        """Call the LLM to synthesize paper summaries into a draft answer."""
        summaries_context = self.build_summaries_context(paper_summaries)

        prompt = self.prompt_template.format(
            query=query,
            summaries_context=summaries_context,
        )

        try:
            response = call_llm(system=prompt, user="", json_mode=True)
        except Exception as e:
            try:
                err_msg = str(e)
            except Exception:
                err_msg = type(e).__name__
            return {
                "draft_answer": "[LLM ERROR: " + err_msg + "]",
                "key_claims": [],
                "limitations_and_uncertainty": [],
            }

        # Check for abstention
        if response.strip().startswith("ABSTAIN:"):
            return {
                "draft_answer": response.strip(),
                "key_claims": [],
                "limitations_and_uncertainty": ["Insufficient evidence to answer the query."],
            }

        parsed = self.parse_llm_response(response)
        if not isinstance(parsed, dict):
            return {
                "draft_answer": f"[PARSE ERROR] Raw response:\n{response}",
                "key_claims": [],
                "limitations_and_uncertainty": [],
            }

        return {
            "draft_answer": parsed.get("draft_answer", ""),
            "key_claims": parsed.get("key_claims", []),
            "limitations_and_uncertainty": parsed.get("limitations_and_uncertainty", []),
        }

    def log_trace(self, query, result):
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "draft_answer": result.get("draft_answer", ""),
            "num_claims": len(result.get("key_claims", [])),
            "num_evidence": len(result.get("evidence", [])),
            "key_claims": result.get("key_claims", []),
            "limitations_and_uncertainty": result.get("limitations_and_uncertainty", []),
        }

        Path(self.traces_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    def save_output(self, result):
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def run(self, summary_pack):
        """
        Run the SynthesizerAgent.

        Input: {
            "query": str,
            "paper_summaries": [ {paper_id, summary_text, url, doi, evidence: [...]} ]
        }

        Output: {
            "query": str,
            "draft_answer": str,
            "key_claims": [ {claim, citation_ids} ],
            "evidence": [ {paper_id, chunk_id, score, section, url} ],
            "limitations_and_uncertainty": [str]
        }
        """
        query = summary_pack["query"]
        paper_summaries = summary_pack["paper_summaries"]

        if not paper_summaries:
            return {
                "query": query,
                "draft_answer": "No paper summaries provided.",
                "key_claims": [],
                "evidence": [],
                "limitations_and_uncertainty": ["No input data available."],
            }

        # Filter out abstained/errored summaries
        active_summaries = [
            ps for ps in paper_summaries
            if not _is_bad_summary(ps.get("summary_text", ""))
        ]

        if not active_summaries:
            return {
                "query": query,
                "draft_answer": "All paper summaries abstained due to insufficient evidence.",
                "key_claims": [],
                "evidence": [],
                "limitations_and_uncertainty": ["All retrieved papers had insufficient evidence."],
            }

        # Call LLM to synthesize
        llm_result = self.synthesize(query, active_summaries)

        # Build evidence list in Python from input data (NOT from LLM)
        cited_ids = self.extract_cited_ids(llm_result.get("key_claims", []))
        evidence = self.build_evidence_list(paper_summaries, cited_ids)

        result = {
            "query": query,
            "draft_answer": llm_result["draft_answer"],
            "key_claims": llm_result["key_claims"],
            "evidence": evidence,
            "limitations_and_uncertainty": llm_result["limitations_and_uncertainty"],
        }

        validate_synthesis_pack(result)
        self.log_trace(query, result)
        self.save_output(result)

        return result
