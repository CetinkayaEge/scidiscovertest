import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from utils.llm_client import call_llm
from utils.schemas import validate_summary_pack


class SummarizerAgent:
    def __init__(
        self,
        prompt_path: str,
        traces_output: str = "logs/summarizer_traces.jsonl",
        output_path: str = "outputs/paper_summaries.json",
    ):
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        self.traces_output = traces_output
        self.output_path = output_path

    def group_by_paper(self, chunks):
        grouped = defaultdict(list)
        for ch in chunks:
            grouped[ch["paper_id"]].append(ch)
        return dict(grouped)

    def select_best_chunks(self, chunks):
        # Prefer ABSTRACT chunks; fall back to whatever is available
        abstracts = [c for c in chunks if c["section"] == "ABSTRACT"]
        if abstracts:
            return abstracts[:3]
        return chunks[:3]

    def build_context(self, chunks):
        parts = []
        for ch in chunks:
            text = ch["text"].strip()
            parts.append(f"[{ch['chunk_id']}]\n{text}")

        return "\n\n".join(parts)

    def summarize_single_paper(self, query, paper_id, chunks):
        selected_chunks = self.select_best_chunks(chunks)
        context = self.build_context(selected_chunks)

        prompt = self.prompt_template.format(
            query=query,
            paper_id=paper_id,
            context=context
        )

        try:
            response = call_llm(system=prompt, user="", json_mode=False)
        except Exception as e:
            response = f"[LLM ERROR: {e}]"

        return {
            "paper_id": paper_id,
            "summary_text": response,
            "url": selected_chunks[0].get("url", "") if selected_chunks else "",
            "doi": selected_chunks[0].get("doi", "") if selected_chunks else "",
            "evidence": [
                {
                    "chunk_id": ch["chunk_id"],
                    "paper_id": ch["paper_id"],
                    "section": ch["section"],
                    "score": ch["score"],
                    "text_preview": ch["text"][:300],
                }
                for ch in selected_chunks
            ]
        }

    def log_trace(self, query, results):
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "num_papers": len(results),
            "papers": [
                {
                    "paper_id": r["paper_id"],
                    "summary_text": r["summary_text"],
                    "url": r.get("url", ""),
                    "doi": r.get("doi", ""),
                    "evidence": r.get("evidence", []),
                }
                for r in results
            ],
        }

        Path(self.traces_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    def save_output(self, results):
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def run(self, evidence_pack):
        query = evidence_pack["query"]
        chunks = evidence_pack["chunks"]

        if not chunks:
            return []

        grouped = self.group_by_paper(chunks)

        results = []

        for paper_id, chs in grouped.items():
            summary = self.summarize_single_paper(query, paper_id, chs)
            results.append(summary)

        validate_summary_pack(results)
        self.log_trace(query, results)
        self.save_output(results)

        return results