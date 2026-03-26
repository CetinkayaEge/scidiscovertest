import json
from collections import defaultdict
from pathlib import Path

from utils.llm_client import call_llm


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

    # 🔥 TITLE'ları filtrele (ÇOK ÖNEMLİ)
    def select_best_chunks(self, chunks):
        # önce ABSTRACT olanları al
        abstracts = [c for c in chunks if c["section"] == "ABSTRACT"]

        if abstracts:
            return abstracts[:3]

        # yoksa diğerlerini kullan
        return chunks[:3]

    # 🔥 daha temiz context
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

        response = call_llm(
            system=prompt,
            user="",
            model="gemini-3-flash-preview"
        )

        return {
            "paper_id": paper_id,
            "summary": response,
            "evidence": [
                {
                    "chunk_id": ch["chunk_id"],
                    "paper_id": ch["paper_id"],
                    "section": ch["section"],
                    "score": ch["score"],
                }
                for ch in selected_chunks
            ]
        }

    def log_trace(self, query, results):
        trace = {
            "query": query,
            "num_papers": len(results),
            "papers": results,
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

        grouped = self.group_by_paper(chunks)

        results = []

        for paper_id, chs in grouped.items():
            summary = self.summarize_single_paper(query, paper_id, chs)
            results.append(summary)

        self.log_trace(query, results)
        self.save_output(results)

        return results