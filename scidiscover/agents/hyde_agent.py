import json
from pathlib import Path

from utils.llm_client import call_llm


class HyDEAgent:
    def __init__(
        self,
        prompt_path: str,
        traces_output: str = "logs/hyde_traces.jsonl",
    ):
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        self.traces_output = traces_output

    def log_trace(self, query: str, hypothetical_doc: str) -> None:
        trace = {"query": query, "hypothetical_doc": hypothetical_doc}
        Path(self.traces_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")

    def run(self, query: str) -> str:
        prompt = self.prompt_template.format(query=query)
        hypothetical_doc = call_llm(system=prompt, user="", json_mode=False)
        self.log_trace(query, hypothetical_doc)
        return hypothetical_doc
