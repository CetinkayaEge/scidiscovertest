import json
from pathlib import Path

from utils.llm_client import call_llm


class QueryDecomposerAgent:
    def __init__(
        self,
        prompt_path: str,
        max_sub_queries: int = 3,
        traces_output: str = "logs/query_decomposer_traces.jsonl",
    ):
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        self.max_sub_queries = max_sub_queries
        self.traces_output = traces_output

    def log_trace(self, query: str, sub_queries: list[str]) -> None:
        trace = {"query": query, "sub_queries": sub_queries}
        Path(self.traces_output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.traces_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")

    def run(self, query: str) -> list[str]:
        prompt = self.prompt_template.format(
            query=query,
            max_sub_queries=self.max_sub_queries,
        )
        response = call_llm(system=prompt, user="", json_mode=True)
        data = json.loads(response)
        sub_queries = data.get("sub_queries", [query])
        if not sub_queries:
            sub_queries = [query]
        self.log_trace(query, sub_queries)
        return sub_queries
