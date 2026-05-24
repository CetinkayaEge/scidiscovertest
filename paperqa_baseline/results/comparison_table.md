# SOTA Baseline Comparison: SciDiscover vs PaperQA2

**Generated:** 2026-05-24 19:10:16
**SciDiscover mode:** `not available`  —  results from `/Users/zynpdgc/SciDiscovery/scidiscovertest/reports/eval_results.json`
**PaperQA2 model:** `gpt-5.4-mini`  —  run timestamp `2026-05-24T16:03:12.865860+00:00`  —  results from `/Users/zynpdgc/SciDiscovery/scidiscovertest/paperqa_baseline/results/paperqa_results.json`

> **Note on LLM fairness:** SciDiscover runs on a local HPC vLLM server (Qwen2.5-32B).
> PaperQA2 uses the real OpenAI API (gpt-4o-mini by default).  Direct numeric comparisons
> are therefore indicative only; for a fully controlled comparison both systems should use
> the same LLM backend.

## Metric Comparison

| Metric | SciDiscover | PaperQA2 | Notes |
| ------ | :---------: | :------: | ----- |
| Abstention rate — answerable queries (↓ better) | *(no results)* | 0.000 | Fraction of answerable queries where the system refused to answer |
| Correct abstention rate — unanswerable queries (↑ better) | *(no results)* | 0.933 | Fraction of unanswerable/OOD queries where the system correctly declined |
| Avg claims / citations per query | *(no results)* | 7.345 | SciDiscover: verified key_claims; PaperQA2: retrieved context passages |
| Avg latency (s) per query (↓ better) | *(no results)* | 22.344 | Mean wall-clock seconds from query submission to final answer |
| ROUGE-L vs reference (↑ better) | *(no results)* | 0.185 | Token-level F1 LCS overlap with ground-truth answer from eval/queries.jsonl |
| RAGAS Faithfulness (↑ better) | *(no results)* | 0.791 | Fraction of answer claims supported by retrieved context (NLI-based) |
| RAGAS Answer Relevancy (↑ better) | *(no results)* | 0.945 | Cosine similarity of answer embeddings to the question |
| RAGAS Context Recall (↑ better) | *(no results)* | 0.705 | Fraction of ground-truth statements attributable to the retrieved context |
| Paper/chunk recall (↑ better) | *(no results)* | 0.826 | SciDiscover: chunk-level recall@k in FAISS; PaperQA2: paper-level recall vs expected_paper_ids |
| Citation coverage (↑ better) | *(no results)* | — | Fraction of key_claims citing ≥1 chunk [SciDiscover only] |
| Claim support rate (↑ better) | *(no results)* | — | Fraction of cited claims verified SUPPORTED by VerifierAgent NLI [SciDiscover only] |
| Synthesizer hallucination rate (↓ better) | *(no results)* | — | Fraction of cited chunk_ids absent from retrieved evidence set [SciDiscover only] |

## Metric Availability Notes

- **RAGAS**: RAGAS computed for both systems (run with `--run-ragas`).
- **Paper/chunk recall**: Paper-level recall computed for PaperQA2.
- **Citation coverage**: PaperQA2 does not expose per-claim citation-coverage in a structured field.
- **Claim support rate**: No equivalent of SciDiscover's VerifierAgent NLI checking in PaperQA2.
- **Synthesizer hallucination rate**: Internal metric tied to SciDiscover's chunk-ID citation format.

## Query-Level Statistics

| | SciDiscover | PaperQA2 |
| - | - | - |
| Total queries evaluated | — | 84 |
| Answerable queries | — | 69 |
| Unanswerable / OOD queries | — | 15 |

## How to Read This Table

- **Abstention rate (answerable)**: lower is better — a high rate means the system often
  refused to answer questions it should have been able to answer.
- **Correct abstention rate**: higher is better — the system should decline unanswerable
  and out-of-domain queries.
- **ROUGE-L**: higher is better; computed only for queries with a non-empty reference.
- **Paper/chunk recall**: SciDiscover uses chunk-level recall@k in its FAISS index;
  PaperQA2 uses paper-level recall vs expected_paper_ids from queries.jsonl.
  Both measure 'did the system retrieve the relevant source?' at different granularities.
- **—** in a column: metric is not available for that system.