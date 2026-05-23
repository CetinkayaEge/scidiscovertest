# Data Collection Pipeline

Papers are collected from up to three sources: PMC (PubMed Central), OpenAlex, and Web of Science (WOS). All ingestors write to the same JSONL file and the same corpus manifest CSV.

**Output files:**
- `data/raw/papers.jsonl` — all ingested papers
- `docs/corpus_manifest.csv` — provenance record for every paper

---

## 1. PMC Ingestor

**File:** `scidiscover/ingest/pmc_ingest.py`

### APIs Used

| API | URL | Purpose |
|-----|-----|---------|
| PMC OA Web Service | `https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi` | Lists open-access articles in a date range |
| PMC OAI-PMH | `https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/` | Fetches front-matter XML per article |

### Key Functions

```python
list_oa_records(from_date, until_date, max_records, require_pdf=False) -> List[OARecord]
```
- Iterates the PMC OA API in monthly batches using resumption tokens
- Returns a list of `OARecord` dataclasses: `pmcid`, `license_note`, `urls`, `doi`
- Politeness delay: 0.34 s per request, exponential backoff on failure
- `max_records` is a hard cutoff — iteration stops as soon as the limit is reached

```python
oai_get_pmc_front_matter_xml(pmc_numeric_id: str) -> ET.Element
```
- Fetches OAI-PMH `pmc_fm` metadata for a single article ID
- Makes one HTTP call per paper — this is the main time cost for large runs

```python
parse_pmc_front_matter(root: ET.Element) -> Dict
```
- Namespace-agnostic XML parsing
- Extracts: `article-title`, `abstract` (all paragraphs joined), `journal-title`, `pub-date/year`, DOI, authors (`surname` + `given-names`)

### Configuration (demo.yaml)

```yaml
sources:
  pmc:
    enabled: true
    from_date: "2023-01-01"
    until_date: "2025-12-31"
    max_papers: 8000
    skip_empty_abstract: true
```

### Filters Applied

- Abstract must be non-empty (`skip_empty_abstract: true`)
- Title must be non-empty

---

## 2. OpenAlex Ingestor

**File:** `scidiscover/ingest/openalex_ingest.py`

### API Used

| API | URL |
|-----|-----|
| OpenAlex Works | `https://api.openalex.org/works` |

### Key Functions

```python
reconstruct_abstract(inverted_index: Optional[Dict]) -> str
```
- OpenAlex stores abstracts as inverted indexes (`{word: [position, ...]}`)
- Reconstructs the original sentence by sorting positions

```python
fetch_papers(queries, from_year, max_papers, email, is_oa=True, has_abstract=True) -> List[Dict]
```
- Cursor-based pagination, 200 results per page
- Runs each query in `queries` list separately, deduplicates by paper ID across all queries
- `max_papers` is a **shared cap across all queries** — if it is reached on an early query, remaining queries never run. Always set `max_papers` high enough to allow all queries to contribute.
- Year filter is a strict inequality: `publication_year > from_year`. So `from_year: 2020` returns papers from 2021 onward.
- Rate-limit handling: 403 → 5 s sleep; politeness delay: 0.2 s between requests

**`OpenAlexIngestor` class** — config-driven wrapper around `fetch_papers()`.

### Allowed Domains

**Only two top-level domains are permitted for this project: `sustainability` and `healthcare`.**
All queries in `sources.openalex.queries` must belong to one of these two domains. Do not add queries from unrelated fields (e.g. machine learning, neuroscience, physics).

### Configuration (demo.yaml)

```yaml
sources:
  openalex:
    enabled: true
    queries:
      - "sustainability"
      - "healthcare"
      - "renewable energy"
      - "climate change"
      - "public health"
      - "environmental science"
      - "circular economy"
      - "carbon emissions"
      - "sustainable agriculture"
      - "biodiversity conservation"
      - "mental health"
      - "infectious disease"
      - "chronic disease"
      - "digital health"
      - "global health equity"
      - "cancer research"
    from_year: 2020
    max_papers: 15000
    is_oa: true
    has_abstract: true
```

---

## 3. Web of Science (WOS) Ingestor

**File:** `scidiscover/ingest/wos_ingest.py`

**Requires:** `WOS_API_KEY` environment variable (Clarivate Developer Portal). Set `sources.wos.enabled: true` in `configs/demo.yaml` to activate.

### API Used

| API | URL |
|-----|-----|
| WOS Starter API | `https://api.clarivate.com/apis/wos-starter/v1/documents` |

### Key Functions

```python
fetch_papers(queries, from_year, to_year, max_papers, api_key, db="WOS") -> List[Dict]
```
- Page-based pagination, 50 results per page (Starter API maximum)
- Runs each query in `queries` list separately, deduplicates by DOI across all queries
- `max_papers` is a **shared cap across all queries** — same behavior as OpenAlex
- Query is constructed as: `TS=({query}) AND PY=({from_year}-{to_year})`
- Politeness delay between pages

**`WOSIngestor` class** — config-driven wrapper around `fetch_papers()`.

### Allowed Domains

Same domain constraint as OpenAlex — **only `sustainability` and `healthcare` domains** are permitted.

### Configuration (demo.yaml)

```yaml
sources:
  wos:
    enabled: false          # set to true once WOS_API_KEY is configured
    from_year: 2020
    to_year: 2024
    max_papers: 5000
    db: WOS
    queries:
      - "sustainability"
      - "healthcare"
      - "renewable energy"
      - "climate change"
      - "public health"
      - "environmental science"
      - "circular economy"
      - "carbon emissions"
      - "sustainable agriculture"
      - "biodiversity conservation"
      - "mental health"
      - "infectious disease"
      - "chronic disease"
      - "digital health"
      - "global health equity"
      - "cancer research"
```

### Standalone CLI

```bash
python -m scidiscover.ingest.wos_ingest \
    --queries "sustainability" "healthcare" \
    --from-year 2020 \
    --to-year 2024 \
    --max-papers 500 \
    --raw-output data/raw/papers.jsonl \
    --manifest-output docs/corpus_manifest.csv
```

---

## 4. Shared Paper Schema

All three ingestors output papers in the same JSONL format:

```json
{
  "paper_id": "pmc_PMC10002645",
  "title": "How Healthy Older Adults Enact Lateral Maneuvers While Walking",
  "abstract": "Background: Walking requires frequent maneuvers...",
  "year": 2023,
  "authors": ["David M. Desmet", "Meghan E. Kazanski"],
  "venue": "bioRxiv",
  "doi": "10.1101/2023.02.24.529927",
  "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10002645/",
  "source": "pmc",
  "retrieved_date": "2026-03-19",
  "license_note": "CC BY-NC-ND"
}
```

**Paper ID format:**
- PMC: `pmc_PMC<numeric_id>` (e.g., `pmc_PMC10002645`)
- OpenAlex: full URL (e.g., `https://openalex.org/W2742189230`)
- WOS: `wos_<DOI>` (e.g., `wos_10.1016/j.example.2023.001`)

---

## 5. Corpus Manifest

**File:** `docs/corpus_manifest.csv`

**Columns:**

| Column | Example |
|--------|---------|
| `paper_id` | `pmc_PMC10002645` |
| `source` | `pmc` |
| `url/doi` | `10.1101/2023.02.24.529927` |
| `retrieved_date` | `2026-03-19` |
| `license_note` | `CC BY-NC-ND` |

The manifest is written/appended by all three ingestors during every ingestion run. It serves as a provenance log — it does not store full text. Check `wc -l docs/corpus_manifest.csv` for the current paper count (subtract 1 for the header row).

---

## 6. Increasing the Corpus

To collect more papers, the primary levers are:

1. **`max_papers`** under `sources.pmc`, `sources.openalex`, and `sources.wos` in `configs/demo.yaml`
2. **Date ranges** (`from_date` / `until_date` for PMC; `from_year` / `to_year` for WOS and OpenAlex)
3. **OpenAlex / WOS queries** — adding more topic strings within the allowed domains increases coverage
4. **Enable WOS** — set `sources.wos.enabled: true` and supply `WOS_API_KEY` to add the Clarivate corpus

After changing these values, re-run the full pipeline:

```bash
python -m scidiscover.run_demo --config configs/demo.yaml
```
