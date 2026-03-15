# Chunking Policy

## Extraction
We extract the following sections from each paper:
- title
- abstract

Full text is not used in the Phase-1 prototype.

## Chunk Size
Chunk size: 200 tokens

## Overlap
Chunk overlap: 40 tokens

## Tokenization
Whitespace tokenization is used for deterministic chunking.

## Chunk ID Policy
Chunk IDs are deterministic and follow the rule:

chunk_id = paper_id + section + offset

Example:

pmc_PMC123456_abstract_000  
pmc_PMC123456_abstract_001  

## Output Format
Chunks are written to:

data/processed/chunks.jsonl

Each line contains:

- chunk_id
- paper_id
- section
- text
- token_len