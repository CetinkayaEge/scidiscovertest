REQUIRED_EVIDENCE_FIELDS = {"chunk_id", "paper_id", "section", "text", "token_len", "score", "url", "doi"}

REQUIRED_SUMMARY_FIELDS = {"paper_id", "summary_text", "evidence", "url", "doi"}

def validate_evidence_pack(pack: list) -> None:
    for i, item in enumerate(pack):
        missing = REQUIRED_EVIDENCE_FIELDS - item.keys()
        if missing:
            raise ValueError(f"Evidence item {i} missing fields: {missing}")

def validate_summary_pack(pack: list) -> None:
    for i, item in enumerate(pack):
        missing = REQUIRED_SUMMARY_FIELDS - item.keys()
        if missing:
            raise ValueError(f"Summary item {i} missing fields: {missing}")