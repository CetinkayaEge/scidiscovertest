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


REQUIRED_SYNTHESIS_FIELDS = {"query", "draft_answer", "key_claims", "evidence", "limitations_and_uncertainty"}
REQUIRED_CLAIM_FIELDS = {"claim", "citation_ids"}
REQUIRED_SYNTH_EVIDENCE_FIELDS = {"paper_id", "chunk_id", "score", "section", "url"}


def validate_synthesis_pack(pack: dict) -> None:
    missing = REQUIRED_SYNTHESIS_FIELDS - pack.keys()
    if missing:
        raise ValueError(f"Synthesis pack missing fields: {missing}")

    if not isinstance(pack["key_claims"], list):
        raise ValueError("key_claims must be a list")

    for i, claim in enumerate(pack["key_claims"]):
        missing_c = REQUIRED_CLAIM_FIELDS - claim.keys()
        if missing_c:
            raise ValueError(f"Claim {i} missing fields: {missing_c}")

    if not isinstance(pack["evidence"], list):
        raise ValueError("evidence must be a list")

    for i, ev in enumerate(pack["evidence"]):
        missing_e = REQUIRED_SYNTH_EVIDENCE_FIELDS - ev.keys()
        if missing_e:
            raise ValueError(f"Evidence item {i} missing fields: {missing_e}")

    if not isinstance(pack["limitations_and_uncertainty"], list):
        raise ValueError("limitations_and_uncertainty must be a list")