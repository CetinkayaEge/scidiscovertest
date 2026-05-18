"""
Schema validation helpers used at agent boundaries.

Each function accepts the same shape it always did (list or dict) and raises
ValueError on failure — so all existing call sites continue to work unchanged.

Internally, validation is now done by Pydantic models imported from
utils.models, which means we get:
  • type checking  (not just field-presence checking)
  • range/constraint enforcement  (e.g. confidence in [0, 1])
  • clear, structured error messages that name the failing field and value
  • recursive validation of nested objects (e.g. evidence items inside PaperSummary)
"""

from pydantic import ValidationError

from utils.models import (
    EvidenceChunk,
    PaperSummary,
    SynthesisPack,
    VerificationPack,
)


def validate_evidence_pack(pack: list) -> None:
    """Validate a list of evidence chunk dicts (Retriever / RerankerAgent output)."""
    for i, item in enumerate(pack):
        try:
            EvidenceChunk.model_validate(item)
        except ValidationError as exc:
            raise ValueError(
                f"Evidence item {i} validation failed:\n{exc}"
            ) from exc


def validate_summary_pack(pack: list) -> None:
    """Validate a list of paper summary dicts (SummarizerAgent output)."""
    for i, item in enumerate(pack):
        try:
            PaperSummary.model_validate(item)
        except ValidationError as exc:
            raise ValueError(
                f"Summary item {i} validation failed:\n{exc}"
            ) from exc


def validate_synthesis_pack(pack: dict) -> None:
    """Validate the synthesis output dict (SynthesizerAgent output)."""
    try:
        SynthesisPack.model_validate(pack)
    except ValidationError as exc:
        raise ValueError(
            f"Synthesis pack validation failed:\n{exc}"
        ) from exc


def validate_verification_pack(pack: dict) -> None:
    """Validate the full verification output dict (VerifierAgent output)."""
    try:
        VerificationPack.model_validate(pack)
    except ValidationError as exc:
        raise ValueError(
            f"Verification pack validation failed:\n{exc}"
        ) from exc
