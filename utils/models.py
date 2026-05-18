"""
Pydantic models for the SciDiscover pipeline.

Two categories of models exist here:

  VALIDATION MODELS — used to validate our own code's output at agent
  boundaries (EvidenceChunk, PaperSummary, SynthesisPack, VerificationPack).
  These are strict: missing or wrong-typed fields raise ValidationError.

  LLM OUTPUT MODELS — used to parse and normalise JSON returned by the LLM
  (SynthesizerLLMOutput, VerifierLLMOutput, QueryDecomposerOutput).
  These are permissive: unknown extra fields are silently ignored, missing
  optional fields fall back to sensible defaults, and validators normalise
  things the LLM commonly gets wrong (wrong status strings, out-of-range
  confidence scores, null citation lists).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval layer  (Retriever → RerankerAgent → SummarizerAgent)
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceChunk(BaseModel):
    """One chunk entry in an evidence_pack["chunks"] list."""
    chunk_id: str
    paper_id: str
    section: str
    text: str
    token_len: int
    score: float
    url: str
    doi: str


# ─────────────────────────────────────────────────────────────────────────────
# Summariser layer  (SummarizerAgent output → SynthesizerAgent input)
# ─────────────────────────────────────────────────────────────────────────────

class SummaryEvidence(BaseModel):
    """One chunk reference inside a PaperSummary's evidence list."""
    chunk_id: str
    paper_id: str
    section: str
    score: float
    text_preview: str


class PaperSummary(BaseModel):
    """One paper's summariser output; a list of these forms a summary_pack."""
    paper_id: str
    summary_text: str
    url: str
    doi: str
    evidence: list[SummaryEvidence]


# ─────────────────────────────────────────────────────────────────────────────
# Synthesiser layer  (SynthesizerAgent output → VerifierAgent input)
# ─────────────────────────────────────────────────────────────────────────────

class KeyClaim(BaseModel):
    """A single factual claim with its supporting chunk citations.

    Defaults are intentionally permissive so that LLM-produced JSON with
    missing fields degrades gracefully rather than crashing; the final
    validate_synthesis_pack() call catches anything structurally wrong.
    """
    model_config = ConfigDict(extra="ignore")

    claim: str = ""
    citation_ids: list[str] = []

    @field_validator("citation_ids", mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, v: Any) -> list:
        # LLMs sometimes return null instead of []; treat both as empty list.
        return v if v is not None else []


class SynthesisEvidence(BaseModel):
    """One chunk reference in the synthesiser evidence lookup table."""
    paper_id: str
    chunk_id: str
    score: float
    section: str
    url: str
    text_preview: str


class SynthesisPack(BaseModel):
    """Full output of SynthesizerAgent; validated by validate_synthesis_pack()."""
    query: str
    draft_answer: str
    key_claims: list[KeyClaim]
    evidence: list[SynthesisEvidence]
    limitations_and_uncertainty: list[str]


# LLM output shape ── permissive, used inside synthesizer.synthesize()
class SynthesizerLLMOutput(BaseModel):
    """Shape of the JSON the LLM returns from the synthesiser prompt.

    Uses ConfigDict(extra="ignore") so that any unexpected top-level keys the
    LLM adds are silently discarded instead of causing a ValidationError.
    All fields have defaults so that a partially-formed response is still
    usable rather than raising an error immediately.
    """
    model_config = ConfigDict(extra="ignore")

    draft_answer: str = ""
    key_claims: list[KeyClaim] = []
    limitations_and_uncertainty: list[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# Verifier layer  (VerifierAgent output → final answer)
# ─────────────────────────────────────────────────────────────────────────────

class VerifiedClaim(BaseModel):
    """A key_claim entry after the verifier has assessed it."""
    claim: str
    citation_ids: list[str]
    status: Literal["SUPPORTED", "UNSUPPORTED", "CONFLICT", "UNKNOWN"]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str


class VerificationSummary(BaseModel):
    """Aggregate statistics produced by VerifierAgent._compute_summary()."""
    citation_coverage: float
    support_rate: float
    avg_confidence: float
    total_claims: int
    supported: int
    unsupported: int
    conflict: int
    unknown: int


class VerificationPack(BaseModel):
    """Full output of VerifierAgent; validated by validate_verification_pack()."""
    query: str
    final_answer: str
    key_claims: list[VerifiedClaim]
    evidence: list[SynthesisEvidence]
    limitations_and_uncertainty: list[str]
    verification_summary: VerificationSummary


# LLM output shapes ── permissive, used inside verifier._parse_llm_response()

class VerifierLLMClaim(BaseModel):
    """One claim entry in the LLM's verifier JSON response.

    Validators normalise the two fields where LLMs most commonly go wrong:
      • status  — any unrecognised string becomes "UNKNOWN"
      • confidence — any value outside [0, 1] is clamped rather than rejected
    extra="ignore" swallows claim_id and any other extra keys the LLM adds.
    All fields have defaults so a missing field never causes a hard failure.
    """
    model_config = ConfigDict(extra="ignore")

    claim: str = ""
    status: str = "UNKNOWN"
    confidence: float = 0.5
    notes: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _normalise_status(cls, v: Any) -> str:
        s = str(v).upper().strip() if v is not None else "UNKNOWN"
        return s if s in {"SUPPORTED", "UNSUPPORTED", "CONFLICT", "UNKNOWN"} else "UNKNOWN"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


class VerifierLLMOutput(BaseModel):
    """Shape of the JSON the LLM returns from the verifier prompt."""
    model_config = ConfigDict(extra="ignore")

    verified_claims: list[VerifierLLMClaim] = []
    final_answer: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Single-agent baseline
# ─────────────────────────────────────────────────────────────────────────────

class SingleAgentLLMOutput(BaseModel):
    """Shape of the JSON the LLM returns from the single-agent baseline prompt.

    Same permissive rules as SynthesizerLLMOutput but also includes final_answer
    at the top level (the baseline collapses synthesis + verification into one call).
    """
    model_config = ConfigDict(extra="ignore")

    final_answer: str = ""
    key_claims: list[KeyClaim] = []
    limitations_and_uncertainty: list[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# Query decomposer
# ─────────────────────────────────────────────────────────────────────────────

class QueryDecomposerOutput(BaseModel):
    """Shape of the JSON the LLM returns from the query-decomposer prompt."""
    model_config = ConfigDict(extra="ignore")

    sub_queries: list[str] = []
