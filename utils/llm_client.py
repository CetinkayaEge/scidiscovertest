"""Multi-provider LLM client (Anthropic + Google Gemini)."""

import os
import anthropic
from google import genai
from google.genai import types as genai_types

_anthropic_client: anthropic.Anthropic | None = None
_gemini_client: genai.Client | None = None


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _call_anthropic(system: str, user: str, model: str, max_tokens: int) -> str:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def _call_gemini(system: str, user: str, model: str, max_tokens: int) -> str:
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


def call_llm(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 16000,
) -> str:
    """Call the appropriate LLM provider based on the model name.

    Model routing:
      - 'claude-*'  → Anthropic (requires ANTHROPIC_API_KEY)
      - 'gemini-*'  → Google Gemini (requires GOOGLE_API_KEY)
    """
    if model.startswith("claude-"):
        return _call_anthropic(system, user, model, max_tokens)
    elif model.startswith("gemini-"):
        return _call_gemini(system, user, model, max_tokens)
    else:
        raise ValueError(
            f"Unknown model '{model}'. "
            "Model must start with 'claude-' or 'gemini-'."
        )
