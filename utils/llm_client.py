"""Multi-provider LLM client """

import os
import warnings
import anthropic
import openai
from google import genai
from google.genai import types as genai_types

_anthropic_client: anthropic.Anthropic | None = None
_gemini_client: genai.Client | None = None
_local_client: openai.OpenAI | None = None

_model: str | None = None
_max_tokens: int | None = None


def configure_llm(model: str, max_tokens: int) -> None:
    global _model, _max_tokens
    _model = model
    _max_tokens = max_tokens


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


def _get_local_client() -> openai.OpenAI:
    global _local_client
    if _local_client is None:
        base_url = os.environ.get("LOCAL_MODEL_URL")
        api_key = os.environ.get("LOCAL_MODEL_API_KEY", "not-needed")
        if not base_url:
            raise EnvironmentError(
                "LOCAL_MODEL_URL is not set. Add your Cloudflare tunnel URL to your .env file."
            )
        _local_client = openai.OpenAI(base_url=base_url, api_key=api_key)
    return _local_client


def _call_local(system: str, user: str, model: str, max_tokens: int) -> str:
    client = _get_local_client()
    actual_model = model[len("local-"):]
    response = client.chat.completions.create(
        model=actual_model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        warnings.warn(
            f"[llm_client] Output truncated at {max_tokens} tokens (finish_reason=length). "
            "Response may be incomplete JSON. Consider raising max_tokens in configs/demo.yaml.",
            RuntimeWarning,
            stacklevel=3,
        )
    return choice.message.content or ""


def _call_anthropic(system: str, user: str, model: str, max_tokens: int) -> str:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "max_tokens":
        warnings.warn(
            f"[llm_client] Output truncated at {max_tokens} tokens (stop_reason=max_tokens). "
            "Response may be incomplete JSON. Consider raising max_tokens in configs/demo.yaml.",
            RuntimeWarning,
            stacklevel=3,
        )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def _call_gemini(system: str, user: str, model: str, max_tokens: int) -> str:
    client = _get_gemini_client()
    contents = user if user.strip() else "Generate the response as instructed."
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )
    candidate = response.candidates[0] if response.candidates else None
    if candidate and str(candidate.finish_reason) in ("FinishReason.MAX_TOKENS", "MAX_TOKENS", "2"):
        warnings.warn(
            f"[llm_client] Output truncated at {max_tokens} tokens (finish_reason=MAX_TOKENS). "
            "Response may be incomplete JSON. Consider raising max_tokens in configs/demo.yaml.",
            RuntimeWarning,
            stacklevel=3,
        )
    return response.text


def call_llm(system: str, user: str) -> str:
    """Call the appropriate LLM provider based on the configured model.

    Model routing:
      - 'claude-*'  → Anthropic (requires ANTHROPIC_API_KEY)
      - 'gemini-*'  → Google Gemini (requires GOOGLE_API_KEY)
      - 'local-*'   → Local model via Cloudflare tunnel (requires LOCAL_MODEL_URL)

    Must call configure_llm() before using this function.
    """
    if _model is None or _max_tokens is None:
        raise RuntimeError("call_llm used before configure_llm() was called.")

    if _model.startswith("claude-"):
        return _call_anthropic(system, user, _model, _max_tokens)
    elif _model.startswith("gemini-"):
        return _call_gemini(system, user, _model, _max_tokens)
    elif _model.startswith("local-"):
        return _call_local(system, user, _model, _max_tokens)
    else:
        raise ValueError(
            f"Unknown model '{_model}'. "
            "Model must start with 'claude-', 'gemini-', or 'local-'."
        )
