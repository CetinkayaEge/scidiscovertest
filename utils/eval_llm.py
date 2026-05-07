"""
Shared LLM factory for evaluation steps (testset generation + RAGAS scoring).

Provider routing by model name prefix:
  openai/*   anthropic/*   meta-llama/*   (any OpenRouter model)
      → OpenRouter via langchain_openai.ChatOpenAI
  gemini-*
      → Google Gemini via langchain_google_genai.ChatGoogleGenerativeAI

Usage:
    from utils.eval_llm import make_eval_llm
    lc_llm = make_eval_llm("openai/gpt-4o-mini")
    lc_llm = make_eval_llm("gemini-2.5-flash")
"""

import os


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _is_openrouter(model: str) -> bool:
    """Any model with a slash is an OpenRouter model ID (provider/model-name)."""
    return "/" in model


def make_eval_llm(model: str, temperature: float = 0.3):
    """Return a LangChain chat LLM for the given model string."""
    if _is_openrouter(model):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    elif model.startswith("gemini-"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            response_mime_type="application/json",
        )
    else:
        raise ValueError(
            f"Unknown eval model '{model}'. "
            "Use an OpenRouter model (e.g. 'openai/gpt-4o-mini') "
            "or a Gemini model (e.g. 'gemini-2.5-flash')."
        )
