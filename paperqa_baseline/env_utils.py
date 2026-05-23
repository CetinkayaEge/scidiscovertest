"""Utilities for temporarily overriding OpenAI environment variables.

The main pipeline is configured to talk to a local HPC vLLM server via
OPENAI_BASE_URL + OPENAI_API_KEY.  PaperQA2 must hit the real OpenAI API
because it relies on gpt-4o-mini and text-embedding-3-small.

The context manager here swaps in the SOTA key and removes the HPC base-URL
for the duration of the PaperQA2 run, then restores the originals on exit.
This way the HPC configuration is never disturbed for the main pipeline.
"""
from __future__ import annotations

import contextlib
import os
from typing import Iterator


@contextlib.contextmanager
def sota_openai_env() -> Iterator[None]:
    """Temporarily override OPENAI_* env vars for the PaperQA2 SOTA run.

    On entry:
      - Sets OPENAI_API_KEY  → value of OPENAI_API_KEY_SOTA
      - Removes OPENAI_BASE_URL and OPENAI_API_BASE so LiteLLM routes to
        the real OpenAI endpoint (api.openai.com) instead of the HPC server.

    On exit: restores all three variables to their original values (or
    removes them if they were not set before).

    Raises:
        EnvironmentError: If OPENAI_API_KEY_SOTA is not set or is a placeholder.
    """
    sota_key = os.environ.get("OPENAI_API_KEY_SOTA", "")
    if not sota_key or sota_key.startswith("<") or sota_key == "sk-...":
        raise EnvironmentError(
            "OPENAI_API_KEY_SOTA is not set or is still a placeholder.\n"
            "Add a real OpenAI API key to your .env file:\n"
            "  OPENAI_API_KEY_SOTA=sk-<your-real-key>"
        )

    orig: dict[str, str | None] = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
        "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
    }

    try:
        os.environ["OPENAI_API_KEY"] = sota_key
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OPENAI_API_BASE", None)
        yield
    finally:
        for var, val in orig.items():
            if val is not None:
                os.environ[var] = val
            else:
                os.environ.pop(var, None)


def check_sota_key_present() -> bool:
    """Return True if OPENAI_API_KEY_SOTA looks like a real key."""
    key = os.environ.get("OPENAI_API_KEY_SOTA", "")
    return bool(key) and not key.startswith("<") and key != "sk-..."
