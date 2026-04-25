import os

from openai import AsyncOpenAI


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_llm_client() -> AsyncOpenAI:
    timeout_sec = _env_float("LLM_TIMEOUT_SEC", 60.0)
    max_retries = _env_int("LLM_MAX_RETRIES", 1)
    return AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=timeout_sec,
        max_retries=max_retries,
        default_headers={
            "HTTP-Referer": "https://github.com/trip-planner-agent",
            "X-Title": "Trip Planner Agent",
        },
    )


DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "google/gemini-2.0-flash")
