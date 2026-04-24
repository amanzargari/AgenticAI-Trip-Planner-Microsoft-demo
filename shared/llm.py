import os

from openai import AsyncOpenAI


def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        default_headers={
            "HTTP-Referer": "https://github.com/trip-planner-agent",
            "X-Title": "Trip Planner Agent",
        },
    )


DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "google/gemini-2.0-flash")
