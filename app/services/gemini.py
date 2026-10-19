from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings


@lru_cache
def get_llm():
    settings = get_settings()

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key.get_secret_value(),
        temperature=0,
        max_retries=settings.max_retries,
    )