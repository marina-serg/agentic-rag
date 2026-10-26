from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: Literal["development", "testing", "production"] = "development"
    log_level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"

    # Google Gemini
    gemini_api_key: SecretStr
    gemini_model: str = "gemini-3.1-flash-lite"
    max_retries: int = Field(default=3, ge=0, le=10)

    # LangSmith
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "agentic-rag-local"

    # Ограничение запросов и кэширование
    rate_limit_per_minute: int = Field(default=20, ge=1)
    cache_ttl_seconds: int = Field(default=300, ge=0)

    # Qdrant
    qdrant_db_path: str = "./qdrant_db"
    qdrant_collection_name: str = "rag_documents"
    qdrant_upsert_batch_size: int = Field(default=64, ge=1, le=1024)

    # Документы
    documents_dir: str = "./documents"

    # Разбиение документов на фрагменты
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # Поиск по документам
    rag_top_k: int = Field(default=4, ge=1, le=20)
    rag_candidate_k: int = Field(default=20, ge=1, le=100)
    rag_final_k: int = Field(default=4, ge=1, le=20)

    # Генерация нескольких вариантов поискового запроса
    rag_multi_query_enabled: bool = False
    rag_multi_query_count: int = Field(default=3, ge=1, le=5)

    # Переранжирование результатов поиска
    rag_reranking_enabled: bool = False
    rag_reranker_model: str = "jina-reranker-v2-base-multilingual"

    # Гибридный поиск
    rag_hybrid_search_enabled: bool = False

    # Фильтрация результатов
    rag_min_relevance_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    rag_max_chunks_per_source: int | None = Field(
        default=2,
        ge=1,
        le=20,
    )

    # Jina Embeddings
    jina_api_key: SecretStr
    jina_embedding_model: str = "jina-embeddings-v3"
    max_input_length: int = Field(default=10_000, ge=1)
    max_output_length: int = Field(default=20_000, ge=1)

    # Redis
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_cache_ttl_seconds: int = Field(default=300, ge=1, le=86400)

    # Grafana
    grafana_user: str = ""
    grafana_password: SecretStr = SecretStr("")

    # Prometheus и Grafana
    prometheus_port: int = Field(default=9090, ge=1, le=65535)
    grafana_port: int = Field(default=3000, ge=1, le=65535)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """Проверяет согласованность настроек."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP должен быть меньше CHUNK_SIZE."
            )

        if self.rag_candidate_k < self.rag_final_k:
            raise ValueError(
                "RAG_CANDIDATE_K должен быть не меньше RAG_FINAL_K."
            )

        return self


@lru_cache
def get_settings() -> Settings:

    return Settings()
