import json
import logging
from functools import lru_cache
from hashlib import sha256
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    """Создаёт один Redis-клиент на процесс приложения."""
    settings = get_settings()

    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
    )


def make_cache_key(message: str, model: str) -> str:
    """Формирует ключ кэша по модели и тексту вопроса."""
    normalized_message = " ".join(message.split())
    raw_key = f"{model}\0{normalized_message}"
    digest = sha256(raw_key.encode("utf-8")).hexdigest()

    return f"rag:response:v1:{digest}"


async def get_cached_response(
    key: str,
) -> dict[str, Any] | None:
    """Возвращает сохранённый ответ или None при промахе."""
    try:
        raw_value = await get_redis_client().get(key)
    except (RedisError, OSError, TimeoutError) as exc:
        logger.warning(
            "Redis read failed; continuing without cache (%s)",
            type(exc).__name__,
        )
        return None

    if raw_value is None:
        return None

    try:
        value = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid JSON in Redis cache")
        return None

    return value if isinstance(value, dict) else None


async def set_cached_response(
    key: str,
    value: dict[str, Any],
    ttl_seconds: int,
) -> None:
    """Сохраняет ответ в Redis с ограниченным сроком жизни."""
    try:
        await get_redis_client().set(
            name=key,
            value=json.dumps(value, ensure_ascii=False),
            ex=ttl_seconds,
        )
    except (RedisError, OSError, TimeoutError) as exc:
        logger.warning(
            "Redis write failed; continuing without cache (%s)",
            type(exc).__name__,
        )


async def close_redis_client() -> None:
    """Закрывает Redis-клиент при остановке приложения."""
    if get_redis_client.cache_info().currsize == 0:
        return

    client = get_redis_client()
    get_redis_client.cache_clear()
    await client.aclose()