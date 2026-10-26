
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
    settings = get_settings()

    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
    )


def make_cache_key(message: str, model: str) -> str:
    normalized_message = " ".join(message.split())
    raw_key = f"{model}\0{normalized_message}"
    digest = sha256(raw_key.encode("utf-8")).hexdigest()

    return f"rag:response:v1:{digest}"


async def get_cached_response(
    key: str,
) -> dict[str, Any] | None:
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

    if not isinstance(value, dict):
        logger.warning("Invalid response cache structure")
        return None

    if not isinstance(value.get("answer"), str):
        logger.warning("Invalid response cache answer")
        return None

    return value


async def set_cached_response(
    key: str,
    value: dict[str, Any],
    ttl_seconds: int,
) -> None:
    if ttl_seconds <= 0:
        logger.warning("Redis cache write skipped: invalid TTL")
        return

    if not isinstance(value.get("answer"), str):
        logger.warning("Redis cache write skipped: invalid answer")
        return

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
        )

        await get_redis_client().set(
            name=key,
            value=serialized,
            ex=ttl_seconds,
        )
    except (RedisError, OSError, TimeoutError) as exc:
        logger.warning(
            "Redis write failed; continuing without cache (%s)",
            type(exc).__name__,
        )


async def close_redis_client() -> None:
    if get_redis_client.cache_info().currsize == 0:
        return

    client = get_redis_client()
    get_redis_client.cache_clear()
    await client.aclose()
