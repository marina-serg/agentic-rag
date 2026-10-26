
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.agents.graph import agent
from app.core.config import get_settings
from app.core.metrics import REQUESTS, LATENCY
from app.core.response_cache import (
    make_cache_key,
    get_cached_response,
    set_cached_response,
)
from app.schemas.chat import ChatRequest, ChatResponse


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    started = perf_counter()
    status = "error"

    try:
        settings = get_settings()
        message = request.message

        cache_key = make_cache_key(
            message=message,
            model=settings.gemini_model,
        )

        cache_data = await get_cached_response(cache_key)
        cache_hit = cache_data is not None

        if cache_hit:
            answer = cache_data.get("answer", "")

        else:
            result = await run_in_threadpool(
                agent.invoke,
                {"question": message},
            )

            answer = result.get("answer", "")

            if not answer:
                raise HTTPException(
                    status_code=500,
                    detail={"error": "empty_agent_response"},
                )

            cache_data = {
                "answer": answer,
                "sources": result.get("sources", []),
                "route": result.get("route"),
                "retrieved_chunks": result.get(
                    "retrieved_chunks", 0
                ),
            }

            if result.get("cacheable", False):
                await set_cached_response(
                    key=cache_key,
                    value=cache_data,
                    ttl_seconds=settings.redis_cache_ttl_seconds,
                )

        response = ChatResponse(
            response=answer,
            thread_id=request.thread_id or uuid4(),
            model_used=settings.gemini_model,
            cached=cache_hit,
            processing_time_ms=(
                perf_counter() - started
            ) * 1000,
            sources=cache_data.get("sources", []),
            agent_route=cache_data.get("route"),
            retrieved_chunks=cache_data.get(
                "retrieved_chunks", 0
            ),
        )

        status = "success"
        return response

    except HTTPException:
        raise

    except ValidationError as exc:
        logger.warning(
            "Ошибка валидации данных чата: %s",
            exc.errors(
                include_input=False,
                include_context=False,
                include_url=False,
            ),
        )

        raise HTTPException(
            status_code=422,
            detail={"error": "validation_error"},
        ) from None

    except Exception:
        logger.exception("Неожиданная ошибка")

        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error"},
        ) from None

    finally:
        REQUESTS.labels(status=status).inc()
        LATENCY.observe(perf_counter() - started)
