import time


from fastapi import FastAPI
from prometheus_client import make_asgi_app
from dotenv import load_dotenv
from app.api.chat import router as chat_router

from app.core.metrics import REQUESTS, LATENCY

load_dotenv()

app = FastAPI(title="Agentic RAG API")

app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
app.mount("/metrics", make_asgi_app())


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.middleware("http")
async def collect_http_metrics(request, call_next):
    if request.url.path in {"/health", "/docs", "/openapi.json"}:
        return await call_next(request)

    start = time.perf_counter()
    status = "5xx"

    try:
        response = await call_next(request)
        status = f"{response.status_code // 100}xx"
        return response
    finally:
        REQUESTS.labels(status=status).inc()
        LATENCY.observe(time.perf_counter() - start)
