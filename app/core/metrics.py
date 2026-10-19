from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "rag_requests_total",
    "Количество запросов",
    ["status"],
)

LATENCY = Histogram(
    "rag_request_duration_seconds",
    "Время обработки запроса",
)

RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Количество найденных фрагментов",
)

AGENT_ROUTES = Counter(
    "rag_agent_routes_total",
    "Количество переходов по маршрутам",
    ["route"],
)