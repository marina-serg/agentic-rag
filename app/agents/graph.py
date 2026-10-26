
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.services.gemini import get_llm
from app.rag.retriever import search_documents
from app.core.metrics import AGENT_ROUTES, RETRIEVED_CHUNKS
from app.core.security import (
    SecurityError,
    check_input,
    check_output,
    mask_pii,
)
from app.core.prompts import (
    DIRECT_SYSTEM_PROMPT,
    EMPTY_MODEL_RESPONSE_MESSAGE,
    INSUFFICIENT_CONTEXT_MESSAGE,
    INPUT_SECURITY_ERROR_MESSAGE,
    INVALID_INPUT_MESSAGE,
    OUTPUT_SECURITY_ERROR_MESSAGE,
    RAG_HUMAN_PROMPT_TEMPLATE,
    RAG_SYSTEM_PROMPT,
    ROUTER_ERROR_MESSAGE,
    ROUTER_SYSTEM_PROMPT,
    SAFE_ERROR_MESSAGE,
)

llm = get_llm()


class AgentState(TypedDict, total=False):
    question: str
    route: str
    router_error: str
    documents: list[Document]
    answer: str
    sources: list[str]
    retrieved_chunks: int
    security_error: str
    cacheable: bool

class RouteDecision(BaseModel):
    route: Literal["rag", "direct"] = Field(
        description="The processing route: rag or direct"
    )


def validate_input(state: AgentState) -> dict:
    try:
        question = state.get("question", "").strip()

        if not question:
            return {
                "question": "",
                "security_error": INVALID_INPUT_MESSAGE,
            }

        return {
            "question": check_input(question),
            "security_error": "",
        }

    except SecurityError:
        return {"security_error": INPUT_SECURITY_ERROR_MESSAGE}
    except Exception:
        return {"security_error": SAFE_ERROR_MESSAGE}


def route_after_validation(
    state: AgentState,
) -> Literal["reject_input", "router"]:
    if state.get("security_error"):
        return "reject_input"
    return "router"


def reject_input(state: AgentState) -> dict:
    return {
        "answer": state.get("security_error") or INVALID_INPUT_MESSAGE,
        "documents": [],
        "sources": [],
        "retrieved_chunks": 0,
        "cacheable": False,
    }
def route_question(state: AgentState) -> dict:
    try:
        router = llm.with_structured_output(RouteDecision)
        decision = router.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ])

        AGENT_ROUTES.labels(route=decision.route).inc()

        return {
            "route": decision.route,
            "router_error": "",
        }

    except Exception:
        return {
            "route": "",
            "router_error": ROUTER_ERROR_MESSAGE,
        }


def route_after_classification(
    state: AgentState,
) -> Literal["rag_agent", "direct_agent", "handle_router_error"]:
    if state.get("router_error"):
        return "handle_router_error"
    if state.get("route") == "direct":
        return "direct_agent"
    if state.get("route") == "rag":
        return "rag_agent"
    return "handle_router_error"


def handle_router_error(state: AgentState) -> dict:
    return {
        "answer": state.get("router_error") or ROUTER_ERROR_MESSAGE,
        "documents": [],
        "sources": [],
        "retrieved_chunks": 0,
        "cacheable": False,
    }

def build_context(
    documents: list[Document],
) -> tuple[str, list[str]]:
    context_parts = []
    sources = []
    source_ids = {}

    for document in documents:
        content = mask_pii(document.page_content or "")
        metadata = document.metadata or {}

        source = str(
            metadata.get("source")
            or metadata.get("file_name")
            or "Unknown source"
        )

        if source not in source_ids:
            sources.append(source)
            source_ids[source] = len(sources)

        context_parts.append(
            f"[{source_ids[source]}] Source: {source}\n"
            f"Content: {content}"
        )

    return "\n\n".join(context_parts), sources


def extract_text(response) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
        )

    return str(content)


def prepare_answer(response) -> str:
    try:
        answer = extract_text(response).strip()

        if not answer:
            return EMPTY_MODEL_RESPONSE_MESSAGE

        answer = mask_pii(answer)
        check_output(answer)

        return answer

    except SecurityError:
        return OUTPUT_SECURITY_ERROR_MESSAGE
    except Exception:
        return SAFE_ERROR_MESSAGE


def rag_agent(state: AgentState) -> dict:
    question = state["question"]

    try:
        documents = search_documents(question) or []
        RETRIEVED_CHUNKS.observe(len(documents))
    except Exception:
        return {
            "answer": SAFE_ERROR_MESSAGE,
            "documents": [],
            "sources": [],
            "retrieved_chunks": 0,
            "cacheable": False,
        }

    if not documents:
        return {
            "answer": INSUFFICIENT_CONTEXT_MESSAGE,
            "documents": [],
            "sources": [],
            "retrieved_chunks": 0,
            "cacheable": False,
        }

    try:
        context, sources = build_context(documents)

        if not context.strip():
            return {
                "answer": INSUFFICIENT_CONTEXT_MESSAGE,
                "documents": documents,
                "sources": sources,
                "retrieved_chunks": len(documents),
                "cacheable": False,
            }

        human_prompt = RAG_HUMAN_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        response = llm.invoke([
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])

        answer = prepare_answer(response)

        # Не кэшируем известные безопасные сообщения об ошибках.
        cacheable = answer not in {
            SAFE_ERROR_MESSAGE,
            EMPTY_MODEL_RESPONSE_MESSAGE,
            OUTPUT_SECURITY_ERROR_MESSAGE,
        }

        return {
            "answer": answer,
            "documents": documents,
            "sources": sources,
            "retrieved_chunks": len(documents),
            "cacheable": cacheable,
        }

    except Exception:
        return {
            "answer": SAFE_ERROR_MESSAGE,
            "documents": documents,
            "sources": [],
            "retrieved_chunks": len(documents),
            "cacheable": False,
        }
        

def direct_agent(state: AgentState) -> dict:
    try:
        response = llm.invoke([
            SystemMessage(content=DIRECT_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ])

        answer = prepare_answer(response)

        cacheable = answer not in {
            SAFE_ERROR_MESSAGE,
            EMPTY_MODEL_RESPONSE_MESSAGE,
            OUTPUT_SECURITY_ERROR_MESSAGE,
        }

        return {
            "answer": answer,
            "documents": [],
            "sources": [],
            "retrieved_chunks": 0,
            "cacheable": cacheable,
        }

    except Exception:
        return {
            "answer": SAFE_ERROR_MESSAGE,
            "documents": [],
            "sources": [],
            "retrieved_chunks": 0,
            "cacheable": False,
        }

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("reject_input", reject_input)
    graph.add_node("router", route_question)
    graph.add_node("handle_router_error", handle_router_error)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("direct_agent", direct_agent)

    graph.add_edge(START, "validate_input")

    graph.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "reject_input": "reject_input",
            "router": "router",
        },
    )

    graph.add_edge("reject_input", END)

    graph.add_conditional_edges(
        "router",
        route_after_classification,
        {
            "rag_agent": "rag_agent",
            "direct_agent": "direct_agent",
            "handle_router_error": "handle_router_error",
        },
    )

    graph.add_edge("handle_router_error", END)
    graph.add_edge("rag_agent", END)
    graph.add_edge("direct_agent", END)

    return graph.compile()


agent = build_graph()
