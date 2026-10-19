import hashlib
import json
import logging
import uuid
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.services.gemini import get_llm

logger = logging.getLogger(__name__)

JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
ALL_MODES = ("classic", "late_batch", "late_wide")
LATE_MODES = {"late_batch", "late_wide"}


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    path = Path(settings.qdrant_db_path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def close_qdrant_client() -> None:
    if get_qdrant_client.cache_info().currsize:
        client = get_qdrant_client()
        get_qdrant_client.cache_clear()
        client.close()


def _get_api_key() -> str:
    key = get_settings().jina_api_key.get_secret_value().strip()

    if not key:
        raise RuntimeError("Jina API key is not configured.")

    return key


def _collection_name(mode: str) -> str:
    if mode not in ALL_MODES:
        raise ValueError(
            f"Unsupported mode: {mode}. "
            f"Allowed modes: {', '.join(ALL_MODES)}."
        )

    return f"{get_settings().qdrant_collection_name}_{mode}"


def _request_embeddings(
    texts: list[str],
    task: str,
    late_chunking: bool = False,
) -> list[list[float]]:
    if not texts:
        return []

    settings = get_settings()

    if any(len(text) > settings.max_input_length for text in texts):
        raise ValueError("An input text exceeds MAX_INPUT_LENGTH.")

    payload: dict[str, Any] = {
        "model": settings.jina_embedding_model,
        "input": texts,
        "task": task,
    }

    if late_chunking:
        payload["late_chunking"] = True

    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                JINA_EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {_get_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "Jina Embeddings request timed out."
        ) from exc

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code

        if status in (401, 403):
            message = (
                "Jina rejected the request. Check the API key "
                "and model permissions."
            )
        elif status == 429:
            message = (
                "Jina rate limit exceeded. Retry the request later."
            )
        else:
            message = f"Jina Embeddings returned HTTP {status}."

        raise RuntimeError(message) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            "Could not connect to Jina Embeddings."
        ) from exc

    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "Jina Embeddings returned an invalid response."
        ) from exc

    data = result.get("data")

    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError(
            "The number of returned vectors does not match "
            "the number of input texts."
        )

    try:
        data.sort(key=lambda item: item["index"])
        embeddings = [item["embedding"] for item in data]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Jina Embeddings returned an unexpected response format."
        ) from exc

    for vector in embeddings:
        if (
            not isinstance(vector, list)
            or not vector
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                for value in vector
            )
        ):
            raise RuntimeError(
                "Jina Embeddings returned an invalid vector."
            )

    if len({len(vector) for vector in embeddings}) != 1:
        raise RuntimeError(
            "Jina Embeddings returned vectors with inconsistent dimensions."
        )

    return embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _request_embeddings(texts, "retrieval.passage")


def embed_query(query: str) -> list[float]:
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")

    return _request_embeddings(
        [query.strip()],
        "retrieval.query",
    )[0]


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")

    if not documents:
        return []

    settings = get_settings()

    if top_n is None:
        top_n = settings.rag_final_k

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    payload = {
        "model": settings.rag_reranker_model,
        "query": query.strip(),
        "documents": [
            document.page_content for document in documents
        ],
        "top_n": min(top_n, len(documents)),
        "return_documents": False,
    }

    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                JINA_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {_get_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "Jina Reranker request timed out."
        ) from exc

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code

        if status in (401, 403):
            message = (
                "Jina rejected the reranking request. Check the API key "
                "and model permissions."
            )
        elif status == 429:
            message = (
                "Jina Reranker rate limit exceeded. Retry the request later."
            )
        else:
            message = f"Jina Reranker returned HTTP {status}."

        raise RuntimeError(message) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            "Could not connect to Jina Reranker."
        ) from exc

    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "Jina Reranker returned an invalid response."
        ) from exc

    data = result.get("results")

    if not isinstance(data, list):
        raise RuntimeError(
            "Jina Reranker response does not contain a results list."
        )

    ranked_documents = []
    seen_indices = set()

    for item in data:
        if not isinstance(item, dict):
            raise RuntimeError(
                "Jina Reranker returned an invalid result."
            )

        index = item.get("index")
        score = item.get("relevance_score")

        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(documents)
            or index in seen_indices
        ):
            raise RuntimeError(
                "Jina Reranker returned an invalid document index."
            )

        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
        ):
            raise RuntimeError(
                "Jina Reranker returned an invalid relevance score."
            )

        seen_indices.add(index)

        document = Document(
            page_content=documents[index].page_content,
            metadata=dict(documents[index].metadata),
        )
        document.metadata["rerank_score"] = float(score)
        ranked_documents.append(document)

    return ranked_documents


def _load_documents(
    documents_dir: str | Path | None = None,
) -> list[Document]:
    settings = get_settings()
    root = Path(documents_dir or settings.documents_dir).resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Documents directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {root}"
        )

    documents = []

    for path in sorted(root.rglob("*.txt")):
        if not path.is_file():
            continue

        try:
            content = path.read_text(
                encoding="utf-8-sig"
            ).strip()
        except (OSError, UnicodeError) as exc:
            logger.warning(
                "Skipping unreadable file %s: %s",
                path.name,
                type(exc).__name__,
            )
            continue

        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.relative_to(root).as_posix()
                },
            )
        )

    if not documents:
        raise ValueError(
            f"No non-empty TXT documents found in {root}."
        )

    return documents


def _chunk_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if chunk_size < 1:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
        )

    text = document.page_content.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start + chunk_size // 2:
                end = boundary

        content = text[start:end].strip()

        if content:
            metadata = dict(document.metadata)
            metadata["chunk_index"] = len(chunks)

            chunks.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        if end >= len(text):
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def _split_documents(
    documents: list[Document],
) -> list[Document]:
    settings = get_settings()
    chunks = []

    for document in documents:
        chunks.extend(
            _chunk_document(
                document,
                settings.chunk_size,
                settings.chunk_overlap,
            )
        )

    return chunks


def _split_for_mode(
    documents: list[Document],
    mode: str,
) -> list[Document]:
    settings = get_settings()

    if mode == "classic":
        return _split_documents(documents)

    if mode == "late_batch":
        chunk_size = max(1, settings.chunk_size // 2)
    elif mode == "late_wide":
        chunk_size = settings.chunk_size
    else:
        raise ValueError(f"Unsupported indexing mode: {mode}.")

    chunks = []

    for document in documents:
        chunks.extend(
            _chunk_document(document, chunk_size, 0)
        )

    return chunks


def _late_embed_chunks(
    chunks: list[Document],
) -> list[list[float]]:
    settings = get_settings()
    char_budget = settings.max_input_length

    grouped: dict[str, list[tuple[int, Document]]] = defaultdict(list)

    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown"))
        grouped[source].append((index, chunk))

    result: list[list[float] | None] = [None] * len(chunks)

    for source_chunks in grouped.values():
        batch: list[tuple[int, Document]] = []
        batch_chars = 0

        def process_batch(
            items: list[tuple[int, Document]],
        ) -> None:
            if not items:
                return

            vectors = _request_embeddings(
                [document.page_content for _, document in items],
                "retrieval.passage",
                late_chunking=True,
            )

            if len(vectors) != len(items):
                raise RuntimeError(
                    "Embedding count does not match chunk count."
                )

            for (original_index, _), vector in zip(items, vectors):
                result[original_index] = vector

        for index, chunk in source_chunks:
            length = len(chunk.page_content)

            if length > char_budget:
                raise ValueError(
                    "A chunk exceeds the maximum late-chunking input length."
                )

            if batch and batch_chars + length > char_budget:
                process_batch(batch)
                batch = []
                batch_chars = 0

            batch.append((index, chunk))
            batch_chars += length

        process_batch(batch)

    if any(vector is None for vector in result):
        raise RuntimeError(
            "Failed to generate embeddings for all chunks."
        )

    return [vector for vector in result if vector is not None]


def _make_point_id(
    source: str,
    chunk_index: int,
    content: str,
) -> str:
    digest = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{source}:{chunk_index}:{digest}",
        )
    )


def _create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
        on_disk_payload=True,
    )


def build_index(
    mode: str = "classic",
    documents_dir: str | Path | None = None,
) -> int:
    settings = get_settings()

    if mode not in ALL_MODES:
        raise ValueError(
            f"Unsupported mode: {mode}. "
            f"Allowed modes: {', '.join(ALL_MODES)}."
        )

    client = get_qdrant_client()
    collection = _collection_name(mode)
    alias = f"{collection}_active"

    documents = _load_documents(documents_dir)
    chunks = _split_for_mode(documents, mode)

    if not chunks:
        raise ValueError("Chunking produced no documents.")

    if mode in LATE_MODES:
        vectors = _late_embed_chunks(chunks)
    else:
        vectors = embed_texts(
            [chunk.page_content for chunk in chunks]
        )

    if len(vectors) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    vector_size = len(vectors[0])

    if any(len(vector) != vector_size for vector in vectors):
        raise RuntimeError("Embedding dimensions are inconsistent.")

    points = []

    for chunk, vector in zip(chunks, vectors):
        source = str(chunk.metadata.get("source", "unknown"))
        chunk_index = int(chunk.metadata.get("chunk_index", 0))

        points.append(
            models.PointStruct(
                id=_make_point_id(
                    source,
                    chunk_index,
                    chunk.page_content,
                ),
                vector=vector,
                payload={
                    "page_content": chunk.page_content,
                    "metadata": dict(chunk.metadata),
                    "source": source,
                    "chunk_index": chunk_index,
                },
            )
        )

    temporary = f"{collection}_build_{uuid.uuid4().hex[:12]}"
    _create_collection(client, temporary, vector_size)

    try:
        batch_size = max(
            1,
            int(settings.qdrant_upsert_batch_size),
        )

        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=temporary,
                points=points[start:start + batch_size],
                wait=True,
            )

        aliases = client.get_aliases().aliases
        alias_exists = any(
            item.alias_name == alias for item in aliases
        )

        operations = []

        if alias_exists:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(
                        alias_name=alias
                    )
                )
            )

        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=temporary,
                    alias_name=alias,
                )
            )
        )

        client.update_collection_aliases(
            change_aliases_operations=operations
        )

    except Exception:
        logger.exception(
            "Index build failed: mode=%s collection=%s",
            mode,
            temporary,
        )

        if client.collection_exists(temporary):
            client.delete_collection(temporary)

        raise

    logger.info(
        "Index build completed: mode=%s documents=%d chunks=%d",
        mode,
        len(documents),
        len(points),
    )

    return len(points)


def _search_one(
    query: str,
    limit: int,
    mode: str,
) -> list[Document]:
    if mode not in ALL_MODES:
        raise ValueError(f"Unsupported search mode: {mode}")

    if limit <= 0:
        return []

    client = get_qdrant_client()
    collection = f"{_collection_name(mode)}_active"

    if not client.collection_exists(collection):
        logger.warning(
            "Active index is missing for mode=%s.",
            mode,
        )
        return []

    vector = embed_query(query)

    try:
        response = client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
    except Exception:
        logger.exception(
            "Qdrant search failed: mode=%s",
            mode,
        )
        raise

    documents = []

    for point in response.points:
        payload = point.payload or {}
        content = payload.get("page_content")

        if not isinstance(content, str) or not content.strip():
            continue

        metadata = payload.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        metadata = dict(metadata)
        metadata["source"] = payload.get(
            "source",
            metadata.get("source", "unknown"),
        )
        metadata["chunk_index"] = payload.get(
            "chunk_index",
            metadata.get("chunk_index", 0),
        )
        metadata["score"] = float(point.score)

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


def _generate_query_variants(question: str) -> list[str]:
    settings = get_settings()

    if not settings.rag_multi_query_enabled:
        return [question]

    count = max(0, int(settings.rag_multi_query_count))

    if count == 0:
        return [question]

    prompt = (
        "Generate alternative search queries for retrieving information "
        "from a document collection. Do not answer the question. "
        "Preserve its original intent, key entities, and language. "
        "Return only a valid JSON array of strings without Markdown. "
        f"Generate {count} additional queries.\n"
        f"Original question: {question}"
    )

    try:
        response = get_llm().invoke(prompt)
        content = getattr(response, "content", response)

        if isinstance(content, list):
            parts = []

            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif (
                    isinstance(block, dict)
                    and isinstance(block.get("text"), str)
                ):
                    parts.append(block["text"])

            content = "\n".join(parts)

        if not isinstance(content, str):
            logger.warning(
                "Query expansion returned an unexpected response type."
            )
            return [question]

        content = content.strip()

        if content.startswith("```"):
            lines = content.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            content = "\n".join(lines).strip()

        parsed = json.loads(content)

        if not isinstance(parsed, list):
            logger.warning(
                "Query expansion returned a non-list JSON value."
            )
            return [question]

        variants = [question]
        seen = {question.casefold()}

        for item in parsed:
            if not isinstance(item, str):
                continue

            item = item.strip()

            if not item or item.casefold() in seen:
                continue

            variants.append(item)
            seen.add(item.casefold())

            if len(variants) >= count + 1:
                break

        return variants

    except Exception:
        logger.warning(
            "Query expansion failed; using the original query.",
            exc_info=True,
        )
        return [question]


def search_documents(
    question: str,
    mode: str = "classic",
    limit: int | None = None,
    *,
    min_score: float | None = None,
    max_per_source: int | None = None,
) -> list[Document]:
    settings = get_settings()

    if not question or not question.strip():
        raise ValueError("Search query must not be empty.")

    if mode not in ALL_MODES:
        raise ValueError(
            f"Unsupported search mode: {mode}. "
            f"Allowed modes: {', '.join(ALL_MODES)}."
        )

    question = question.strip()
    final_k = limit if limit is not None else settings.rag_final_k

    if final_k < 1:
        raise ValueError("The result limit must be greater than zero.")

    candidate_k = max(settings.rag_candidate_k, final_k)

    if min_score is None:
        min_score = settings.rag_min_relevance_score

    if max_per_source is None:
        max_per_source = settings.rag_max_chunks_per_source

    if max_per_source is not None and max_per_source < 1:
        raise ValueError(
            "RAG_MAX_CHUNKS_PER_SOURCE must be at least 1."
        )

    variants = _generate_query_variants(question)
    unique: dict[tuple[str, str], Document] = {}

    for variant in variants:
        results = _search_one(
            query=variant,
            limit=candidate_k,
            mode=mode,
        )

        for document in results:
            score = float(document.metadata.get("score", 0.0))

            if min_score is not None and score < min_score:
                continue

            source = str(
                document.metadata.get("source", "unknown")
            )
            digest = hashlib.sha256(
                document.page_content.encode("utf-8")
            ).hexdigest()

            key = (source, digest)
            existing = unique.get(key)

            if (
                existing is None
                or score
                > float(existing.metadata.get("score", 0.0))
            ):
                unique[key] = document

    ranked = sorted(
        unique.values(),
        key=lambda document: float(
            document.metadata.get("score", 0.0)
        ),
        reverse=True,
    )

    if settings.rag_reranking_enabled and ranked:
        ranked = rerank_documents(
            query=question,
            documents=ranked,
            top_n=min(candidate_k, len(ranked)),
        )

    if max_per_source is not None:
        counts: dict[str, int] = defaultdict(int)
        filtered = []

        for document in ranked:
            source = str(
                document.metadata.get("source", "unknown")
            )

            if counts[source] >= max_per_source:
                continue

            filtered.append(document)
            counts[source] += 1

        ranked = filtered

    return ranked[:final_k]


def load_evaluation_cases(
    dataset_path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(dataset_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read evaluation dataset: {path}"
        ) from exc

    if not isinstance(cases, list) or not cases:
        raise ValueError(
            "Evaluation dataset must be a non-empty JSON array."
        )

    validated = []

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(
                f"Evaluation case {index + 1} must be a JSON object."
            )

        query = case.get("query")
        sources = case.get("relevant_sources")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Evaluation case {index + 1} has an invalid query."
            )

        if (
            not isinstance(sources, list)
            or not sources
            or any(
                not isinstance(source, str) or not source.strip()
                for source in sources
            )
        ):
            raise ValueError(
                f"Evaluation case {index + 1} has invalid relevant_sources."
            )

        validated.append(
            {
                "query": query.strip(),
                "relevant_sources": list(
                    dict.fromkeys(
                        source.strip().replace("\\", "/")
                        for source in sources
                    )
                ),
            }
        )

    return validated


def evaluate_retrieval(
    cases: list[dict[str, Any]],
    modes: tuple[str, ...] = ALL_MODES,
    k: int = 5,
) -> dict[str, Any]:
    import math

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not cases:
        raise ValueError("Evaluation dataset is empty.")

    for mode in modes:
        if mode not in ALL_MODES:
            raise ValueError(f"Unsupported evaluation mode: {mode}.")

    report: dict[str, Any] = {
        "k": k,
        "query_count": len(cases),
        "modes": {},
    }

    for mode in modes:
        recalls = []
        precisions = []
        reciprocal_ranks = []
        ndcgs = []

        for case in cases:
            relevant = {
                source.replace("\\", "/").casefold()
                for source in case["relevant_sources"]
            }

            results = search_documents(
                question=case["query"],
                mode=mode,
                limit=k,
            )

            retrieved_sources = []
            seen_sources = set()

            for document in results:
                source = str(
                    document.metadata.get("source", "unknown")
                ).replace("\\", "/").casefold()

                if source in seen_sources:
                    continue

                seen_sources.add(source)
                retrieved_sources.append(source)

                if len(retrieved_sources) >= k:
                    break

            hits = [
                1 if source in relevant else 0
                for source in retrieved_sources
            ]

            hit_count = sum(hits)
            recalls.append(hit_count / len(relevant))
            precisions.append(hit_count / k)

            reciprocal_rank = 0.0

            for rank, hit in enumerate(hits, start=1):
                if hit:
                    reciprocal_rank = 1.0 / rank
                    break

            reciprocal_ranks.append(reciprocal_rank)

            dcg = sum(
                hit / math.log2(rank + 1)
                for rank, hit in enumerate(hits, start=1)
            )

            ideal_hits = min(len(relevant), k)
            idcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, ideal_hits + 1)
            )

            ndcgs.append(dcg / idcg if idcg else 0.0)

        query_count = len(cases)

        report["modes"][mode] = {
            f"Recall@{k}": round(
                sum(recalls) / query_count, 4
            ),
            f"Precision@{k}": round(
                sum(precisions) / query_count, 4
            ),
            f"MRR@{k}": round(
                sum(reciprocal_ranks) / query_count, 4
            ),
            f"nDCG@{k}": round(
                sum(ndcgs) / query_count, 4
            ),
        }

    return report