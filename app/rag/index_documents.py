from app.rag.retriever import build_index, close_qdrant_client


if __name__ == "__main__":
    try:
        count = build_index()
        print(f"Индексация завершена. Фрагментов: {count}")
    finally:
        close_qdrant_client()