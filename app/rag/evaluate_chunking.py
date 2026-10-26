import argparse
import json

from app.rag.retriever import (
    build_index,
    close_qdrant_client,
    evaluate_retrieval,
    load_evaluation_cases,
)


MODES = ("classic", "late_batch", "late_wide")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to labelled JSON dataset")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for Recall@k and nDCG@k")
    parser.add_argument(
        "--build-indexes",
        action="store_true",
        help="Build all three indexes before evaluating (takes Jina API calls)",
    )
    args = parser.parse_args()

    cases = load_evaluation_cases(args.dataset)

    if args.build_indexes:
        for mode in MODES:
            count = build_index(mode=mode)
            print(f"Built {mode}: {count} chunks")

    report = evaluate_retrieval(cases, modes=MODES, k=args.k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        close_qdrant_client()