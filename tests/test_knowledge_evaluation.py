"""小规模开发集检索评估，不代表独立泛化性能。"""

import json
from pathlib import Path

from src.knowledge.retriever import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_PATH = (
    PROJECT_ROOT / "data/knowledge/ecg_knowledge.jsonl"
)

QUERY_PATH = (
    PROJECT_ROOT / "evaluation/knowledge_queries.jsonl"
)


def load_queries(path):
    rows = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected object")

            query_id = row.get("id")
            query = row.get("query")
            relevant_ids = row.get("relevant_ids")

            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"Line {line_number}: invalid id")

            if query_id in seen_ids:
                raise ValueError(f"Duplicate query id: {query_id}")

            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{query_id}: empty query")

            if (
                not isinstance(relevant_ids, list)
                or not relevant_ids
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in relevant_ids
                )
            ):
                raise ValueError(f"{query_id}: invalid relevant_ids")

            if len(relevant_ids) != len(set(relevant_ids)):
                raise ValueError(f"{query_id}: duplicate relevant_ids")

            seen_ids.add(query_id)
            rows.append(row)

    if not rows:
        raise ValueError("Query set is empty")

    return rows


def main():
    retriever = BM25Retriever.from_jsonl(KNOWLEDGE_PATH)
    queries = load_queries(QUERY_PATH)

    known_ids = {chunk.id for chunk in retriever.chunks}

    if not known_ids:
        raise ValueError("Knowledge corpus is empty")

    for row in queries:
        unknown = set(row["relevant_ids"]) - known_ids

        if unknown:
            raise ValueError(
                f'{row["id"]}: unknown knowledge ids {sorted(unknown)}'
            )

    top_k = 3
    recall_values = []
    reciprocal_ranks = []
    full_recall_count = 0

    print(f"Knowledge chunks: {len(known_ids)}")
    print(f"Development queries: {len(queries)}")
    print(f"Top K: {top_k}")

    for row in queries:
        hits = retriever.search(row["query"], top_k=top_k)

        retrieved_ids = [hit.chunk.id for hit in hits]
        relevant_ids = set(row["relevant_ids"])

        matched = relevant_ids.intersection(retrieved_ids)
        missing = relevant_ids - set(retrieved_ids)

        recall = len(matched) / len(relevant_ids)

        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, chunk_id in enumerate(
                    retrieved_ids,
                    start=1,
                )
                if chunk_id in relevant_ids
            ),
            0.0,
        )

        recall_values.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        if not missing:
            full_recall_count += 1

        print(f'\n{row["id"]}: {row["query"]}')

        for hit in hits:
            marker = "*" if hit.chunk.id in relevant_ids else "-"
            print(
                f"  {marker} rank={hit.rank} "
                f"id={hit.chunk.id} "
                f"score={hit.score:.4f}"
            )

        print(f"  Recall@3={recall:.3f}")
        print(f"  RR@3={reciprocal_rank:.3f}")

        if missing:
            print(f"  Missing: {sorted(missing)}")

    count = len(queries)

    print("\n========== DEVELOPMENT EVALUATION ==========")
    print(f"Mean Recall@3: {sum(recall_values) / count:.4f}")
    print(f"MRR@3: {sum(reciprocal_ranks) / count:.4f}")
    print(f"Full-recall queries: {full_recall_count}/{count}")
    print("Evaluation completed; no quality threshold is enforced.")
    print("这是小规模开发集结果，不是医学准确率或泛化能力证明。")


if __name__ == "__main__":
    main()