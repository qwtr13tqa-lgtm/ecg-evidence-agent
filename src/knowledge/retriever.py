"""无第三方依赖的 BM25 知识检索基线。

知识文件格式：JSONL，每行一个对象，字段为：
    id, title, text, source, locator

source：来源 URL 或文档标识。
locator：页码、章节等定位信息。

检索分数不是概率，也不能证明来源真实或内容权威。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Union


def tokenize(text: str) -> List[str]:
    """英文单词 + 中文单字/双字；不进行医学同义词扩展。"""
    tokens = []

    for part in re.findall(
        r"[a-z0-9]+|[\u4e00-\u9fff]+",
        text.lower(),
    ):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(part)
            tokens.extend(
                part[i:i + 2]
                for i in range(len(part) - 1)
            )
        else:
            tokens.append(part)

    return tokens


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    title: str
    text: str
    source: str
    locator: str

    def __post_init__(self):
        for name in ("id", "title", "text", "source", "locator"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} must be a non-empty string"
                )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    score: float
    chunk: KnowledgeChunk

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "retrieval_score": self.score,
            **self.chunk.to_dict(),
        }


class BM25Retriever:
    """对标题与正文建立内存索引，来源字段仅用于引用。"""

    def __init__(
        self,
        chunks: List[KnowledgeChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.chunks = tuple(chunks)
        self.k1 = k1
        self.b = b

        ids = [chunk.id for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate knowledge chunk id")

        self.term_counts = []
        self.lengths = []
        document_frequency = Counter()

        for chunk in self.chunks:
            terms = tokenize(f"{chunk.title} {chunk.text}")

            if not terms:
                raise ValueError(
                    f"Chunk {chunk.id} has no searchable terms"
                )

            counts = Counter(terms)
            self.term_counts.append(counts)
            self.lengths.append(len(terms))
            document_frequency.update(counts.keys())

        count = len(self.chunks)

        self.average_length = (
            sum(self.lengths) / count if count else 0.0
        )

        self.idf: Dict[str, float] = {
            term: math.log(
                1.0 + (count - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
    ) -> "BM25Retriever":
        chunks = []
        expected = {"id", "title", "text", "source", "locator"}

        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                    if not isinstance(record, dict):
                        raise ValueError("Expected a JSON object")

                    if set(record) != expected:
                        raise ValueError(
                            f"Expected exactly these fields: "
                            f"{sorted(expected)}"
                        )

                    chunks.append(KnowledgeChunk(**record))

                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{path}, line {line_number}: {exc}"
                    ) from exc

        return cls(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[RetrievalHit]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")

        query_terms = set(tokenize(query))

        if not query_terms or not self.chunks:
            return []

        scored = []

        for index, counts in enumerate(self.term_counts):
            length_ratio = (
                self.lengths[index] / self.average_length
            )
            normalization = self.k1 * (
                1 - self.b + self.b * length_ratio
            )

            score = 0.0

            # 固定顺序，便于复现。
            for term in sorted(query_terms):
                frequency = counts.get(term, 0)

                if frequency:
                    score += (
                        self.idf[term]
                        * frequency
                        * (self.k1 + 1)
                        / (frequency + normalization)
                    )

            # 没有词面匹配时不强行返回文档。
            if score > 0:
                scored.append((score, index))

        scored.sort(
            key=lambda item: (
                -item[0],
                self.chunks[item[1]].id,
            )
        )

        return [
            RetrievalHit(
                rank=rank,
                score=score,
                chunk=self.chunks[index],
            )
            for rank, (score, index) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]