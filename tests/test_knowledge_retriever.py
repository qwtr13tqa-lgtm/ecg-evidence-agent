import json
import tempfile
import unittest
from pathlib import Path

from src.knowledge.retriever import (
    BM25Retriever,
    KnowledgeChunk,
)


class TestKnowledgeRetriever(unittest.TestCase):

    def setUp(self):
        # 仅用于软件测试，不包含正式医学知识。
        self.chunks = [
            KnowledgeChunk(
                id="TEST_ONLY_001",
                title="心率 heart rate",
                text="虚构检索测试条目：心率 heart rate。",
                source="test-fixture://not-medical-evidence",
                locator="fixture-1",
            ),
            KnowledgeChunk(
                id="TEST_ONLY_002",
                title="RR interval",
                text="虚构检索测试条目：RR interval variability。",
                source="test-fixture://not-medical-evidence",
                locator="fixture-2",
            ),
            KnowledgeChunk(
                id="TEST_ONLY_003",
                title="模型重建 reconstruction",
                text="虚构检索测试条目：reconstruction error。",
                source="test-fixture://not-medical-evidence",
                locator="fixture-3",
            ),
        ]

        self.retriever = BM25Retriever(self.chunks)

    def test_english_retrieval(self):
        hits = self.retriever.search("RR interval")

        self.assertEqual(hits[0].chunk.id, "TEST_ONLY_002")
        self.assertEqual(hits[0].rank, 1)
        self.assertGreater(hits[0].score, 0)

    def test_chinese_retrieval(self):
        hits = self.retriever.search("心率")

        self.assertEqual(hits[0].chunk.id, "TEST_ONLY_001")

    def test_no_match(self):
        self.assertEqual(
            self.retriever.search("zzzzunknown"),
            [],
        )

    def test_empty_query(self):
        self.assertEqual(self.retriever.search(""), [])

    def test_empty_corpus(self):
        self.assertEqual(
            BM25Retriever([]).search("heart"),
            [],
        )

    def test_top_k(self):
        hits = self.retriever.search("虚构", top_k=2)

        self.assertEqual(len(hits), 2)
        self.assertEqual([hit.rank for hit in hits], [1, 2])
        self.assertGreaterEqual(hits[0].score, hits[1].score)

    def test_source_preserved(self):
        hit = self.retriever.search("RR")[0]
        result = hit.to_dict()

        self.assertEqual(
            result["source"],
            "test-fixture://not-medical-evidence",
        )
        self.assertEqual(result["locator"], "fixture-2")
        self.assertEqual(result["text"], self.chunks[1].text)

        json.dumps(result, allow_nan=False)

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            BM25Retriever([self.chunks[0], self.chunks[0]])

    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeChunk(
                id="bad",
                title="test",
                text="test",
                source="",
                locator="test",
            )

    def test_invalid_top_k(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.retriever.search("RR", top_k=value)

    def test_jsonl_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_only.jsonl"

            with path.open("w", encoding="utf-8") as handle:
                for chunk in self.chunks:
                    handle.write(
                        json.dumps(
                            chunk.to_dict(),
                            ensure_ascii=False,
                        ) + "\n"
                    )

            retriever = BM25Retriever.from_jsonl(path)
            hits = retriever.search("reconstruction")

            self.assertEqual(
                hits[0].chunk.id,
                "TEST_ONLY_003",
            )

    def test_invalid_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text('{"id": "incomplete"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1"):
                BM25Retriever.from_jsonl(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)