import importlib.util
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adapter", P / "18_v6_source_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class SourceAdapterTest(unittest.TestCase):
    def test_bilibili(self):
        row = adapter.adapt_source("bilibili_reply_chain", {"oid": "xxx", "comments": [{"user": "A", "text": "这英雄是不是寄了"}, {"user": "B", "text": "不是英雄寄，是设计师寄"}]})
        self.assertEqual(row["source"], "bilibili_reply_chain")
        self.assertEqual(len(row["dialogue"]), 2)

    def test_lccc(self):
        row = adapter.adapt_source("lccc_chat", ["你好", "你好，有什么事情"])
        self.assertEqual(row["speaker_count"], 2)
        self.assertEqual(row["dialogue"][1]["text"], "你好，有什么事情")

    def test_knowledge_role(self):
        row = adapter.adapt_source("knowledge_only", {"id": "m1", "source": "memebench", "content": "一个梗"})
        self.assertEqual(row["candidate_role"], "knowledge")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            adapter.adapt_source("lccc_chat", [])
        with self.assertRaises(ValueError):
            adapter.adapt_source("unknown", {})


if __name__ == "__main__":
    unittest.main()
