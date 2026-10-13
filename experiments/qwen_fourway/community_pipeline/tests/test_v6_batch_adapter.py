import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("batch_adapter", P / "18_v6_batch_adapter.py")
batch_adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch_adapter)


class BatchAdapterTest(unittest.TestCase):
    def run_batch(self, rows):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "raw_source.jsonl"
            output, errors, report = (root / name for name in ("out.jsonl", "errors.jsonl", "report.json"))
            source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            stats = batch_adapter.convert_jsonl(source, output, errors, report)
            return stats, output.read_text(encoding="utf-8").splitlines(), errors.read_text(encoding="utf-8").splitlines(), json.loads(report.read_text(encoding="utf-8"))

    def test_multiple_sources_convert(self):
        stats, output, errors, report = self.run_batch([
            {"source_type": "bilibili_reply_chain", "data": {"oid": "1", "comments": [{"user": "A", "text": "你好"}, {"user": "B", "text": "世界"}]}},
            {"source_type": "lccc_chat", "data": ["你好", "在吗"]},
            {"source_type": "knowledge_only", "data": {"id": "k1", "content": "一个梗"}},
        ])
        self.assertEqual(stats, report)
        self.assertEqual((report["total"], report["success"], report["failed"]), (3, 3, 0))
        self.assertEqual(len(output), 3)
        self.assertEqual(errors, [])

    def test_invalid_continues_and_logs_errors(self):
        stats, output, errors, report = self.run_batch([
            {"source_type": "lccc_chat", "data": ["有效", "记录"]},
            {"source_type": "lccc_chat", "data": []},
            {"source_type": "unknown", "data": {}},
        ])
        self.assertEqual(report, {"total": 3, "success": 1, "failed": 2, "source_counts": {"lccc_chat": 2, "unknown": 1}})
        self.assertEqual(len(output), 1)
        self.assertEqual(len(errors), 2)
        self.assertEqual(json.loads(errors[0])["line_number"], 2)
        self.assertEqual(json.loads(errors[1])["source_type"], "unknown")

    def test_malformed_json_is_logged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "raw.jsonl"
            source.write_text('{"source_type":"lccc_chat","data":["ok"]}\nnot-json\n', encoding="utf-8")
            out, err, report = root / "o", root / "e", root / "r"
            result = batch_adapter.convert_jsonl(source, out, err, report)
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["success"], 1)
            self.assertEqual(json.loads(err.read_text(encoding="utf-8"))["source_type"], None)


if __name__ == "__main__":
    unittest.main()
