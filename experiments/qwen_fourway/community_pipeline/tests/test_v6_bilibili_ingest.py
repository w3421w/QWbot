import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ingest", HERE / "24_v6_bilibili_ingest.py")
ingest = importlib.util.module_from_spec(spec); spec.loader.exec_module(ingest)

class TestBilibiliIngest(unittest.TestCase):
    def run_ingest(self, content):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src = root / "in.jsonl"; src.write_text(content, encoding="utf-8")
            out, err, rep = root / "raw.jsonl", root / "err.jsonl", root / "report.json"
            result = ingest.ingest(src, out, err, rep)
            return result, [json.loads(x) for x in out.read_text().splitlines()], [json.loads(x) for x in err.read_text().splitlines()]

    def test_json_input(self):
        result, rows, errors = self.run_ingest(json.dumps({"oid":"1","title":"标题","comments":[{"user":"A","text":"你好"}]}))
        self.assertEqual(result["success"], 1); self.assertEqual(rows[0]["data"]["topic_hint"], "标题"); self.assertFalse(errors)

    def test_jsonl_input(self):
        result, rows, _ = self.run_ingest('{"oid":"1","comments":[{"user":"A","text":"一"}]}\n{"oid":"2","comments":[{"user":"B","text":"二"}]}\n')
        self.assertEqual(result["success"], 2); self.assertEqual(len(rows), 2)

    def test_empty_comments_filtered(self):
        result, rows, _ = self.run_ingest(json.dumps({"oid":"1","comments":[{"user":"","text":"x"},{"user":"A","text":" "},{"user":"B","text":"ok"},3]}))
        self.assertEqual(result["empty_removed"], 3); self.assertEqual(len(rows[0]["data"]["comments"]), 1)

    def test_error_record(self):
        result, _, errors = self.run_ingest('{"comments":[]}\nnot json\n')
        self.assertEqual(result["failed"], 2); self.assertEqual([e["line_number"] for e in errors], [1, 2])

    def test_report_counts(self):
        result, _, _ = self.run_ingest('{"oid":"1","comments":[{"user":"A","text":"x"}]}\n{}\n')
        self.assertEqual(result, {"total_input":2,"success":1,"failed":1,"empty_removed":0})

if __name__ == "__main__": unittest.main()
