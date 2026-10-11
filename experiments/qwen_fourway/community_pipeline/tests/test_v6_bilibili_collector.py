import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bilibili_collector", HERE / "26_v6_bilibili_collector.py")
collector = importlib.util.module_from_spec(spec); spec.loader.exec_module(collector)


class TestBilibiliCollector(unittest.TestCase):
    def test_tasks_empty_results_report_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "video_tasks.jsonl"
            tasks.write_text(json.dumps({"oid": "a1", "title": "标题"}, ensure_ascii=False) + "\n{bad}\n", encoding="utf-8")
            result = collector.run_collection_tasks(tasks, root / "collector_raw", root / "report.json", root / "errors.jsonl")
            self.assertEqual(result, {"total_tasks": 2, "success": 1, "failed": 1})
            self.assertEqual(json.loads((root / "collector_raw" / "a1.json").read_text())["comments"], [])
            self.assertEqual(json.loads((root / "report.json").read_text()), result)
            self.assertEqual(len((root / "errors.jsonl").read_text().splitlines()), 1)

    def test_collect_task_schema(self):
        self.assertEqual(collector.collect_task({"oid": 7, "title": "T"}), {"oid": "7", "title": "T", "comments": []})


if __name__ == "__main__":
    unittest.main()
