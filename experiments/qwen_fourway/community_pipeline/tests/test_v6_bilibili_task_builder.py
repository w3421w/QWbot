import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bilibili_task_builder", HERE / "27_v6_bilibili_task_builder.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class TestBilibiliTaskBuilder(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_builder(self, records):
        source = self.root / "local_video_list.jsonl"
        source.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
        return builder.build_video_tasks(source, self.root / "video_tasks.jsonl", self.root / "task_report.json", self.root / "task_errors.jsonl")

    def read_tasks(self):
        return [json.loads(line) for line in (self.root / "video_tasks.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_task_generation(self):
        task = {"oid": "BV123", "title": "标题", "category": "lol", "tags": ["比赛", "英雄讨论"]}
        report = self.run_builder([dict(task, extra="ignored"), {"oid": 7}])
        self.assertEqual(self.read_tasks(), [task, {"oid": "7", "title": "", "category": "", "tags": []}])
        self.assertEqual(report["valid_tasks"], 2)
        self.assertEqual(json.loads((self.root / "task_report.json").read_text()), report)
        self.assertEqual((self.root / "task_errors.jsonl").read_text(), "")

    def test_duplicate_filter(self):
        report = self.run_builder([{"oid": "BV1", "title": "first"}, {"oid": " BV1 ", "title": "second"}, {"oid": "BV2"}])
        self.assertEqual(report["duplicate_count"], 1)
        self.assertEqual(report["total_input"], 3)
        self.assertEqual([task["oid"] for task in self.read_tasks()], ["BV1", "BV2"])
        self.assertEqual(self.read_tasks()[0]["title"], "first")

    def test_error_records(self):
        source = self.root / "local_video_list.jsonl"
        source.write_text('\n{bad}\n{}\n{"oid":" "}\nnull\n{"oid":true}\n{"oid":"BV1","tags":"bad"}\n{"oid":"BV1"}\n', encoding="utf-8")
        report = builder.build_video_tasks(source, self.root / "video_tasks.jsonl", self.root / "task_report.json", self.root / "task_errors.jsonl")
        errors = [json.loads(line) for line in (self.root / "task_errors.jsonl").read_text().splitlines()]
        self.assertEqual(report["invalid_count"], 6)
        self.assertEqual(report["total_input"], 7)
        self.assertEqual(report["valid_tasks"], 1)
        self.assertEqual([error["line"] for error in errors], [2, 3, 4, 5, 6, 7])
        self.assertTrue(all(error["error"] for error in errors))

    def test_category_distribution(self):
        report = self.run_builder([{"oid": "BV1", "category": "lol"}, {"oid": "BV2", "category": "lol"}, {"oid": "BV3", "category": "游戏"}, {"oid": "BV1", "category": "其他"}, {"category": "其他"}])
        self.assertEqual(report["category_distribution"], {"lol": 2, "游戏": 1})
        self.assertEqual(report["total_input"], report["valid_tasks"] + report["duplicate_count"] + report["invalid_count"])

    def test_empty_input(self):
        report = self.run_builder([])
        self.assertEqual(report, {"total_input": 0, "valid_tasks": 0, "duplicate_count": 0, "invalid_count": 0, "category_distribution": {}})
        self.assertEqual(self.read_tasks(), [])


if __name__ == "__main__":
    unittest.main()
