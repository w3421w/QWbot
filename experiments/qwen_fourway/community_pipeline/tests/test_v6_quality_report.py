import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quality", P / "20_v6_quality_report.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class V6QualityReportTests(unittest.TestCase):
    def write(self, directory, name, rows):
        path = Path(directory) / name
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_basic_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            raw = self.write(d, "raw.jsonl", [{"dialogue": [{"speaker": "a", "text": "你好"}, {"speaker": "b", "text": "世界"}]}, {"dialogue": []}])
            cls = self.write(d, "cls.jsonl", [])
            report = quality.generate_quality_report(raw, cls, Path(d) / "report.json")
            self.assertEqual(report["dataset_summary"]["total_records"], 2)
            self.assertEqual(report["dataset_summary"]["valid_records"], 1)
            self.assertEqual(report["dataset_summary"]["empty_records"], 1)
            self.assertEqual(report["dataset_summary"]["average_speaker_count"], 2)
            self.assertEqual(report["dataset_summary"]["average_rounds"], 2)
            self.assertEqual(report["dataset_summary"]["average_text_length"], 4)

    def test_role_style_community_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            raw = self.write(d, "raw.jsonl", [])
            cls = self.write(d, "cls.jsonl", [{"data_role": "finetune", "style_tags": ["roasting", "sarcasm"], "community_hint": "bilibili"}, {"data_role": "review", "style_tags": ["roasting"], "community_hint": "bilibili"}])
            report = quality.generate_quality_report(raw, cls, Path(d) / "report.json")
            self.assertEqual(report["role_distribution"], {"finetune": 1, "knowledge": 0, "discard": 0, "review": 1})
            self.assertEqual(report["style_distribution"], {"roasting": 2, "sarcasm": 1})
            self.assertEqual(report["community_distribution"], {"bilibili": 2})

    def test_malformed_input_is_skipped_and_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw.jsonl"
            raw.write_text('{"dialogue": [{"speaker": "a", "text": "ok"}]}\nnot-json\n', encoding="utf-8")
            cls = self.write(d, "cls.jsonl", [{"style_tags": "bad", "community_hint": None}])
            report = quality.generate_quality_report(raw, cls, Path(d) / "report.json")
            self.assertEqual(report["dataset_summary"]["valid_records"], 1)
            self.assertGreaterEqual(len(report["errors"]), 3)


if __name__ == "__main__":
    unittest.main()
