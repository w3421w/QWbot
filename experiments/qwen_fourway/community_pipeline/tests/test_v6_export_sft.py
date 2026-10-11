import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_sft", P / "21_v6_export_sft.py")
export_sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_sft)


class V6ExportSftTests(unittest.TestCase):
    def run_export(self, raw_rows, cls_rows):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("raw.jsonl", "cls.jsonl", "out.jsonl", "errors.jsonl", "report.json")]
            paths[0].write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in raw_rows) + "\n", encoding="utf-8")
            paths[1].write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in cls_rows) + "\n", encoding="utf-8")
            report = export_sft.export_sft(*paths)
            output = [json.loads(x) for x in paths[2].read_text(encoding="utf-8").splitlines()]
            errors = [json.loads(x) for x in paths[3].read_text(encoding="utf-8").splitlines()]
            return report, output, errors

    def test_two_turn_conversion_and_metadata(self):
        report, output, errors = self.run_export(
            [{"id": "a", "source": "src", "dialogue": [{"speaker": "A", "text": "问题"}, {"speaker": "B", "text": "回答"}]}],
            [{"id": "a", "data_role": "finetune", "style_tags": ["x"], "community_hint": "c"}])
        self.assertEqual(report, {"total_input": 1, "exported": 1, "skipped": 0, "error_count": 0})
        self.assertEqual(output[0]["messages"], [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}])
        self.assertEqual(output[0]["metadata"], {"id": "a", "source": "src", "style_tags": ["x"], "community_hint": "c"})
        self.assertEqual(errors, [])

    def test_multi_turn_conversion(self):
        _, output, _ = self.run_export([{"id": "a", "source": "s", "dialogue": [{"speaker": "A", "text": "1"}, {"speaker": "B", "text": "2"}, {"speaker": "A", "text": "3"}, {"speaker": "B", "text": "4"}]}], [{"id": "a", "data_role": "finetune"}])
        self.assertEqual([m["role"] for m in output[0]["messages"]], ["user", "assistant", "user", "assistant"])

    def test_non_finetune_not_exported(self):
        for role in ("knowledge", "discard", "review"):
            with self.subTest(role=role):
                report, output, _ = self.run_export([{"id": "k", "source": "s", "dialogue": [{"text": "a"}, {"text": "b"}]}], [{"id": "k", "data_role": role}])
                self.assertEqual((report["exported"], report["skipped"]), (0, 1))
                self.assertEqual(output, [])

    def test_malformed_json_counts_as_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, classified = root / "raw", root / "classified"
            raw.write_text("not-json\n", encoding="utf-8")
            classified.write_text("", encoding="utf-8")
            report = export_sft.export_sft(raw, classified, root / "out", root / "errors", root / "report")
            self.assertEqual(report, {"total_input": 1, "exported": 0, "skipped": 1, "error_count": 1})

    def test_empty_text_is_recorded(self):
        report, output, errors = self.run_export(
            [{"id": "a", "source": "s", "dialogue": [{"speaker": "A", "text": "ok"}, {"speaker": "B", "text": " "}]}],
            [{"id": "a", "data_role": "finetune"}])
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(output, [])
        self.assertEqual(errors[0]["id"], "a")

    def test_invalid_data_is_recorded(self):
        report, output, errors = self.run_export([{"id": "bad", "source": "s", "dialogue": [{"speaker": "A", "text": "only"}]}], [{"id": "bad", "data_role": "finetune"}])
        self.assertEqual((report["exported"], report["error_count"]), (0, 1)); self.assertEqual(len(errors), 1); self.assertEqual(output, [])


if __name__ == "__main__":
    unittest.main()
