import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collection_import", HERE / "25_v6_bilibili_collection_import.py")
collection_import = importlib.util.module_from_spec(spec); spec.loader.exec_module(collection_import)


class TestBilibiliCollectionImport(unittest.TestCase):
    def test_imports_successes_and_records_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "collector_input"; source.mkdir()
            (source / "ok.json").write_text(json.dumps({"oid": 123, "title": "主题", "comments": [
                {"user": " A ", "text": " hello ", "reply_to": "root"}, {"user": "B", "text": "world"}
            ]}), encoding="utf-8")
            (source / "bad.json").write_text("{broken", encoding="utf-8")
            output, errors, report = root / "out.jsonl", root / "errors.jsonl", root / "report.json"
            result = collection_import.import_collection(source, output, errors, report)
            self.assertEqual(result, {"files_total": 2, "files_success": 1, "files_failed": 1, "comments_total": 2})
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["source_type"], "bilibili_reply_chain")
            self.assertEqual(row["data"]["topic_hint"], "主题")
            self.assertEqual(row["data"]["comments"][0]["reply_to"], "root")
            self.assertEqual(len(errors.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8")), result)

    def test_nested_json_and_invalid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"; (source / "nested").mkdir(parents=True)
            (source / "nested" / "empty.json").write_text(json.dumps({"oid": "x", "comments": []}), encoding="utf-8")
            result = collection_import.import_collection(source, Path(directory) / "out", Path(directory) / "err", Path(directory) / "rep")
            self.assertEqual((result["files_total"], result["files_success"], result["files_failed"]), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
