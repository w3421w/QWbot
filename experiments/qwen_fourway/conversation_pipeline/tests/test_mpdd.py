import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

fetch = load("04_fetch_mpdd.py")
parser = load("05_parse_mpdd.py")
validator = load("06_validate_mpdd.py")

class MPDDTest(unittest.TestCase):
    def fixture(self):
        return [
            {"id": "a", "dialogue": [
                {"speaker": "王甲", "text": "你好", "listener": "李乙", "emotion": "happy"},
                {"speaker": "李乙", "text": "", "listener": ["王甲"], "relationship": "friend"},
                {"speaker": "王甲", "text": "再见"}]},
            {"id": "b", "drama_id": "tv-1", "dialogue": [
                {"speaker": "王甲", "text": "第一句"},
                {"speaker": "李乙", "text": "第二句"},
                {"speaker": "张丙", "text": "第三句"},
                {"speaker": "赵丁", "text": "第四句"}]},
            {"id": "c", "drama_id": "tv-1", "dialogue": [
                {"speaker": "甲", "text": "第一句"},
                {"speaker": "乙", "text": "第二句"},
                {"speaker": "丙", "text": "第三句"},
                {"speaker": "丁", "text": "第四句"}]},
        ]

    def test_people_order_listener_empty_and_training(self):
        convs = parser.parse(self.fixture(), {})
        self.assertEqual([c["speaker_count"] for c in convs], [2, 4, 4])
        first = convs[0]["messages"]
        self.assertEqual([m["message_order"] for m in first], [0, 1, 2])
        self.assertEqual([m["text"] for m in first], ["你好", "", "再见"])
        self.assertEqual([m["speaker_id"] for m in first], ["speaker_01", "speaker_02", "speaker_01"])
        self.assertEqual(first[0]["listeners"], "李乙")
        self.assertIsNone(first[0]["reply_to_id"])
        self.assertEqual(first[0]["reply_relation"], "unknown")
        self.assertEqual(first[1]["listeners"], ["王甲"])
        self.assertEqual(first[1]["interpersonal_relation"], "friend")
        self.assertEqual(convs[1]["source_group_id"], convs[2]["source_group_id"])
        self.assertEqual(convs[1]["split"], convs[2]["split"])
        self.assertTrue(all(c["trainable"] is False for c in convs))
        self.assertTrue(all(m["trainable"] is False for c in convs for m in c["messages"]))

    def test_duplicate_audit(self):
        raw = self.fixture()
        raw.append({"id": "d", "dialogue": raw[0]["dialogue"]})
        convs = parser.parse(raw, {})
        stats = validator.audit([{k:v for k,v in c.items() if k != "messages"} for c in convs],
                                [m for c in convs for m in c["messages"]], [])
        self.assertEqual(stats["duplicate_conversation_count"], 2)
        self.assertEqual(stats["two_person_conversations"], 2)
        self.assertEqual(stats["four_or_more_person_conversations"], 2)
        self.assertFalse(stats["validation_errors"])

    def test_zip_safety(self):
        for name in ("../evil", "/tmp/evil", "C:\\evil"):
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, "w") as z: z.writestr(name, "bad")
            with self.assertRaises(ValueError): fetch.validate_zip_bytes(bio.getvalue())
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as z:
            z.writestr("mpdd/dialogue.json", json.dumps(self.fixture(), ensure_ascii=False))
            z.writestr("mpdd/metadata.json", "{}")
        self.assertGreater(fetch.validate_zip_bytes(bio.getvalue()), 0)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "mpdd.zip").write_bytes(bio.getvalue())
            inputs, _ = parser.load_inputs(Path(td))
            self.assertEqual(len(parser.parse(inputs["dialogue.json"], inputs["metadata.json"])), 3)

    def test_default_fetch_only_checks_source(self):
        old = fetch.inspect_source
        fetch.inspect_source = lambda: {"url": fetch.SOURCE_URL, "content_length": 1024, "status": 200}
        try:
            with tempfile.TemporaryDirectory() as td:
                self.assertEqual(fetch.main(["--output-dir", td]), 0)
                self.assertEqual(list(Path(td).iterdir()), [])
        finally: fetch.inspect_source = old

if __name__ == "__main__": unittest.main()
