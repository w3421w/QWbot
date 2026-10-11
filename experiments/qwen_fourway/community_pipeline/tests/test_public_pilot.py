import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_pilot", P / "08_fetch_public_pilot.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class PublicPilotTests(unittest.TestCase):
    def test_allowlist_and_limit(self):
        with self.assertRaises(ValueError): mod.fetch_rows("other/dataset", 1)
        with self.assertRaises(ValueError): mod.fetch_rows("Midsummra/bilibilicomment", 101)

    def test_bilibili_adapter_does_not_invent_relationships(self):
        rows = [{"message": "炉石评论", "time": "2024", "timestamp": 1}]
        out = mod.adapt_rows("Midsummra/bilibilicomment", rows, "/tmp/pilot.jsonl")
        self.assertEqual(len(out), 1)
        r = out[0]
        self.assertIsNone(r["thread_id"])
        self.assertIsNone(r["parent_comment_id"])
        self.assertIsNone(r["speaker_id"])
        self.assertFalse(r["trainable"])
        self.assertEqual(r["adapter_kind"], "bilibili_independent_comment_candidate")
        self.assertEqual(r["source_fields"]["message"], "炉石评论")

    def test_instruction_response_is_preserved_and_not_chat(self):
        row = {"instruction": "问候", "input": "", "output": "NBA", "extra": "keep?"}
        out = mod.adapt_rows("helloollel/chat_dataset_bilibili", [row], "/tmp/pilot.jsonl", "cc-by")
        r = out[0]
        self.assertEqual((r["instruction"], r["input"], r["output"]), ("问候", "", "NBA"))
        self.assertEqual(r["relation_type"], "derived_instruction_response")
        self.assertIn("not_verified_multi_user_chat", r["quality_flags"])
        self.assertIsNone(r["thread_id"])
        self.assertFalse(r["trainable"])
        self.assertEqual(r["license_status"], "dataset_declared_unverified")

    def test_metadata_request_is_bounded_and_no_token(self):
        seen = {}
        old = mod._json_request
        def fake(url, timeout):
            seen["url"] = url; seen["timeout"] = timeout
            return ({"cardData": {"license": "mit"}, "siblings": []}, b"{}")
        mod._json_request = fake
        try:
            meta = mod.inspect_source("Midsummra/bilibilicomment", 7)
        finally:
            mod._json_request = old
        self.assertEqual(seen["timeout"], 7)
        self.assertNotIn("token", seen["url"].lower())
        self.assertEqual(meta["declared_license"], "mit")
        self.assertEqual(meta["expected_fields"], ["message", "time", "timestamp"])

    def test_cli_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=P) as d:
            path = Path(d) / "Midsummra__bilibilicomment.metadata.json"
            path.write_text("{}")
            old = mod.inspect_source
            mod.inspect_source = lambda repo, timeout: {"dataset": repo}
            try:
                with self.assertRaises(SystemExit) as ctx: mod.main(["--source", "Midsummra/bilibilicomment", "--output-dir", d])
            finally: mod.inspect_source = old
            self.assertIn("File exists", str(ctx.exception))

    def test_discover_real_config_and_preserve_preview_row_index(self):
        calls = []
        original = mod._json_request
        def fake(url, timeout):
            calls.append(url)
            if "/splits?" in url:
                return ({"splits": [{"dataset": "Midsummra/bilibilicomment", "config": "actual-config", "split": "validation"}]}, b"splits")
            return ({"rows": [{"row_idx": 42, "row": {"message": "炉石", "time": "2024"}}]}, b"rows")
        mod._json_request = fake
        try:
            rows, url, raw = mod.fetch_rows("Midsummra/bilibilicomment", 1)
            record = mod.adapt_rows("Midsummra/bilibilicomment", rows, "/tmp/example.jsonl")[0]
        finally:
            mod._json_request = original
        self.assertIn("config=actual-config", url)
        self.assertIn("split=validation", url)
        self.assertTrue(any("/splits?" in u for u in calls))
        self.assertIn("42", record["source_record_id"])
        self.assertFalse(record["trainable"])

    def test_nested_exported_text_fields_are_sanitized(self):
        phone = "13812345678"
        email = "privateperson@example.com"
        samples = [
            ("Midsummra/bilibilicomment", [{"message": "联系我" + phone, "time": "2024"}]),
            ("helloollel/chat_dataset_bilibili", [{"instruction": "邮箱" + email, "input": phone, "output": "回复" + email}]),
        ]
        for repo, rows in samples:
            result = mod.adapt_rows(repo, rows, "/tmp/example.jsonl")
            self.assertEqual(len(result), 1)
            serialized = json.dumps(result[0], ensure_ascii=False)
            self.assertNotIn(phone, serialized)
            self.assertNotIn(email, serialized)
            self.assertFalse(result[0]["trainable"])

    def test_nested_json_text_values_are_sanitized(self):
        phone = "13812345678"
        email = "privateperson@example.com"
        rows = [{
            "message": "外层",
            "time": {"note": phone},
            "timestamp": [email],
        }]
        result = mod.adapt_rows("Midsummra/bilibilicomment", rows, "/tmp/example.jsonl")
        serialized = json.dumps(result[0], ensure_ascii=False)
        self.assertNotIn(phone, serialized)
        self.assertNotIn(email, serialized)
        self.assertEqual(result[0]["source_fields"]["time"]["note"], "[REDACTED]")



if __name__ == "__main__": unittest.main()
