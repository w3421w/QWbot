import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch", ROOT / "01_fetch_smoothconv_json.py")
fetch = importlib.util.module_from_spec(spec); spec.loader.exec_module(fetch)

class FetchTest(unittest.TestCase):
    def test_entries_only_keeps_annotation_json(self):
        old = fetch.fetch_json
        fetch.fetch_json = lambda _: [
            {"type": "file", "path": "none_Edu/a.json", "size": 1},
            {"type": "file", "path": "none_Edu/a.wav", "size": 1},
            {"type": "file", "path": "models/config.json", "size": 1},
        ]
        try:
            self.assertEqual([x["path"] for x in fetch.entries()], ["none_Edu/a.json"])
        finally: fetch.fetch_json = old

    def test_exact_first_directory_classification(self):
        self.assertEqual(fetch.scenario_for_path("none_Edu_3ch/a.json"), "social")
        self.assertEqual(fetch.scenario_for_path("NONE_edu_x/a.json"), "social")
        self.assertEqual(fetch.scenario_for_path("edu_10_2ch/a.json"), "tutoring")
        self.assertEqual(fetch.scenario_for_path("foo/none_edu/a.json"), None)
        self.assertEqual(fetch.scenario_for_path("education/a.json"), "tutoring")

    def test_social_empty_stops_without_tutoring_fallback(self):
        old = fetch.entries; fetch.entries = lambda: [{"path": "edu_1/a.json", "size": 1, "oid": None}]
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                code = fetch.main(["--list-only"])
            self.assertNotEqual(code, 0); self.assertIn("social", err.getvalue())
        finally: fetch.entries = old

    def test_download_preserves_path_and_refuses_different_hash_and_media(self):
        blob = b'{"instances": []}'
        entries = [
            {"path": "none_Edu_3ch/nested/a.json", "size": len(blob), "oid": None},
            {"path": "none_Edu_3ch/nested/a.wav", "size": 2, "oid": None},
        ]
        old_entries, old_urlopen = fetch.entries, fetch.urlopen
        fetch.entries = lambda: entries
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return blob
        fetch.urlopen = lambda *args, **kwargs: Response()
        try:
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "out"
                self.assertEqual(fetch.main(["--download", "--output-dir", str(out)]), 0)
                dest = out / "none_Edu_3ch/nested/a.json"
                self.assertTrue(dest.exists()); self.assertFalse((out / "none_Edu_3ch/nested/a.wav").exists())
                dest.write_bytes(b"different")
                self.assertEqual(fetch.main(["--download", "--output-dir", str(out)]), 4)
        finally:
            fetch.entries, fetch.urlopen = old_entries, old_urlopen

if __name__ == "__main__": unittest.main()
