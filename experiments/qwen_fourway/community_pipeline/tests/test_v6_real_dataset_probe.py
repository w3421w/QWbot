import json
import tempfile
import unittest
from pathlib import Path
import importlib.util

P = Path(__file__).parents[1] / "30_v6_real_dataset_probe.py"
spec = importlib.util.spec_from_file_location("probe", P); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class TestRealDatasetProbe(unittest.TestCase):
    def test_jsonl_roles(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_text(json.dumps({"user": "a", "comment": "你好", "messages": [{"text": "hi"}]}, ensure_ascii=False) + "\n")
            out = mod.probe(str(p))
            self.assertEqual(out["format"], "jsonl")
            self.assertIn("comment", out["possible_fields"]["comment"])
            self.assertTrue(out["possible_fields"]["user"])

    def test_json_array(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"; p.write_text('[{"prompt":"q","response":"a"}]')
            self.assertEqual(mod.probe(str(p))["records_sampled"], 1)
