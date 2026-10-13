import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("ranker", Path(__file__).with_name("31_v6_dataset_ranker.py"))
ranker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ranker)


class DatasetRankerTests(unittest.TestCase):
    def test_scores_all_requested_rules(self):
        rows = [{"file": "抗吧/community.jsonl", "format": "jsonl", "size": 300000,
                 "observed_fields": ["messages[].content", "author"]},
                {"file": "small.txt", "format": "txt", "size": 1}]
        result = ranker.rank_datasets(rows)[0]
        self.assertEqual(result["score"], 24)
        self.assertEqual(result["suggested_role"], "finetune")
        self.assertEqual(len(result["reason"]), 6)

    def test_sort_and_report_top_twenty(self):
        rows = [{"file": f"f{i}.jsonl", "format": "jsonl", "size": i} for i in range(25)]
        ranked = ranker.rank_datasets(rows)
        self.assertEqual(len(ranked), 25)
        self.assertEqual(ranked[0]["file"], "f0.jsonl")  # ties break by path
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            ranker.write_report(ranked, out / "candidate.json", out / "top.md")
            self.assertEqual(len(json.loads((out / "candidate.json").read_text())), 25)
            self.assertEqual((out / "top.md").read_text().count("| `"), 20)


if __name__ == "__main__":
    unittest.main()
