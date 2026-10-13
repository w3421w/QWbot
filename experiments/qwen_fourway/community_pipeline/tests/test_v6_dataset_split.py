import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dataset_split", P / "22_v6_dataset_split.py")
dataset_split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_split)


def sample(index):
    return {"messages": [{"role": "user", "content": f"问题{index}"},
                         {"role": "assistant", "content": "回答"}],
            "metadata": {"id": str(index), "source": f"src{index % 2}",
                         "community_hint": f"c{index % 2}",
                         "style_tags": [f"style{index % 2}", "shared"]}}


class V6DatasetSplitTests(unittest.TestCase):
    def run_split(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            report = dataset_split.split_dataset(source, root, **kwargs)
            files = {path.name: path.read_text(encoding="utf-8") for path in root.iterdir() if path != source}
            groups = {name: [json.loads(line) for line in files[name + ".jsonl"].splitlines()]
                      for name in dataset_split.SPLITS}
            return report, groups, files

    def test_duplicates_and_role_order(self):
        first = sample(0)
        duplicate = sample(0)
        duplicate["metadata"]["id"] = "other"
        reordered = sample(0)
        reordered["messages"].reverse()
        different_role = sample(0)
        different_role["messages"][0]["role"] = "system"
        report, groups, files = self.run_split([first, duplicate, reordered, different_role])
        self.assertEqual(report["total"], 3)
        self.assertEqual(json.loads(files["duplicate_report.json"]),
                         {"total_input": 4, "unique_count": 3, "duplicate_count": 1})
        self.assertIn(first, sum(groups.values(), []))
        self.assertNotIn(duplicate, sum(groups.values(), []))

    def test_ratios_and_no_overlap(self):
        rows = [sample(i) for i in range(100)]
        for ratios, counts in (((0.9, 0.05, 0.05), (90, 5, 5)), ((0.6, 0.2, 0.2), (60, 20, 20)), ((1, 0, 0), (100, 0, 0))):
            with self.subTest(ratios=ratios):
                report, groups, _ = self.run_split(rows, train_ratio=ratios[0], valid_ratio=ratios[1], test_ratio=ratios[2])
                self.assertEqual(tuple(report[f"{name}_count"] for name in dataset_split.SPLITS), counts)
                self.assertCountEqual(sum(groups.values(), []), rows)
        report, _, _ = self.run_split([sample(i) for i in range(11)])
        self.assertEqual((report["train_count"], report["validation_count"], report["test_count"]), (10, 1, 0))

    def test_seed_reproducible(self):
        rows = [sample(i) for i in range(100)]
        first = self.run_split(rows, seed=7)[2]
        self.assertEqual(first, self.run_split(rows, seed=7)[2])
        self.assertNotEqual(first["train.jsonl"], self.run_split(rows, seed=8)[2]["train.jsonl"])

    def test_metadata_and_extra_fields_preserved(self):
        rows = [sample(i) for i in range(20)]
        rows[0]["extra"] = {"nested": [1, "中文"]}
        rows[0]["metadata"]["custom"] = True
        _, groups, _ = self.run_split(rows)
        self.assertCountEqual(sum(groups.values(), []), rows)

    def test_invalid_input_recorded_and_processing_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text('not-json\n\n[]\n{"messages": []}\n' + json.dumps(sample(1)) + '\n', encoding="utf-8")
            report = dataset_split.split_dataset(source, root)
            errors = [json.loads(line) for line in (root / "errors.jsonl").read_text().splitlines()]
            self.assertEqual(report["total"], 1)
            self.assertEqual([error["line_number"] for error in errors], [1, 2, 3, 4])
            self.assertTrue(all(error["error"] for error in errors))
            self.assertEqual(json.loads((root / "duplicate_report.json").read_text())["total_input"], 5)

    def test_multilabel_stratification_and_report(self):
        report, groups, files = self.run_split([sample(i) for i in range(200)])
        self.assertEqual(json.loads(files["split_report.json"]), report)
        for name, count in (("train", 90), ("validation", 5), ("test", 5)):
            self.assertEqual(report["source_distribution"][name], {"src0": count, "src1": count})
            self.assertEqual(report["community_distribution"][name], {"c0": count, "c1": count})
            self.assertEqual(report["style_distribution"][name], {"shared": count * 2, "style0": count, "style1": count})
            self.assertEqual(len(groups[name]), count * 2)

    def test_empty_missing_metadata_and_invalid_ratios(self):
        report, _, _ = self.run_split([])
        self.assertEqual(report["total"], 0)
        row = sample(0)
        del row["metadata"]
        report, _, _ = self.run_split([row])
        self.assertEqual(report["source_distribution"]["train"], {"__missing__": 1})
        for ratios in ((0.8, 0.1, 0.2), (-0.1, 0.6, 0.5), (float("nan"), 0, 1)):
            with self.assertRaises(ValueError):
                self.run_split([], train_ratio=ratios[0], valid_ratio=ratios[1], test_ratio=ratios[2])

    def test_cli_reproducible_across_hash_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text("".join(json.dumps(sample(i)) + "\n" for i in range(100)), encoding="utf-8")
            outputs = []
            for hash_seed in ("1", "99"):
                dest = root / hash_seed
                result = subprocess.run([sys.executable, str(P / "22_v6_dataset_split.py"), str(source),
                                         "--output-dir", str(dest), "--seed", "42"],
                                        env={**os.environ, "PYTHONHASHSEED": hash_seed}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append({path.name: path.read_bytes() for path in dest.iterdir()})
            self.assertEqual(*outputs)


if __name__ == "__main__":
    unittest.main()
