import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("taxonomy", P / "15_v5_taxonomy_batch.py")
taxonomy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(taxonomy)


class TaxonomyBatchTests(unittest.TestCase):
    def _row(self, **overrides):
        row = {"id": "chime:1", "origin_category": "game", "origin_subcategory": None,
               "specific_source": None, "related_categories": [], "usage_communities": [],
               "expression_styles": [], "origin_evidence": "", "origin_verification": "unknown",
               "confidence": 0, "needs_manual_review": True, "trainable": False}
        row.update(overrides)
        return row

    def test_unknown_expression_style_value_is_allowed_and_recorded(self):
        row = self._row(expression_styles=["new_style"])
        taxonomy.validate([{"id": "chime:1"}], [row])
        self.assertEqual(taxonomy.schema_candidates([row], "032")["expression_styles.values.new_style"]["first_seen_batch"], "032")

    def test_unknown_usage_community_value_is_allowed(self):
        row = self._row(usage_communities=["new_community"])
        taxonomy.validate([{"id": "chime:1"}], [row])

    def test_missing_core_field_still_fails(self):
        row = self._row()
        del row["confidence"]
        with self.assertRaises(ValueError):
            taxonomy.validate([{"id": "chime:1"}], [row])

    def test_validate_new_schema_and_allows_unknown_fields(self):
        item = {"id": "chime:1", "origin_category": "game", "origin_subcategory": "RPG",
                "specific_source": None, "related_categories": ["internet_language"],
                "usage_communities": ["internet_general"], "expression_styles": ["slang"],
                "origin_evidence": "未提供具体作品", "origin_verification": "unknown",
                "confidence": 0.2, "needs_manual_review": True, "trainable": False}
        taxonomy.validate([{"id": "chime:1"}], [item])
        extra = dict(item, qwb_usefulness="low", new_taxonomy_field={"x": 1})
        taxonomy.validate([{"id": "chime:1"}], [extra])

    def test_schema_candidates_records_unknown_fields(self):
        row = {"id": "chime:1", "origin_category": "unknown", "origin_subcategory": None,
               "specific_source": None, "related_categories": [], "usage_communities": ["unknown"],
               "expression_styles": ["unknown"], "origin_evidence": "", "origin_verification": "unknown",
               "confidence": 0, "needs_manual_review": True, "trainable": False,
               "new_field": "发现值"}
        taxonomy.validate([{"id": "chime:1"}], [row])
        self.assertEqual(taxonomy.schema_candidates([row], "007"), {
            "new_field": {"count": 1, "examples": ["发现值"], "first_seen_batch": "007"}
        })

    def test_prepare_excludes_meme_bench_and_uses_25(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "source"; run = root / "run"
            source.mkdir()
            chime = [{"source_row_index": i, "fields": {"meme": f"梗{i}", "meaning": "含义"},
                      "review_status": "unreviewed"} for i in range(26)]
            bench = [{"source_row_index": 0, "fields": {"term": "评测梗"}, "data_role": "benchmark_reference_only"}]
            for name, rows in (("chime_review.jsonl", chime), ("meme_bench_review.jsonl", bench)):
                (source / name).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
            taxonomy.prepare(source, run)
            self.assertEqual(len(taxonomy.read_jsonl(run / "batches" / "001" / "input.jsonl")), 25)
            self.assertEqual(len(taxonomy.read_jsonl(run / "batches" / "002" / "input.jsonl")), 1)
            holdout = taxonomy.read_jsonl(run / "evaluation_holdout.jsonl")
            self.assertEqual(holdout[0]["evaluation_holdout"], True)
            self.assertEqual(json.loads((run / "manifest.json").read_text())["candidate_source"], "chime")

    def test_unknown_source_is_allowed_only_with_nullable_specific_source(self):
        row = {"id": "chime:1", "origin_category": "unknown", "origin_subcategory": None,
               "specific_source": None, "related_categories": [], "usage_communities": ["unknown"],
               "expression_styles": ["unknown"], "origin_evidence": "", "origin_verification": "unknown",
               "confidence": 0, "needs_manual_review": True, "trainable": False}
        taxonomy.validate([{"id": "chime:1"}], [row])
        with self.assertRaises(ValueError):
            taxonomy.validate([{"id": "chime:1"}], [dict(row, specific_source="猜测来源")])

    def test_completed_supports_resume_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            taxonomy.write_jsonl(folder / "input.jsonl", [{"id": "chime:1"}])
            self.assertIsNone(taxonomy.completed(folder))
            row = {"id": "chime:1", "origin_category": "unknown", "origin_subcategory": None,
                   "specific_source": None, "related_categories": [], "usage_communities": ["unknown"],
                   "expression_styles": ["unknown"], "origin_evidence": "", "origin_verification": "unknown",
                   "confidence": 0, "needs_manual_review": True, "trainable": False}
            taxonomy.write_jsonl(folder / "classification.jsonl", [row])
            self.assertEqual(taxonomy.completed(folder), [row])
            with self.assertRaises(ValueError):
                taxonomy.validate([{"id": "chime:1"}], [row, row])

    def test_completed_recovers_valid_attempt_classification(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            taxonomy.write_jsonl(folder / "input.jsonl", [{"id": "chime:1"}])
            row = self._row()
            attempt = folder / "attempt_2"
            attempt.mkdir()
            taxonomy.write_jsonl(attempt / "classification.jsonl", [row])
            self.assertEqual(taxonomy.completed(folder), [row])

    def test_completed_rejects_invalid_attempt_classification(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            taxonomy.write_jsonl(folder / "input.jsonl", [{"id": "chime:1"}])
            row = self._row(origin_category="not-a-category")
            attempt = folder / "attempt_1"
            attempt.mkdir()
            taxonomy.write_jsonl(attempt / "classification.jsonl", [row])
            self.assertIsNone(taxonomy.completed(folder))


if __name__ == "__main__":
    unittest.main()
