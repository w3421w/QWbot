import importlib.util
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dialogue_schema", P / "17_v6_dialogue_schema.py")
schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema)


class V6DialogueSchemaTests(unittest.TestCase):
    def row(self, **changes):
        row = {"id": "source:1", "source": "source", "dialogue": [{"speaker": "A", "text": "你好"}],
               "speaker_count": 1, "topic_hint": None, "community_hint": "bilibili", "candidate_role": "finetune"}
        row.update(changes)
        return row

    def test_valid_record_and_required_fields(self):
        schema.validate_record(self.row())
        self.assertEqual(schema.REQUIRED_FIELDS, {"id", "source", "dialogue", "speaker_count", "topic_hint", "community_hint", "candidate_role"})

    def test_candidate_role_enum(self):
        for role in ("finetune", "knowledge", "both", "unknown"):
            schema.validate_record(self.row(candidate_role=role))
        with self.assertRaises(ValueError):
            schema.validate_record(self.row(candidate_role="discard"))

    def test_types_and_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            schema.validate_record(self.row(dialogue="你好"))
        with self.assertRaises(ValueError):
            schema.validate_records([self.row(), self.row()])


if __name__ == "__main__":
    unittest.main()
