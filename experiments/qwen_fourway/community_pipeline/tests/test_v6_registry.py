import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("registry", P / "16_v6_source_registry.py")
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)


class V6RegistryTests(unittest.TestCase):
    def test_default_registry_has_required_sources_and_fields(self):
        registry.validate_registry(registry.DEFAULT_SOURCES)
        self.assertEqual({r["name"] for r in registry.DEFAULT_SOURCES},
                         {"Bilibili评论回复链", "Bilibili大规模评论数据", "LCCC中文聊天数据", "MemeBench", "CHIME"})

    def test_duplicate_ids_and_bad_enum_fail(self):
        rows = [dict(registry.DEFAULT_SOURCES[0]), dict(registry.DEFAULT_SOURCES[0])]
        with self.assertRaises(ValueError):
            registry.validate_registry(rows)
        bad = dict(registry.DEFAULT_SOURCES[0], candidate_role="discard")
        with self.assertRaises(ValueError):
            registry.validate_source(bad)

    def test_write_registry_is_json_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v6_source_registry.json"
            registry.write_registry(path)
            self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))), 5)
            with self.assertRaises(FileExistsError):
                registry.write_registry(path)


if __name__ == "__main__":
    unittest.main()
