import importlib.util, json, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("parser", ROOT/"02_parse_smoothconv.py"); parser=importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
class ParserTest(unittest.TestCase):
    def test_minimal_multispeaker_overlap_and_no_reply(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"none_edu_3ch.json"
            p.write_text(json.dumps({"instances":[{"channelIndex":2,"attributes":{"speaker":"Alice"},"start":0,"end":2,"text":"你好"},{"channelIndex":3,"attributes":{"speaker":"Bob"},"start":1,"end":3,"text":"收到"},{"channelIndex":2,"attributes":{"speaker":"Alice"},"start":4,"end":5,"text":"再见"}]},ensure_ascii=False),encoding="utf-8")
            out, errors=parser.parse_file(p)
            self.assertFalse(errors); self.assertEqual(out[0]["speaker_count"],2); self.assertEqual(out[0]["messages"][1]["reply_to_id"],None); self.assertEqual(out[0]["messages"][1]["reply_relation"],"unknown"); self.assertFalse(out[0]["messages"][0]["trainable"]); self.assertEqual([m["message_order"] for m in out[0]["messages"]],[0,1,2])
if __name__ == "__main__": unittest.main()

class ParseErrorTest(unittest.TestCase):
    def test_missing_text_is_structured_error(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"bad.json"
            p.write_text(json.dumps({"instances":[{"channelIndex":1,"start":0,"end":1,"attributes":{"speaker":"A1"}}]}),encoding="utf-8")
            out, errors=parser.parse_file(p)
            self.assertEqual(out, [])
            self.assertEqual(len(errors),1)
            self.assertEqual(errors[0]["error_type"],"missing_required_field")
            self.assertEqual(errors[0]["missing_fields"],["text"])
            self.assertFalse(errors[0]["affects_speaker_id"])
            self.assertFalse(errors[0]["affects_time_order"])

    def test_same_stem_in_different_directories_has_different_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = {"instances": [{"channelIndex": 1, "attributes": {"speaker": "A"}, "start": 0, "end": 1, "text": "hi"}]}
            p1, p2 = root / "none_Edu_a" / "same.json", root / "none_Edu_b" / "same.json"
            p1.parent.mkdir(); p2.parent.mkdir()
            for p in (p1, p2): p.write_text(json.dumps(payload), encoding="utf-8")
            c1, _ = parser.parse_file(p1, root)
            c2, _ = parser.parse_file(p2, root)
            self.assertNotEqual(c1[0]["conversation_id"], c2[0]["conversation_id"])
