import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(n):
 s=importlib.util.spec_from_file_location(n,ROOT/n); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
parser=load("08_parse_irc.py"); validator=load("09_validate_irc.py"); fetch=load("07_fetch_irc.py")
class IRCTest(unittest.TestCase):
 def test_multispeaker_links_self_start_system_and_order(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td); (d/"sample.raw.txt").write_text("a\t2020\tAlice\tstart\nb\t2020\tBob\treply\nc\t2020\tCarol\tjoin\nd\t2020\tSYSTEM\tNotice\n",encoding="utf8")
   (d/"sample.annotation.txt").write_text("reply\ta\ta\nreply\ta\tb\nreply\ta\tc\n",encoding="utf8")
   cs,ms,es=parser.parse(d); self.assertEqual([m["message_order"] for m in ms],[0,1,2,3]); self.assertEqual(len({m["speaker_id"] for m in ms}),4); self.assertEqual(ms[0]["reply_relation"],"self_start"); self.assertEqual(len(ms[1]["reply_to_ids"]),1); self.assertTrue(ms[3]["system_message"]); self.assertTrue(all(not m["trainable"] for m in ms)); self.assertEqual(len(es),0)
 def test_unresolved_and_out_of_context_count(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td); (d/"x.raw.txt").write_text("a\tA\thello\n",encoding="utf8"); (d/"x.annotation.txt").write_text("a\tb\n",encoding="utf8"); cs,ms,es=parser.parse(d); st=validator.audit(cs,ms,es,[]); self.assertEqual(st["unresolved_association_count"],1); self.assertEqual(st["quantity_conservation"],True)
 def test_allowlist(self):
  self.assertTrue(fetch.official_url(fetch.url_for("data/x.raw.txt"))); self.assertFalse(fetch.official_url("https://example.com/x.raw.txt"))
if __name__=="__main__": unittest.main()
