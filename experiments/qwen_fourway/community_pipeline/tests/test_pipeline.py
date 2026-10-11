import csv,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(P))
from common.core import parse_export,privacy,write_new,assign_group_splits
from common.classifier import classify

def load(name):
 spec=importlib.util.spec_from_file_location(name,P/name); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

class PipelineTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=P/'output_community_pilot'); self.addCleanup(self.tmp.cleanup)
  self.path=Path(self.tmp.name)/'fixture.jsonl'
 def parse(self,items,platform='bilibili'):
  self.path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items),encoding='utf-8')
  return parse_export(self.path,platform,synthetic=True)
 def test_tree_branches_and_speakers(self):
  rows,stats=self.parse([{'id':'1','text':'LOL','video_id':'v1','author_id':'a','order':0},{'id':'2','text':'CS2','video_id':'v1','parent_id':'1','root_id':'1','author_id':'b','order':1},{'id':'3','text':'回复二楼','video_id':'v1','parent_id':'2','root_id':'1','author_id':'a','order':2},{'id':'4','text':'另一个分支','video_id':'v1','parent_id':'1','author_id':'c','order':3},{'id':'5','text':'缺父','video_id':'v1','parent_id':'lost'}])
  self.assertEqual(stats['retained'],5); self.assertIsNone(rows[0]['parent_comment_id']); self.assertEqual(rows[2]['parent_comment_id'],rows[1]['message_id']); self.assertEqual(rows[3]['parent_comment_id'],rows[0]['message_id']); self.assertIn('parent_out_of_scope',rows[4]['quality_flags']); self.assertEqual([r['message_order'] for r in rows],[0,1,2,3,None]); self.assertEqual(rows[0]['speaker_id'],rows[2]['speaker_id']); self.assertNotEqual(rows[0]['speaker_id'],rows[1]['speaker_id']); self.assertNotIn('nickname',rows[0]['speaker_id'])
 def test_cross_platform_and_content_types(self):
  b,_=self.parse([{'id':'1','text':'弹幕','video_id':'v1','content_type':'danmaku'},{'id':'2','text':'直播','video_id':'v1','content_type':'livestream_chat'}]); x,_=self.parse([{'id':'1','text':'日常','post_id':'p1'}],'xiaohongshu')
  self.assertNotEqual(b[0]['message_id'],x[0]['message_id']); self.assertEqual([r['content_type'] for r in b],['danmaku','livestream_chat']); self.assertEqual(x[0]['content_type'],'comment')
 def test_duplicate_and_errors_conservation(self):
  rows,stats=self.parse([{'id':'1','text':'A','video_id':'v'},{'id':'1','text':'B','video_id':'v'},{'text':'C','video_id':'v'},{'id':'4','video_id':'v'}]); self.assertEqual(stats['raw_records'],4); self.assertEqual(stats['filtered'],3); self.assertEqual(stats['retained'],1); self.assertEqual(stats['raw_records'],stats['retained']+stats['filtered'])
 def test_anonymization_stable_and_trainable(self):
  rows,_=self.parse([{'id':'1','text':'hi','video_id':'v','author_id':'nickname'}]); again,_=self.parse([{'id':'1','text':'hi','video_id':'v','author_id':'nickname'}]); self.assertEqual(rows[0]['speaker_id'],again[0]['speaker_id']); self.assertFalse(rows[0]['trainable']); self.assertTrue(rows[0]['is_synthetic'])
 def test_topics_and_ambiguity(self):
  cases={'LOL':'league_of_legends','CS2':'counter_strike','CSGO老比赛':'counter_strike','VALORANT':'valorant','炉石传说':'hearthstone','酒馆战棋':'hearthstone','DOTA2':'dota2','NBA总决赛':'nba','英超足球':'football','VUP直播切片':'cn_vup','Hololive VTuber':'jp_vtuber','虚拟偶像':'virtual_idol'}
  for text,topic in cases.items():
   with self.subTest(text=text): self.assertIn(topic,classify(text)['topics'])
  self.assertEqual(classify('VT')['topics'],['unknown']); self.assertNotIn('valorant',classify('VTuber')['topics']); self.assertNotIn('hearthstone',classify('云顶之弈战棋')['topics']); self.assertNotIn('football',classify('美式橄榄球')['topics']); self.assertEqual(classify('真人主播')['topics'],['unknown']); self.assertEqual(classify('LAL')['topics'],['unknown'])
 def test_multi_label_context_manual(self):
  c=classify('VUP与CS2、VALORANT、炉石、DOTA2联动'); self.assertIn('virtual_streamer',c['domain']); self.assertIn('esports',c['domain']); self.assertEqual(len(set(c['topics']) & {'counter_strike','valorant','hearthstone','dota2'}),4); self.assertIn('gaming_collaboration',c['subtopics']); self.assertIn('valorant',classify('这个游戏好玩','VALORANT赛事')['topics']); self.assertEqual(classify('VT',manual={'topics':['cn_vup'],'domain':['virtual_streamer']})['classification_confidence'],1)
 def test_privacy_review(self):
  safe,flags=privacy('手机号13812345678 邮箱a@b.com 中之人是张三'); self.assertNotIn('13812345678',safe); self.assertIn('private_identity',flags)
  rows,_=self.parse([{'id':'1','text':'微信号 abcdef12','video_id':'v','author_id':'nickname'}]); out=Path(self.tmp.name)/'review.csv'; load('07_export_review.py').export(rows,out); content=out.read_text(encoding='utf-8-sig'); self.assertNotIn('nickname',content); self.assertNotIn('abcdef12',content)
 def test_audit_and_blind_isolation(self):
  rows,stats=self.parse([{'id':'1','text':'一条不同的消息','video_id':'v'},{'id':'2','text':'回复','video_id':'v','parent_id':'1','author_id':'b'}]); audit=load('03_validate_community.py').audit(rows,stats); self.assertEqual(audit['status'],'synthetic_fixture'); self.assertTrue(audit['count_conserved']); self.assertEqual(audit['real_reply_relations'],1); self.assertEqual(audit['trainable_true'],0)
  dup,blind=load('06_deduplicate.py').inspect(rows,['一条不同的消息']); self.assertEqual(blind['exact'],1); self.assertFalse(dup)
 def test_group_split_stays_together(self):
  rows,_=self.parse([{'id':'1','text':'一','video_id':'v'},{'id':'2','text':'二','video_id':'v'}]); assign_group_splits(rows,.5); self.assertEqual(rows[0]['split'],rows[1]['split']); self.assertFalse(any(r['trainable'] for r in rows))
 def test_dedup_candidates_and_cross_split_audit(self):
  rows,_=self.parse([{'id':'1','text':'这是一段足够长的转载文本甲','video_id':'v1'},{'id':'2','text':'这是一段足够长的转载文本甲','video_id':'v2'}]); results,_=load('06_deduplicate.py').inspect(rows); self.assertTrue(any(x['type']=='cross_thread_exact' for x in results)); rows[0]['split']='train'; rows[1]['split']='validation'; rows[1]['source_group_id']=rows[0]['source_group_id']; self.assertEqual(load('03_validate_community.py').audit(rows)['cross_split_source_groups'],1)
 def test_review_quarantines_private_identity(self):
  rows,_=self.parse([{'id':'1','text':'中之人是某私人姓名','video_id':'v'}]); out=Path(self.tmp.name)/'private.csv'; load('07_export_review.py').export(rows,out); self.assertNotIn('某私人姓名',out.read_text(encoding='utf-8-sig'))
 def test_no_overwrite(self):
  path=Path(self.tmp.name)/'out.json'; write_new(path,{}); self.assertRaises(FileExistsError,write_new,path,{})

if __name__=='__main__': unittest.main()
