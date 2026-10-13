import argparse
from collections import Counter,defaultdict
from common.core import read_records,write_new,normalized

def audit(rows,parse_stats=None,duplicate_report=None):
 if rows is None: return {'status':'not_measured','reason':'no real community input'}
 n=len(rows); topic=Counter(t for r in rows for t in r.get('topics',[])); domain=Counter(d for r in rows for d in r.get('domain',[])); sub=Counter(s for r in rows for s in r.get('subtopics',[]))
 platforms=Counter(r.get('platform') or 'unknown' for r in rows); threads=defaultdict(list); ids=set(); texts={}; duplicate=0; cross_split=0
 for r in rows:
  if r.get('thread_id') is not None: threads[r.get('thread_id')].append(r)
  key=normalized(r.get('text',''))
  if r.get('message_id') in ids or (key and (r.get('thread_id'),key) in texts): duplicate+=1
  ids.add(r.get('message_id')); texts[(r.get('thread_id'),key)]=1
 groups=defaultdict(set)
 for r in rows: groups[r.get('source_group_id')].add(r.get('split'))
 cross_split=sum(len(v- {None})>1 for v in groups.values())
 raw=parse_stats.get('raw_records') if parse_stats else n; filtered=parse_stats.get('filtered') if parse_stats else 0
 return {'category_counts': {'bilibili':platforms['bilibili'],'xiaohongshu':platforms['xiaohongshu'],'other_platform':sum(v for k,v in platforms.items() if k not in ('bilibili','xiaohongshu','unknown')),'unknown_platform':platforms['unknown'],'lol':topic['league_of_legends'],'cs2':topic['counter_strike'],'valorant':topic['valorant'],'hearthstone':topic['hearthstone'],'battlegrounds':sub['battlegrounds'],'dota2':topic['dota2'],'nba':topic['nba'],'football':topic['football'],'virtual_streamer':domain['virtual_streamer'],'cn_vup':topic['cn_vup'],'jp_vtuber':topic['jp_vtuber'],'international_vtuber':topic['international_vtuber'],'virtual_idol':topic['virtual_idol'],'danmaku':sum(r.get('content_type')=='danmaku' for r in rows),'stream_clip_comments':sum(r.get('content_type')=='comment' and 'stream_clips' in r.get('subtopics',[]) for r in rows),'fan_community':sum('fan_community' in r.get('subtopics',[]) for r in rows),'gaming_collaboration':sum('gaming_collaboration' in r.get('subtopics',[]) for r in rows),'internet_memes':topic['internet_memes'],'community_interaction':sum('fan_community' in r.get('subtopics',[]) or 'internet_memes' in r.get('topics',[]) for r in rows),'daily_chat':topic['daily_life']}, 'status':'synthetic_fixture' if rows and all(r.get('is_synthetic') for r in rows) else 'real_candidate',
  'raw_records':raw,'retained_messages':n,'filtered_messages':filtered,'count_conserved':raw==n+filtered,
  'filter_reasons':parse_stats.get('filter_reasons',{}) if parse_stats else {},'parse_error_count':len(parse_stats.get('parse_errors',[])) if parse_stats else 'not_measured',
  'error_types':dict(Counter(e['type'] for e in parse_stats.get('parse_errors',[]))) if parse_stats else 'not_measured',
  'platform_counts':dict(platforms),'topic_counts':dict(topic),'domain_counts':dict(domain),'subtopic_counts':dict(sub),
  'independent_threads':len(threads),'messages_with_parent':sum(bool(r.get('parent_comment_id')) for r in rows),
  'missing_parents':sum('parent_out_of_scope' in r.get('quality_flags',[]) for r in rows),
  'multi_speaker_threads':sum(len({r.get('speaker_id') for r in v if r.get('speaker_id')})>1 for v in threads.values()),
  'real_reply_relations':sum(bool(r.get('parent_comment_id')) and 'parent_out_of_scope' not in r.get('quality_flags',[]) for r in rows),
  'duplicate_messages':duplicate,'cross_split_source_groups':cross_split,
  'privacy_risk_messages':sum(bool(r.get('privacy_flags')) for r in rows),
  'abnormal_text_messages':sum(not r.get('text','').strip() or '\ufffd' in r.get('text','') for r in rows),
  'language_distribution':{'contains_han':sum(any('\u4e00'<=c<='\u9fff' for c in r.get('text','')) for r in rows),'no_han':sum(not any('\u4e00'<=c<='\u9fff' for c in r.get('text','')) for r in rows)},
  'license_status_distribution':dict(Counter(r.get('license_status') or 'unknown' for r in rows)),
  'blind_overlap':duplicate_report.get('blind_overlap_counts') if duplicate_report else 'not_measured',
  'trainable_true':sum(r.get('trainable') is True for r in rows),
  'note':'Topic counts overlap; regex privacy scan is incomplete; dated opinions and rumors are not current facts.'}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--parse-stats'); p.add_argument('--duplicate-report'); p.add_argument('--output',required=True)
 a=p.parse_args(); import json
 rows=[r for _,r in read_records(a.input)] if a.input else None
 stats=json.load(open(a.parse_stats,encoding='utf-8')) if a.parse_stats else None
 dup=json.load(open(a.duplicate_report,encoding='utf-8')) if a.duplicate_report else None
 result=audit(rows,stats,dup); write_new(a.output,result); print(result['status'])
if __name__=='__main__': main()
