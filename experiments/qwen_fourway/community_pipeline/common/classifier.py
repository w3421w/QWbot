"""Conservative rule candidates, never validated semantic labels."""
import re
from .core import VERSION

RULES = {
 'league_of_legends': ('esports', {'lol': r'(?<![a-z])lol(?![a-z])|英雄联盟|LPL|LCK|抗压背锅吧|抗吧', 'game': r'召唤师峡谷|英雄联盟'}),
 'counter_strike': ('esports', {'cs2': r'(?<![a-z])CS2(?![a-z])', 'csgo': r'(?<![a-z])CSGO(?![a-z])|反恐精英', 'major': r'CS(?:2|GO)\s*Major'}),
 'valorant': ('esports', {'valorant': r'VALORANT|无畏契约|瓦罗兰特|(?<![a-z])VCT(?![a-z])'}),
 'hearthstone': ('esports', {'hearthstone': r'炉石传说|炉石|Hearthstone|(?<![a-z])HS(?![a-z])', 'battlegrounds': r'酒馆战棋|Hearthstone\s+Battlegrounds'}),
 'dota2': ('esports', {'dota2': r'DOTA\s*2|刀塔2|国际邀请赛\s*TI'}),
 'nba': ('sports', {'nba': r'(?<![a-z])NBA(?![a-z])|湖人队|勇士队|凯尔特人队|尼克斯队|季后赛.*(?:篮球|湖人|勇士)'}),
 'football': ('sports', {'football': r'英超|西甲|意甲|德甲|法甲|欧冠|世界杯|足球|Premier League|Champions League|(?<![a-z])FIFA(?![a-z])'}),
 'cn_vup': ('virtual_streamer', {'cn_vup': r'国V|虚拟UP主|(?<![a-z])VUP(?![a-z])|A-SOUL|VirtuaReal'}),
 'jp_vtuber': ('virtual_streamer', {'jp_vtuber': r'日V|Hololive|Nijisanji'}),
 'international_vtuber': ('virtual_streamer', {'international_vtuber': r'海外V|EN\s*VTuber'}),
 'virtual_idol': ('virtual_streamer', {'virtual_idol': r'虚拟偶像'}),
 'general_vtuber_discussion': ('virtual_streamer', {'vtuber': r'VTuber|虚拟主播|V圈'}),
 'internet_memes': ('internet_community', {'meme': r'网络热梗|接梗|抽象表达|玩梗'}),
 'daily_life': ('internet_community', {'daily': r'上班|下班|吃饭|宿舍|通勤|日常'}),
}
SUB = {
 'hearthstone': {'standard': r'标准模式', 'wild': r'狂野模式', 'arena': r'竞技场', 'battlegrounds': r'酒馆战棋|Hearthstone\s+Battlegrounds', 'deck_building': r'卡组|构筑', 'classes': r'职业', 'competitive': r'赛事', 'streamers': r'主播', 'community_memes': r'梗', 'player_discussion': r'吐槽', 'game_updates': r'补丁|更新'},
 'counter_strike': {'cs2': r'CS2', 'csgo_history': r'CSGO'},
 'football': {'premier_league': r'英超|Premier League', 'la_liga': r'西甲', 'serie_a': r'意甲', 'bundesliga': r'德甲', 'ligue_1': r'法甲', 'champions_league': r'欧冠|Champions League', 'world_cup': r'世界杯', 'transfers': r'转会'},
}

def classify(text, context='', manual=None):
    if manual is not None:
        return {'domain': manual.get('domain', []), 'topics': manual.get('topics', []), 'subtopics': manual.get('subtopics', []), 'classification_confidence': 1.0, 'classification_evidence': ['manual_review'], 'classification_version': VERSION}
    combined = text + (' ' + context if context else '')
    topics, domains, subtopics, evidence = [], [], [], []
    ambiguous = False
    if re.search(r'云顶之弈|TFT', combined, re.I) and re.search(r'战棋', text) and not re.search(r'炉石|酒馆战棋|Hearthstone', text, re.I):
        ambiguous = True
    for topic, (domain, aliases) in RULES.items():
        hits = [key for key, pattern in aliases.items() if re.search(pattern, text, re.I)]
        context_hits = [key for key, pattern in aliases.items() if re.search(pattern, context, re.I)] if context else []
        if not hits and not (context_hits and re.search(r'这个|这场|他们|主播|游戏', text)): continue
        if topic == 'hearthstone' and ambiguous: continue
        topics.append(topic)
        if domain not in domains: domains.append(domain)
        evidence.append({'topic': topic, 'matched_rules': hits, 'context_rules': context_hits, 'kind': 'rule_candidate'})
        for sub, pattern in SUB.get(topic, {}).items():
            if re.search(pattern, text, re.I) and sub not in subtopics: subtopics.append(sub)
    if 'virtual_streamer' in domains:
        for sub, pattern in {'stream_clips': r'直播切片|切片', 'gaming_collaboration': r'联动|双排|开黑', 'fan_community': r'粉丝|舰长|SC', 'fan_creation': r'二创|鬼畜', 'livestream_interaction': r'弹幕|直播间'}.items():
            if re.search(pattern, text, re.I): subtopics.append(sub)
    if re.search(r'(?<![a-z])VT(?![a-z])|(?<![a-z])HS(?![a-z])|(?<![a-z])LAL(?![a-z])|(?<![a-z])NYK(?![a-z])', text, re.I) and not topics: ambiguous = True
    return {'domain': domains or ['other'], 'topics': topics or ['unknown'], 'subtopics': list(dict.fromkeys(subtopics)),
            'classification_confidence': (0.55 if context and topics else 0.72 if topics else 0.0),
            'classification_evidence': evidence + ([{'ambiguity': 'insufficient_context'}] if ambiguous else []),
            'classification_version': VERSION}
