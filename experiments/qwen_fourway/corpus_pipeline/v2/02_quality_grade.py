"""Deterministic, deliberately conservative V2 text grading rules."""
import re
import unicodedata

AD = re.compile(r'购物礼金|优惠礼包|领红包|返现(?:活动|领取)|刷单|兼职日结|推广邀请码|扫码(?:进群|购买|领取|加群)|代练接单|陪玩接单|招商加盟|招代理|买粉|出售账号|京东首页搜|淘宝(?:店铺|优惠券)|拼多多(?:砍价|领券)|点击(?:链接|购买)|联系(?:客服|商家)下单|优惠券(?:领取|入口)', re.I)
URL = re.compile(r'https?://\S+|www\.\S+|t\.cn/\S+', re.I)
CONTACT = re.compile(r'(?:加我?|联系我?)\s*(?:微信|微|vx|v信|qq)', re.I)
PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
EMAIL = re.compile(r'(?<![\w.-])[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}(?![\w.-])')
CHAT = re.compile(r'(?:微信|微 信|vx|v信|QQ号|QQ号码)\s*[:：= ]\s*[A-Za-z0-9_-]{5,}', re.I)
IDCARD = re.compile(r'(?<!\d)\d{17}[0-9Xx](?!\d)')
ADDRESS = re.compile(r'(?:住址|家庭住址|详细地址|收货地址)\s*[:：]\s*[^，。\n]{6,}')
THREAT = re.compile(r'杀你全家|死全家|人肉搜索(?:地址|电话)|开盒(?:地址|家人)|种族灭绝', re.I)
RUMOR = re.compile(r'(?:听说|据说|网传|爆料|传闻).{0,25}(?:私生活|出轨|吸毒|犯罪|家人|住址|手机号|身份证)', re.I)
LOW = re.compile(r'^(?:[哈啊呃哦嗯啧哎]{3,}|666+|\d+|顶|插眼|蹲|前排|cy|dd|mark)$', re.I)
MEME = re.compile(r'回旋镖|绷不住|典中典|大的要来|闹麻了|乐子|神中神|小丑|打脸|反噬|嘴硬|逆天|牢[\u4e00-\u9fff]|赢麻|幽默|汗流浃背|急了|纯路人|一眼丁真|有一说一|快进到|这下|哥哥|[？?！!]{2,}')
SPORTS = re.compile(r'(?i)\b(?:lpl|lck|blg|jdg|ig|tes|rng|wbg|edg|bp|lol|msi|s赛)\b|打野|上单|中单|下路|辅助|对线|团战|赛区|比赛|选手|战队|冠军|教练|版本')
QUESTION = re.compile(r'[？?]|怎么|如何|为啥|为什么|谁的锅|能不能|是不是|评价|怎么看|咋办|有没有')
ANSWER = re.compile(r'因为|就是|我觉得|我看|感觉|应该|可以|不能|能赢|打不过|[\u4e00-\u9fff]{1,5}的锅')


def norm(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value)).strip() if isinstance(value, str) else ''


def privacy_kinds(text):
    return [name for name, pattern in [('phone', PHONE), ('email', EMAIL), ('chat_id', CHAT), ('id_card', IDCARD), ('address', ADDRESS)] if pattern.search(text)]


def redact(text):
    for pattern, label in [(PHONE, '[手机号已脱敏]'), (EMAIL, '[邮箱已脱敏]'), (CHAT, '[联系方式已脱敏]'), (IDCARD, '[身份证号已脱敏]'), (ADDRESS, '[地址已脱敏]')]:
        text = pattern.sub(label, text)
    return text


def spam_or_garbage(text):
    if not text: return 'empty'
    if AD.search(text) or CONTACT.search(text) or (URL.search(text) and re.search(r'购买|优惠|下单|私聊|推广|扫码|返现|免费领|客服|包邮|出售', text)): return 'advertisement'
    if THREAT.search(text): return 'explicit_threat'
    if len(text) > 3000: return 'oversize_spam_review'
    if len(text) >= 12 and len(set(text)) <= 2: return 'machine_repetition'
    if re.search(r'(.)\1{11,}', text): return 'machine_repetition'
    printable = [ch for ch in text if not ch.isspace()]
    if len(printable) >= 10 and sum(unicodedata.category(ch)[0] in 'LNP' for ch in printable) / len(printable) < .35: return 'garbled'
    if len(text) >= 16 and len(set(text)) / len(text) < .12: return 'machine_repetition'
    return None


def bigrams(text):
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', norm(text).casefold())
    return {text[i:i+2] for i in range(len(text)-1)}


def semantic_evidence(title, body, reply):
    context = ' '.join(x for x in (title, body) if x and x != '[无法获取楼主内容]')
    if not context: return {'possible': False, 'score': 0, 'evidence': ['missing_context']}
    shared = bigrams(reply) & bigrams(context)
    stop = {'这个','那个','什么','怎么','就是','还是','一个','不是','感觉','可以','的话','真的','有点','他们','我们','你们','自己','现在','因为','没有','知道','已经','时候','里面','所以','应该','如果','可能','不会','都是','还有','一下','今天','以后','这么','这么','人家'}
    overlap = sorted(shared - stop)
    question = bool(QUESTION.search(context))
    answer = bool(ANSWER.search(reply))
    score = min(2, len(overlap)) + int(question) + int(answer)
    evidence = (['question_in_context'] if question else []) + (['answer_form'] if answer else []) + ([f'lexical_overlap:{",".join(overlap[:5])}'] if overlap else [])
    possible = bool(overlap and (question or len(overlap) >= 2))
    return {'possible': possible, 'score': score, 'evidence': evidence or ['no_lexical_alignment']}


def grade(text, title, body, reasons, near_duplicate, rumor):
    """Return quality independent of review, license, and relation status."""
    score = {'text': 0, 'style': 0, 'context': 0, 'penalty': 0}
    if reasons: return 'REJECT', score
    n = len(text)
    score['text'] = 2 if 6 <= n <= 240 else (1 if n > 0 else 0)
    score['style'] = int(bool(MEME.search(text))) + int(bool(SPORTS.search(text))) + int(bool(re.search(r'[，。！？?!]', text)))
    score['context'] = int(bool(title)) + int(bool(body and body != '[无法获取楼主内容]'))
    score['penalty'] = int(bool(near_duplicate)) + int(bool(rumor)) + int(bool(LOW.fullmatch(text))) + int(n > 500)
    total = sum(score[k] for k in ('text','style','context')) - score['penalty']
    if n <= 2 or LOW.fullmatch(text) or rumor or near_duplicate or not title: return 'C', score
    if total >= 5 and n <= 240: return 'A', score
    if total >= 3: return 'B', score
    return 'C', score
