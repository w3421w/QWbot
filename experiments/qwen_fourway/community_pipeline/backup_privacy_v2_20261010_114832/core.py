"""Offline community record helpers. No training eligibility is inferred here."""
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'rules-1.1-privacy-mentions'

# A mention is deliberately conservative: usernames are made of word/CJK
# characters (plus a hyphen), while email addresses and URLs are handled
# outside this expression.  The preceding boundary permits ``你好@用户`` but
# prevents matching the ``@`` in ``name@example.com``.
_MENTION_RE = re.compile(
    r'(?<![A-Za-z0-9_@.])@(?P<name>[\w\u3400-\u4dbf\u4e00-\u9fff-]+)',
    re.UNICODE,
)
_URL_RE = re.compile(r'(?:(?:https?|ftp)://|www\.)[^\s<>"\']+', re.I)
_URL_SENTINEL = '\x00QWBOT_URL_{}\x00'


def digest(*parts):
    return hashlib.sha256('\x1f'.join(map(str, parts)).encode()).hexdigest()[:20]


def write_new(path, data, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        if jsonl:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')


def read_records(path):
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in ('.json', '.jsonl'):
        raise ValueError('Expected an existing .json or .jsonl file: ' + str(path))
    with path.open(encoding='utf-8-sig') as f:
        if path.suffix.lower() == '.jsonl':
            for n, line in enumerate(f, 1):
                if line.strip():
                    try: yield n, json.loads(line)
                    except json.JSONDecodeError as e: yield n, {'_parse_error': str(e)}
        else:
            data = json.load(f)
            if isinstance(data, dict): data = data.get('records', data.get('comments', [data]))
            if not isinstance(data, list): raise ValueError('JSON root must be a list or contain records/comments')
            for n, row in enumerate(data, 1): yield n, row


def normalized(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text or '').lower())


def privacy(text):
    flags = []
    patterns = [
        ('phone', r'(?<!\d)1[3-9]\d{9}(?!\d)'),
        ('identity_card', r'(?<!\d)\d{17}[\dXx](?!\d)'),
        ('email', r'(?<![\w.])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'),
        ('private_address', r'(?:住址|家庭地址|我家地址|私人地址)\s*[:：]?\s*[^，。\n]{6,50}'),
        ('private_account', r'(?:微信号|QQ号|私人账号)\s*[:：]?\s*[\w-]{5,}'),
        ('private_identity', r'(?:中之人|真人身份|皮下是)\s*[:：]?\s*[^，。\n]{2,30}'),
    ]
    result = text or ''

    # Keep URL contents opaque while applying text privacy rules.  In
    # particular, ``https://example/@name`` is a URL, not a user mention.
    urls = []
    def protect_url(match):
        urls.append(match.group(0))
        return _URL_SENTINEL.format(len(urls) - 1)
    result = _URL_RE.sub(protect_url, result)

    for name, pattern in patterns:
        result, count = re.subn(pattern, '[REDACTED]', result, flags=re.I)
        if count: flags.append(name)

    def replace_mention(match):
        # Only the immediately preceding text determines the interaction
        # marker.  This preserves the words ``回复 @用户`` without inventing a
        # parent_comment_id or identifying a reply target.
        prefix = result_before_mentions[:match.start()]
        is_reply = bool(re.search(r'回复\s*$', prefix))
        if 'user_mention' not in flags:
            flags.append('user_mention')
        if is_reply and 'reply_mention' not in flags:
            flags.append('reply_mention')
        return '@用户'

    # The callback needs the current pre-replacement string to inspect the
    # prefix.  ``result_before_mentions`` is intentionally fixed for this
    # single re.sub pass, so multiple mentions are handled independently.
    result_before_mentions = result
    result = _MENTION_RE.sub(replace_mention, result_before_mentions)

    def restore_url(match):
        return urls[int(match.group(1))]
    result = re.sub(r'\x00QWBOT_URL_(\d+)\x00', restore_url, result)
    return result, flags


def make_record(platform, raw, source_file, source_record_id, *, content_type='comment', synthetic=False):
    """Adapter-neutral normalized record; caller supplies only observed identifiers."""
    rid = raw.get('id') or raw.get('comment_id')
    if rid is None: raise ValueError('missing_comment_id')
    if raw.get('text') is None or not isinstance(raw['text'], str): raise ValueError('missing_text')
    scope = str(raw.get('video_id') or raw.get('post_id') or raw.get('thread_id') or Path(source_file).resolve())
    source_ns = digest(platform, scope)
    parent = raw.get('parent_id')
    root = raw.get('root_id')
    author = raw.get('author_id')
    safe_text, pflags = privacy(raw['text'])
    return {
        'platform': platform, 'domain': [], 'topics': [], 'subtopics': [],
        'classification_confidence': 0.0, 'classification_evidence': [], 'classification_version': None,
        'thread_id': f'{platform}:{source_ns}:{scope}',
        'message_id': f'{platform}:{source_ns}:{rid}',
        'parent_comment_id': f'{platform}:{source_ns}:{parent}' if parent not in (None, '', 0, '0') else None,
        'root_comment_id': f'{platform}:{source_ns}:{root}' if root not in (None, '', 0, '0') else None,
        'speaker_id': f'{platform}:speaker:{digest(platform, source_ns, author)}' if author is not None else None,
        'message_order': raw.get('order'), 'text': safe_text, 'content_type': content_type,
        'created_at': raw.get('created_at'), 'video_id': raw.get('video_id'), 'post_id': raw.get('post_id'),
        'source_url': raw.get('source_url'), 'source_file': str(Path(source_file).resolve()),
        'source_record_id': str(source_record_id), 'provenance_type': 'synthetic_fixture' if synthetic else 'offline_user_export',
        'is_synthetic': synthetic, 'source_license': raw.get('source_license'),
        'license_status': 'synthetic_fixture' if synthetic else 'verification_pending',
        'review_status': 'unreviewed', 'trainable': False, 'quality_flags': [], 'privacy_flags': pflags,
        'parse_status': 'parsed', 'split': None, 'source_group_id': f'{platform}:{source_ns}:{scope}',
    }


def finish_tree(records):
    ids = {r['message_id'] for r in records}
    for r in records:
        if r['parent_comment_id'] and r['parent_comment_id'] not in ids:
            r['quality_flags'].append('parent_out_of_scope')
        if r['parent_comment_id'] == r['message_id']:
            r['quality_flags'].append('self_parent')
    return records


def parse_export(path, platform, *, synthetic=False):
    records, errors, seen = [], [], set()
    for line, raw in read_records(path):
        if not isinstance(raw, dict) or '_parse_error' in raw:
            errors.append({'line': line, 'type': 'invalid_json_or_record'}); continue
        try:
            if platform == 'bilibili':
                content_type = raw.get('content_type', 'comment')
                if content_type not in ('comment', 'danmaku', 'livestream_chat'): raise ValueError('invalid_content_type')
                if content_type == 'comment' and not (raw.get('video_id') or raw.get('thread_id')): raise ValueError('missing_video_or_thread_id')
            else:
                content_type = raw.get('content_type', 'comment')
                if content_type not in ('comment', 'post'): raise ValueError('invalid_content_type')
                if not raw.get('post_id'): raise ValueError('missing_post_id')
            record = make_record(platform, raw, path, line, content_type=content_type, synthetic=synthetic)
            if record['message_id'] in seen:
                errors.append({'line': line, 'type': 'duplicate_id'}); continue
            seen.add(record['message_id'])
            records.append(record)
        except ValueError as e:
            errors.append({'line': line, 'type': str(e)})
    finish_tree(records)
    return records, {'raw_records': len(records) + len(errors), 'retained': len(records), 'filtered': len(errors), 'parse_errors': errors,
                     'filter_reasons': {k: sum(e['type'] == k for e in errors) for k in sorted({e['type'] for e in errors})}}

def assign_group_splits(records, validation_fraction=0.1):
    """Technical grouping only. Does not change trainable or licensing."""
    if not 0 <= validation_fraction <= 1: raise ValueError('validation_fraction must be within [0,1]')
    groups = {}
    for r in records:
        group = r.get('source_group_id')
        if not group: raise ValueError('source_group_id required for split')
        groups.setdefault(group, 'validation' if int(digest(group), 16) / 16**20 < validation_fraction else 'train')
        r['split'] = groups[group]
        r['trainable'] = False
    return records
