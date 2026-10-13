"""Privacy-v2 regression tests for QWbot."""
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))

from common.core import privacy, parse_export

def load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace('.', '_'), P / name
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class PrivacyV2Tests(unittest.TestCase):

    def test_delimited_reply(self):
        safe, flags = privacy(
            '回复 @马同学快去学习 :啊啊啊'
        )
        self.assertEqual(
            safe, '回复 @用户 :啊啊啊'
        )
        self.assertIn('reply_mention', flags)
        self.assertNotIn('ambiguous_mention', flags)

    def test_boundary_ambiguity(self):
        safe, flags = privacy(
            '今天@张三和@李四都来了'
        )
        self.assertEqual(
            safe, '今天@用户@用户'
        )
        self.assertIn('ambiguous_mention', flags)
        self.assertNotIn('张三', safe)

    def test_single_cjk_mention_without_separator(self):
        safe, flags = privacy(
            '@音乐演奏大王Kiff'
        )
        self.assertEqual(safe, '@用户')
        self.assertIn('ambiguous_mention', flags)

    def test_english_and_placeholder(self):
        self.assertNotIn(
            'ambiguous_mention',
            privacy('hello @someone')[1]
        )
        self.assertEqual(
            privacy('@用户'), ('@用户', [])
        )

    def test_email_and_url(self):
        safe, flags = privacy(
            'test@example.com https://example.com/@name'
        )
        self.assertIn('email', flags)
        self.assertIn(
            'https://example.com/@name', safe
        )
        self.assertNotIn('user_mention', flags)

    def test_no_invented_parent(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'records.jsonl'
            f.write_text(
                json.dumps({
                    'id': '1',
                    'video_id': 'v1',
                    'text': '回复 @昵称 :好'
                }, ensure_ascii=False) + '\n',
                encoding='utf-8'
            )
            rows, _ = parse_export(f, 'bilibili')
            self.assertIsNone(
                rows[0]['parent_comment_id']
            )
            self.assertFalse(rows[0]['trainable'])

    def test_csv_quarantine_and_formula(self):
        exporter = load('07_export_review.py')
        rows = [
            {
                'message_id': '1',
                'text': '今天@张三和@李四都来了',
                'privacy_flags': [],
                'quality_flags': [],
                'trainable': False
            },
            {
                'message_id': '2',
                'text': '=HYPERLINK(123)',
                'privacy_flags': [],
                'quality_flags': [],
                'trainable': False
            }
        ]

        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'review.csv'
            exporter.export(rows, f)
            with f.open(
                encoding='utf-8-sig', newline=''
            ) as h:
                got = list(csv.DictReader(h))

        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]['message_id'], '2')
        self.assertTrue(
            got[0]['text'].startswith("'=")
        )

    def test_reprocess_and_nested_privacy(self):
        proc = load('09_reprocess_privacy_pilot.py')

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / 'old'
            dest = root / 'new'
            src.mkdir()

            a = src / 'Midsummra__bilibilicomment.pilot.jsonl'
            b = src / 'helloollel__chat_dataset_bilibili.pilot.jsonl'

            r1 = {
                'message_id': 'r1',
                'text': '今天@张三和@李四都来了',
                'source_fields': {
                    'message': '今天@张三和@李四都来了',
                    'extra': ['13812345678']
                },
                'quality_flags': [],
                'privacy_flags': [],
                'parent_comment_id': None,
                'trainable': True
            }

            r2 = {
                'message_id': 'r2',
                'text': '普通文字',
                'instruction': '联系 foo@example.com',
                'input': {
                    'nested': ['回复 @马同学 :好']
                },
                'output': '回复 @马同学 :好',
                'privacy_flags': [],
                'quality_flags': [],
                'parent_comment_id': None,
                'trainable': True
            }

            a.write_text(
                json.dumps(r1, ensure_ascii=False) + '\n',
                encoding='utf-8'
            )
            b.write_text(
                json.dumps(r2, ensure_ascii=False) + '\n',
                encoding='utf-8'
            )

            original = (a.read_bytes(), b.read_bytes())
            proc.run(src, dest)

            self.assertEqual(
                (a.read_bytes(), b.read_bytes()),
                original
            )

            aa = json.loads(
                (dest / a.name).read_text().strip()
            )
            bb = json.loads(
                (dest / b.name).read_text().strip()
            )

            self.assertFalse(aa['trainable'])
            self.assertFalse(bb['trainable'])
            self.assertEqual(
                aa['review_status'], 'privacy_quarantine'
            )
            self.assertIn(
                'ambiguous_mention', aa['privacy_flags']
            )
            self.assertIsNone(aa['parent_comment_id'])
            self.assertIsNone(bb['parent_comment_id'])

            self.assertNotIn(
                '13812345678',
                json.dumps(aa, ensure_ascii=False)
            )
            self.assertNotIn(
                'foo@example.com',
                json.dumps(bb, ensure_ascii=False)
            )

            csv_path = dest / (
                'Midsummra__bilibilicomment.review.csv'
            )
            with csv_path.open(
                encoding='utf-8-sig', newline=''
            ) as h:
                self.assertEqual(
                    len(list(csv.DictReader(h))), 0
                )

            self.assertTrue(
                (dest / 'Midsummra__bilibilicomment.quarantine.jsonl').is_file()
            )

            with self.assertRaises(FileExistsError):
                proc.run(src, dest)

if __name__ == '__main__':
    unittest.main()
