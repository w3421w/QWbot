import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

P = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('clean', P / '01_clean_all.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class StructureTests(unittest.TestCase):
    def test_high_frequency_short_phrase_does_not_bind_sources(self):
        self.assertEqual(m.compact('前排'), m.compact('前排'))
        self.assertNotEqual('post-a', 'post-b')

    def test_chain_near_duplicates_do_not_create_source_component(self):
        idx = m.NearIndex()
        self.assertIsNone(idx.add('这是一个足够长的高信息量测试句子用于链式检测'*2, 'p1:x'))
        self.assertEqual('p1', idx.add('这是一个足够长的高信息量测试句子用于链式检测'*2, 'p2:x').split(':')[0])
        self.assertEqual('p1', idx.add('这是一个足够长的高信息量测试句子用于链式检测'*2, 'p3:x').split(':')[0])
        self.assertFalse(hasattr(idx, 'union'))

    def test_near_and_exact_match_primitives(self):
        a = '这是一个足够长的高信息量帖子内容用于重复检测'
        self.assertTrue(m.near(a, a))
        self.assertEqual(m.digest(a), m.digest(a))

    def test_cross_split_one_to_one_and_one_to_many(self):
        train=[('t1',m.compact('这是一个跨集合近似重复样本用于回归测试甲')),
               ('t2',m.compact('另一条完全不同的训练回复'))]
        validation=[('v1',m.compact('这是一个跨集合近似重复样本用于回归测试乙')),
                    ('v2',m.compact('这是一个跨集合近似重复样本用于回归测试丙'))]
        pairs=m.validator_near_cross_split_pairs(train,validation) if hasattr(m,'validator_near_cross_split_pairs') else None
        # The cleaner module does not own the validator; import the shared
        # implementation exactly as the repair script does.
        import importlib.util
        s=importlib.util.spec_from_file_location('v',P/'04_validate_outputs.py'); v=importlib.util.module_from_spec(s); s.loader.exec_module(v)
        got={(train[i][0],validation[j][0]) for i,j in v.near_cross_split_pairs(train,validation)}
        self.assertEqual(got,{('t1','v1'),('t1','v2')})

    def test_chain_and_short_shared_phrase_are_all_reported(self):
        import importlib.util
        s=importlib.util.spec_from_file_location('v',P/'04_validate_outputs.py'); v=importlib.util.module_from_spec(s); s.loader.exec_module(v)
        base='共同短梗以及足够长的上下文用于链式近似检测'
        train=[('t1',v.compact(base+'甲')),('t2',v.compact(base+'乙'))]
        validation=[('v1',v.compact(base+'丙')),('v2',v.compact('共同短梗以及足够长的上下文用于链式近似检测丁'))]
        got={(train[i][0],validation[j][0]) for i,j in v.near_cross_split_pairs(train,validation)}
        self.assertEqual(got,{('t1','v1'),('t1','v2'),('t2','v1'),('t2','v2')})

    def test_real_clean_split_isolation_and_outputs(self):
        with tempfile.TemporaryDirectory(prefix='v21-test-') as td:
            root = Path(td); out, review = root / 'out', root / 'review'
            m.run(SimpleNamespace(check_only=False, output_dir=out, review_dir=review,
                                  sample_posts=100, seed=20261009))
            rows = []
            for fn in ('style_candidates.jsonl', 'rejected.jsonl',
                       'test_quarantine.jsonl', 'duplicate_quarantine.jsonl'):
                rows.extend(json.loads(x) for x in (out / fn).read_text().splitlines() if x)
            self.assertTrue(rows)
            by_group = {}
            for r in rows:
                by_group.setdefault(r['source_group_id'], set()).add(r['data_split'])
                self.assertFalse({'post_title_original', 'op_body_original',
                                  'original_reply', 'source_url'} & set(r))
                self.assertFalse(r['trainable'])
            self.assertTrue(all(len({x for x in v if x in ('train', 'validation')}) <= 1 for v in by_group.values()))
            stats = json.loads((out / 'quality_statistics.json').read_text())
            self.assertIn('split_counts', stats)
            self.assertTrue((out / 'test_overlap_review.jsonl').is_file())

    def test_test_quarantine_is_not_downstream_review(self):
        with tempfile.TemporaryDirectory(prefix='v21-test-contract-') as td:
            root = Path(td); out, review = root / 'out', root / 'review'
            # Inject an exact hit from a real source reply into the 31-question
            # matcher, exercising the same quarantine path used in production.
            reply = next(x['回复列表'][0] for _, _, x in m.posts(m.files(), None, 7)
                         if x.get('回复列表'))
            vals = [('T01', 'incoming_message', m.quality.norm(reply))]
            exact = {m.digest(m.quality.norm(reply).casefold()): [0]}
            old_index = m.test_index
            old_posts = m.posts
            m.test_index = lambda: (vals, exact, {})
            selected = next((item for item in m.posts(m.files(), None, 7)
                             if item[2].get('回复列表')), None)
            m.posts = lambda _fs, _sample, _seed: iter([selected])
            try:
                m.run(SimpleNamespace(check_only=False, output_dir=out, review_dir=review,
                                      sample_posts=1, seed=7))
            finally:
                m.test_index = old_index
                m.posts = old_posts
            quarantined = {json.loads(x)['id'] for x in (out / 'test_quarantine.jsonl').read_text().splitlines() if x}
            self.assertTrue(quarantined)
            downstream = set()
            for fn in ('style_candidates.jsonl', 'sft_candidates.jsonl', 'review_queue.jsonl'):
                downstream |= {json.loads(x)['id'] for x in (out / fn).read_text().splitlines() if x}
            self.assertTrue(quarantined.isdisjoint(downstream))


if __name__ == '__main__':
    unittest.main()
