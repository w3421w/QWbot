import json, tempfile, unittest
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runner', ROOT/'23_v6_pipeline_runner.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)

class TestV6PipelineRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.src = self.root/'raw.jsonl'
        self.src.write_text(json.dumps({'source_type':'lccc_chat','data':['问题内容足够长','回答内容也足够长']}, ensure_ascii=False)+'\n', encoding='utf8')
    def tearDown(self): self.tmp.cleanup()
    def test_full_pipeline_and_intermediates(self):
        out = self.root/'run'; self.assertEqual(runner.main([str(self.src),'--output-dir',str(out)]), 0)
        for p in ['dialogue/v6_dialogue_raw.jsonl','classified/v6_classified.jsonl','reports/v6_quality_report.json','sft/qwen_sft_train.jsonl','split/train.jsonl','split/validation.jsonl','split/test.jsonl']:
            self.assertTrue((out/p).exists(), p)
    def test_error_stage_recorded(self):
        out = self.root/'bad'; self.assertEqual(runner.main([str(self.root/'missing.jsonl'),'--output-dir',str(out)]), 1)
        errors = json.loads((out/'pipeline_errors.json').read_text())
        self.assertTrue(any(e['stage'] == 'adapter' for e in errors))
    def test_seed_is_passed(self):
        out = self.root/'seed'; runner.main([str(self.src),'--output-dir',str(out),'--seed','123'])
        report = json.loads((out/'split/split_report.json').read_text())
        self.assertEqual(report.get('seed'), 123)

if __name__ == '__main__': unittest.main()
