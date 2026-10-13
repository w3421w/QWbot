import importlib.util
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("classifier", P / "19_v6_data_role_classifier.py")
classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classifier)


class DataRoleClassifierTests(unittest.TestCase):
    def row(self, texts, speakers=None, **changes):
        speakers = speakers or [f"s{i % 2}" for i in range(len(texts))]
        row = {"id": "x:1", "source": "test", "dialogue": [{"speaker": s, "text": t} for s, t in zip(speakers, texts)],
               "speaker_count": len(set(speakers)), "topic_hint": None, "community_hint": "bilibili", "candidate_role": "unknown"}
        row.update(changes)
        return row

    def test_high_quality_multi_person_chat_is_finetune(self):
        self.assertEqual(classifier.classify_record(self.row(["你今天完成项目了吗？", "完成了，刚刚把测试也跑完。"]))["data_role"], "finetune")

    def test_meme_explanation_is_knowledge(self):
        result = classifier.classify_record(self.row(["这个梗是什么意思？出处是什么？"], speakers=["A"], source="memebench"))
        self.assertEqual(result["data_role"], "knowledge")

    def test_meaningless_text_is_discard(self):
        self.assertEqual(classifier.classify_record(self.row(["哈哈", "嗯"]))["data_role"], "discard")

    def test_uncertain_is_review(self):
        self.assertEqual(classifier.classify_record(self.row(["天气不错"]))["data_role"], "review")


if __name__ == "__main__":
    unittest.main()
