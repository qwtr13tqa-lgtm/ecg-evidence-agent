import json
import tempfile
import unittest
from pathlib import Path
from src.analysis.model_decision import evaluate_decision,render_decision
from evaluation.select_anomaly_threshold import select_threshold
from evaluation.prepare_demo_threshold import activate

class Tests(unittest.TestCase):
    def record(self):
        return {'split':'development_reused','used_for_checkpoint_selection':True,'split_id':'dev',
            'profile':{'model':'x'},'dataset_sha256':'d','labels_sha256':'l',
            'rows':[{'sample_index':0,'score':-1.,'label':0},{'sample_index':1,'score':-.8,'label':1}]}
    def test_disclosed_prediction(self):
        c=select_threshold(self.record());d=evaluate_decision(-.8,c['profile'],c)
        self.assertEqual(d['prediction'],'model_anomaly')
        self.assertIn('开发集演示',render_decision(d))
        self.assertFalse(c['selection']['independent_test_evaluated'])
    def test_no_false_validation_label(self):
        c=select_threshold(self.record())
        self.assertEqual(c['selection']['split'],'development_reused')
    def test_missing_disclosure_rejected(self):
        r=self.record();r.pop('used_for_checkpoint_selection')
        with self.assertRaises(ValueError):select_threshold(r)
    def test_normal_prediction(self):
        c=select_threshold(self.record())
        self.assertEqual(evaluate_decision(-1,c['profile'],c)['prediction'],'model_normal')
    def test_activation_backup(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'threshold.json';path.write_text('{"enabled":false}')
            activate(select_threshold(self.record()),path)
            self.assertTrue(json.loads(path.read_text())['enabled'])
            self.assertEqual(len(list(Path(d).glob('*.backup-*'))),1)
if __name__=='__main__':unittest.main()
