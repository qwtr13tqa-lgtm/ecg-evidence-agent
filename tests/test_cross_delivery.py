import unittest
from copy import deepcopy
from test_cross_record_agent import Gateway,History,REPORT
from src.review.cross_record_agent import run
from src.review.cross_context import model_evidence
class Tests(unittest.TestCase):
    def snapshot(self):return run(History(),Gateway(),'root','比较',REPORT,'hash')
    def test_retry_uses_snapshot_without_history_or_report(self):
        old=self.snapshot();before=deepcopy(old)
        out=run(None,Gateway(),'root','比较',None,'hash',max_model_calls=1,snapshot=old)
        self.assertEqual(out['status'],'completed_draft');self.assertEqual(out['tool_calls'],0)
        self.assertEqual(out['model_calls'],1);self.assertEqual(old,before)
    def test_wrong_batch_rejected(self):
        out=run(None,Gateway(),'root','比较',None,'other',snapshot=self.snapshot())
        self.assertEqual(out['error'],'SNAPSHOT_BATCH_MISMATCH');self.assertEqual(out['model_calls'],0)
    def test_wrong_analysis_rejected(self):
        self.assertEqual(run(None,Gateway(),'other','比较',None,'hash',snapshot=self.snapshot())['error'],'SNAPSHOT_ANALYSIS_MISMATCH')
    def test_wrong_values_still_rejected(self):
        self.assertEqual(run(None,Gateway(wrong=True),'root','比较',None,'hash',snapshot=self.snapshot())['error'],'OBSERVATION_VALUE_MISMATCH')
    def test_projection_retains_group_values_and_order(self):
        e=next(iter(self.snapshot()['evidence'].values()));copy=deepcopy(e);m=model_evidence(e)
        self.assertEqual(e,copy)
        for a,b in zip(e['data']['groups'],m['data']['groups']):
            for k,v in b.items():self.assertEqual(v,a[k])
        self.assertNotIn('observation_paths',m)
if __name__=='__main__':unittest.main()
