import unittest
from unittest.mock import Mock
from types import SimpleNamespace as N
from src.review.conversation_batch import review,run

class Tests(unittest.TestCase):
 def setUp(self):
  self.r={'threshold':0.,'adapter_sha256':'a','checkpoint_sha256':'c','rows':[dict(index=i,label=0,prediction=1,score=1.,input_sha256=str(i)*64) for i in (1,2,3)]}
  self.h=Mock();self.h.list_analyses.return_value=[]
 def test_all_missing_covered(self):
  rows=review(self.r,self.h,'fp');self.assertEqual(len(rows),3);self.assertTrue(all(x['state']=='needs_local_analysis' for x in rows))
 def test_configured_not_sufficient(self):
  self.h.list_analyses.return_value=[dict(sample_index=1,analysis_id='x')]
  self.h.load_analysis.return_value=(N(provenance={'model_decision':{'status':'configured'}}),None,None)
  self.assertEqual(review(self.r,self.h,'fp')[0]['state'],'needs_local_analysis')
 def test_matched_evidence(self):
  self.h.list_analyses.return_value=[dict(sample_index=1,analysis_id='x')]
  r=N(analysis_id='x',model=N(anomaly_score=1.),provenance=dict(input_sha256='1'*64,checkpoint_sha256='c',review_adapter_sha256='a',sample_index=1,crop_start_sample=100,model_decision=dict(status='configured',threshold=0.,prediction='model_anomaly')),to_llm_context=lambda:{'model':{'anomaly_score':1}})
  self.h.load_analysis.return_value=(r,None,None)
  rows=review(self.r,self.h,'fp');self.assertEqual(rows[0]['source_analysis_id'],'x');self.assertEqual(rows[1]['state'],'needs_local_analysis')
 def test_other_index_not_read(self):
  self.h.list_analyses.return_value=[dict(sample_index=9,analysis_id='other')];review(self.r,self.h,'fp');self.h.load_analysis.assert_not_called()
 def test_loop_output_is_full_batch_not_active_record(self):
  g=Mock();g.complete.return_value={'finish_reason':'tool_calls','tool_calls':[{'type':'function','function':{'name':'review_batch_history','arguments':'{"category":"fp"}'}}]}
  out=run(self.h,g,'unrelated_current','检查误报',self.r,'hash')
  self.assertEqual(len(out['trace'][0]['task_state']),3);self.assertEqual(out['status'],'completed_draft')
 def test_invalid_category(self):
  with self.assertRaises(ValueError):review(self.r,self.h,'other_report')
if __name__=='__main__':unittest.main()
