import importlib.util,unittest,tempfile,json,hashlib
from pathlib import Path
spec=importlib.util.spec_from_file_location('report',Path(__file__).resolve().parents[1]/'evaluation/report_interview_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def record(self,d,review=None,fields=None,status='completed_draft'):
  p=Path(d)/'result.json';r={'run_id':'r','config':{'batch_id':'b','scheme':'agent'},'case':{'id':'c'},'output':{'status':status,'trace':[]},'automatic':{'field_checks':fields or {},'metrics':{'structure_values_references':True}}};p.write_text(json.dumps(r))
  if review:
   v={'run_id':'r','reviewer':'reviewer','result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),**review};p.with_name('review.json').write_text(json.dumps(v))
  return m.inspect(p)
 def test_pending_is_not_pass(self):
  with tempfile.TemporaryDirectory() as d:r=self.record(d);self.assertEqual(r['status'],'pending');self.assertEqual(m.aggregate([r])[0]['confirmed_success'],0)
 def test_failure_overrides_review(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(self.record(d,dict(task_correct='pass',evidence_support='pass',text_complete='pass'),status='failed')['status'],'fail')
 def test_fields_override_pass(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(self.record(d,dict(task_correct='pass',evidence_support='pass',text_complete='pass'),fields={'mean':False})['status'],'fail')
 def test_empty_denominator_missing(self):
  with tempfile.TemporaryDirectory() as d:self.assertIsNone(m.aggregate([self.record(d)])[0]['coverage'])
 def test_review_hash_binding(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(self.record(d,dict(task_correct='pass',evidence_support='pass',text_complete='pass',result_sha256='bad'))['status'],'pending')
 def test_full_pass(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(self.record(d,dict(task_correct='pass',evidence_support='pass',text_complete='pass'))['status'],'pass')
 def test_quantile(self):self.assertAlmostEqual(m.quantile([0,10],.95),9.5)
if __name__=='__main__':unittest.main()
