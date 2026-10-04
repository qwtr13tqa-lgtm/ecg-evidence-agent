import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from evaluation.export_comparison import collect,export,failure_kind
from src.evaluation.records import RunRecord,atomic_json

class Tests(unittest.TestCase):
    def setup_batch(self,root):
        batch=str(uuid.uuid4());cfg={'batch_id':batch,'model':'test'}
        atomic_json(root/(batch+'.manifest.json'),{'config':cfg,'tasks':[{'case_id':'C','repetition':1,'schemes':['summary','agent']}]})
        r=RunRecord(root,{'id':'C','sample_index':0},{**cfg,'scheme':'summary','repetition':1})
        r.finish({'status':'failed','error':'GATEWAY_REQUEST_FAILED','model_calls':1,'elapsed_seconds':180,
                  'trace':[{'stage':'model','error_type':'APITimeoutError'}]}, {},{'version':'old'})
        return batch,r.path
    def collect(self,root,b):
        with patch('evaluation.export_comparison.score_output',return_value={'version':'development-checks-2.0','metrics':{'completed_draft':False,'task_observations':None}}):
            return collect(root,b)
    def test_missing_null_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b,p=self.setup_batch(root);r=self.collect(root,b)
            self.assertEqual(r['groups']['agent']['missing'],1)
            self.assertEqual(r['groups']['summary']['failure_categories']['gateway_timeout'],1)
            self.assertIsNone(r['groups']['summary']['metrics']['task_observations']['pass_rate_assessed'])
    def test_exports_preserve_original(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b,p=self.setup_batch(root);before=(p/'result.json').read_bytes()
            report=self.collect(root,b);export(report,root/'out')
            self.assertEqual(before,(p/'result.json').read_bytes())
            self.assertTrue((root/'out/cases.csv').exists());self.assertIn('未评估',(root/'out/report.md').read_text())
            self.assertEqual(report['rows'][0]['result_sha256'],hashlib.sha256(before).hexdigest())
            with self.assertRaises(FileExistsError):export(report,root/'out')
    def test_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b,p=self.setup_batch(root);r=json.loads((p/'result.json').read_text());r['config']['model']='changed';atomic_json(p/'result.json',r)
            with self.assertRaises(ValueError):self.collect(root,b)
    def test_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b,p=self.setup_batch(root);r=json.loads((p/'result.json').read_text());other=root/str(uuid.uuid4());other.mkdir();r['run_id']=other.name;atomic_json(other/'result.json',r)
            with self.assertRaises(ValueError):self.collect(root,b)
    def test_no_sidecar_trust(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b,p=self.setup_batch(root);atomic_json(p/'automatic_v2.json',{'metrics':{'completed_draft':True}})
            self.assertEqual(self.collect(root,b)['groups']['summary']['metrics']['completed_draft']['false'],1)
    def test_phases(self):
        for phase in ('answer_json','response_protocol','answer_validation'):
            r={'output':{'status':'failed','trace':[{'error_type':'ValueError','failure_phase':phase}]}}
            self.assertEqual(failure_kind(r),phase+'_failure')
    def test_path_rejected(self):
        with self.assertRaises(ValueError):collect('.', '../outside')
if __name__=='__main__':unittest.main()
