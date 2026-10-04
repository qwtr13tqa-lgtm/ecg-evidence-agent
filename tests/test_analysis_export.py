import copy
import json
import unittest
from types import SimpleNamespace as S
from unittest.mock import patch
from src.reporting.analysis_export import build_export,json_bytes,markdown_bytes,request_matches

class Tests(unittest.TestCase):
    def fixture(self):
        context={'provenance':{'analysis_id':'A'},'input':{'num_samples':4800},'model':{}}
        result=S(analysis_id='A',rhythm=None,to_llm_context=lambda:copy.deepcopy(context))
        item={'analysis_id':'A','evidence_id':'A:E','ok':True,'data':{'n':9}}
        answer={'answer':'共9个间隔。','evidence_ids':['A:E'],'knowledge_ids':[],
                'observations':[{'evidence_id':'A:E','path':'/n','value':9}]}
        out={'analysis_id':'A','status':'completed_draft','draft':answer,'evidence':{'A:E':item},'knowledge':{},'trace':[]}
        return result,out
    def test_local(self):
        r,o=self.fixture();b=build_export(r);self.assertIsNone(b['agent_output'])
        self.assertEqual(b['export_validation']['status'],'not_requested')
        self.assertIn('RR_TRACE_UNAVAILABLE',markdown_bytes(b).decode())
    def test_valid_no_mutation(self):
        r,o=self.fixture();old=copy.deepcopy(o);b=build_export(r,question='Q',output=o,request_id='R')
        self.assertEqual(o,old);self.assertEqual(b['export_validation']['status'],'passed')
        self.assertEqual(json.loads(json_bytes(b))['request']['question'],'Q')
    def test_wrong_analysis(self):
        r,o=self.fixture();o['analysis_id']='B'
        with self.assertRaises(ValueError):build_export(r,question='Q',output=o,request_id='R')
    def test_wrong_evidence(self):
        r,o=self.fixture();o['evidence']['A:E']['analysis_id']='B'
        with self.assertRaises(ValueError):build_export(r,question='Q',output=o,request_id='R')
    def test_invalid_draft_preserved(self):
        r,o=self.fixture();o['draft']['observations'][0]['value']=99
        b=build_export(r,question='Q',output=o,request_id='R')
        self.assertEqual(b['export_validation']['status'],'failed')
        self.assertEqual(b['agent_output']['draft']['observations'][0]['value'],99)
    def test_failed_request(self):
        r,o=self.fixture();o.update(status='failed',draft={},error='GATEWAY_REQUEST_FAILED')
        b=build_export(r,question='Q',output=o,request_id='R');self.assertEqual(b['export_validation']['status'],'no_completed_draft')
    def test_nan_rejected(self):
        r,o=self.fixture();o['elapsed_seconds']=float('nan')
        with self.assertRaises(ValueError):build_export(r,question='Q',output=o,request_id='R')
    def test_configuration_allowlist(self):
        r,o=self.fixture();b=build_export(r,question='Q',output=o,request_id='R',configuration={'API_KEY':'secret','model':'model'})
        self.assertNotIn('secret',json_bytes(b).decode())
    def test_request_identity(self):
        state={'analysis_id':'B','request_id':'new'}
        self.assertFalse(request_matches(state,'A','old'));self.assertFalse(request_matches(state,'B','old'))
        self.assertTrue(request_matches(state,'B','new'))
    def test_question_required(self):
        r,o=self.fixture()
        with self.assertRaises(ValueError):build_export(r,output=o)
if __name__=='__main__':unittest.main()
