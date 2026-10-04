import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
from src.evaluation.baselines import run_baseline
from src.evaluation.cases import load_cases
from evaluation.run_comparison import plan
from evaluation.compare_results import summarize
from src.evaluation.records import RunRecord, atomic_json

class Executor:
    def __init__(self,*a):pass
    def execute(self,name,args):
        return {'analysis_id':'A','ok':True,'evidence_id':'A:'+name,'data':{'value':1,'provenance':{'secret':'local'}}}

class Gateway:
    def __init__(self,fail=False):self.calls=[];self.fail=fail
    def complete(self,messages,tools):
        self.calls.append((messages,tools))
        if self.fail:raise TimeoutError()
        ans={'answer':'证据不足，不能补充未测量值。','evidence_ids':['A:get_analysis_summary'],
             'knowledge_ids':[],'observations':[{'evidence_id':'A:get_analysis_summary','path':'/value','value':1}]}
        return {'finish_reason':'tool_calls','tool_calls':[{'type':'function','id':'x','function':{'name':'submit_answer','arguments':json.dumps(ans)}}]}

class TestComparison(unittest.TestCase):
    def run_scheme(self,scheme,fail=False):
        g=Gateway(fail);r=S(search=lambda **kw:[]);store=S(get_result=lambda aid:S())
        with patch('src.evaluation.baselines.ECGToolExecutor',Executor),patch('src.evaluation.baselines.build_interpretation',return_value={'limitations':[]}):
            out=run_baseline(store,r,g,'A','QTc是多少',scheme)
        return out,g
    def test_summary(self):
        o,g=self.run_scheme('summary');self.assertEqual(o['status'],'completed_draft')
        self.assertEqual([t['tool'] for t in o['trace'] if t['stage']=='tool'],['get_analysis_summary'])
        self.assertNotIn('secret',json.dumps(g.calls))
        self.assertEqual([x['function']['name'] for x in g.calls[0][1]],['submit_answer'])
    def test_rag(self):
        o,g=self.run_scheme('summary_rag');self.assertEqual(o['trace'][1]['tool'],'search_knowledge')
        self.assertEqual(o['trace'][1]['arguments']['query'],'QTc是多少')
    def test_fixed(self):
        o,g=self.run_scheme('fixed_tools');self.assertEqual(o['tool_calls'],5)
        self.assertEqual([x['tool'] for x in o['trace'] if x['stage']=='tool'],['get_analysis_summary','inspect_rr_intervals','inspect_recent_error','search_knowledge'])
    def test_timeout(self):
        o,g=self.run_scheme('summary',True);self.assertEqual(len(g.calls),1)
        self.assertEqual(o['status'],'failed');self.assertEqual(o['draft'],{})
    def test_suite(self):
        cs,_=load_cases(Path(__file__).resolve().parents[1]/'evaluation/coverage_cases.jsonl')
        self.assertEqual(len(cs),18);self.assertEqual(len({c['sample_index'] for c in cs}),3)
        self.assertEqual(len({c['expectation']['kind'] for c in cs}),6)
    def test_plan(self):
        cases=[{'id':'a'},{'id':'b'}];schemes=['summary','fixed_tools','agent']
        self.assertEqual(plan(cases,schemes,2,17),plan(cases,schemes,2,17))
        self.assertEqual(len(plan(cases,schemes,2,17)),4)
    def test_summary_missing_and_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg={'batch_id':'B'}
            atomic_json(root/'B.manifest.json',{'config':cfg,'tasks':[{'case_id':'C','repetition':1,'schemes':['summary','agent']}]})
            r=RunRecord(root,{'id':'C'},{**cfg,'scheme':'summary','repetition':1})
            r.finish({'status':'failed','trace':[],'model_calls':1},{},{},wall_seconds=2)
            s=summarize(root,'B');self.assertEqual(s['groups']['summary']['completion_rate_recorded'],0)
            self.assertEqual(s['groups']['agent']['missing_runs'],1)
            self.assertEqual(s['groups']['summary']['reviews']['task_correct']['unscored'],1)
if __name__=='__main__':unittest.main()
