import unittest
from types import SimpleNamespace as NS
import numpy as np
from src.evaluation.complex_plan_v1 import expression,execute_plan
from src.evaluation.complex_cases_v1 import reference,score

class FakeExecutor:
    def __init__(self):self.calls=[]
    def execute(self,name,args):
        self.calls.append((name,args))
        return {'ok':True,'analysis_id':'a','evidence_id':'a:1','data':{'start_sample':args.get('start_sample',0)}}

def result(regions=None):
    x=np.zeros((4800,12));x[4600,6]=3;x[4500,7]=2
    return NS(analysis_id='a',input=NS(num_samples=4800,sampling_rate=500),model=NS(error_map=x),
        rhythm=NS(r_peaks=[0,1000,1500,2200,2900],rr_details={'valid_mask':[False,True,True,True]}),
        to_llm_context=lambda:{'evidence':{'temporal_regions':regions or []}})

class ComplexTests(unittest.TestCase):
    def test_lazy_branch(self):
        self.assertEqual(expression({'op':'if','args':[True,1,{'ref':'missing','path':''}]},{}),1)
    def test_dependency_execution(self):
        ex=FakeExecutor();pool={};trace=[]
        execute_plan({'steps':[{'id':'start','value':{'op':'sub','args':[4800,600]}},
            {'id':'w','tool':'inspect_error_window','args':{'lead':'V2','start_sample':{'ref':'start','path':''},'end_sample':4800}}]},ex,{},pool,trace)
        self.assertEqual(ex.calls[0][1]['start_sample'],4200)
    def test_scope_rejected(self):
        with self.assertRaises(ValueError):execute_plan({'steps':[{'id':'w','tool':'inspect_error_window','args':{'analysis_id':'other'}}]},FakeExecutor(),{}, {},[])
    def test_unknown_code_rejected(self):
        with self.assertRaises(ValueError):expression({'op':'eval','args':['1+1']},{})
    def test_forward_reference_rejected(self):
        with self.assertRaises(KeyError):expression({'ref':'future','path':''},{})
    def test_duplicate_query_rejected(self):
        with self.assertRaises(ValueError):execute_plan({'steps':[{'id':i,'tool':'get_analysis_summary','args':{}} for i in ['one','two']]},FakeExecutor(),{},{},[])
    def test_query_cap(self):
        with self.assertRaises(ValueError):execute_plan({'steps':[{'id':'one','tool':'get_analysis_summary','args':{}}]},FakeExecutor(),{},{},[],max_queries=0)
    def test_v1_v2_and_tie(self):
        r=result();self.assertEqual(reference(r,'SELECT_LEAD')['branch'],'V1')
        r.model.error_map[4500,7]=4;self.assertEqual(reference(r,'SELECT_LEAD')['branch'],'V2')
        r.model.error_map[4500,7]=3;self.assertEqual(reference(r,'SELECT_LEAD')['branch'],'V1')
    def test_longest_retained_tie(self):
        ref=reference(result(),'RR_WINDOW');self.assertEqual(ref['targets'][0]['value'],2)
        self.assertEqual(ref['windows'][0]['start_sample'],1500)
    def test_region_and_fallback(self):
        r=reference(result(),'REGION_FALLBACK');self.assertEqual(r['branch'],'fallback');self.assertEqual(r['windows'][0]['start_sample'],4500)
        r=reference(result([{'start':100,'end':300,'score':.9},{'start':400,'end':600,'score':.9}]),'REGION_FALLBACK')
        self.assertEqual(r['branch'],'region');self.assertEqual(r['windows'][0]['start_sample'],100)
    def test_empty_rr(self):
        r=result();r.rhythm.rr_details['valid_mask']=[False]*4
        with self.assertRaises(ValueError):reference(r,'RR_WINDOW')
    def test_failure_counts_zero_coverage(self):
        s=score({'status':'failed','analysis_id':'a','draft':{},'evidence':{}},reference(result(),'SELECT_LEAD'))
        self.assertEqual(s['target_covered'],0);self.assertEqual(s['target_total'],7)
    def test_wrong_analysis_rejected(self):
        s=score({'status':'completed_draft','analysis_id':'wrong','draft':{},'evidence':{}},reference(result(),'SELECT_LEAD'))
        self.assertFalse(s['structure_pass'])
    def test_filter_and_argmax(self):
        rows=[{'keep':False,'n':9},{'keep':True,'n':2},{'keep':True,'n':3}]
        x={'op':'argmax','args':[{'op':'filter_eq','args':[rows,'keep',True]},'n']}
        self.assertEqual(expression(x,{})['n'],3)

    def test_scoring_correct_and_wrong_window(self):
        ref=reference(result(),'REGION_FALLBACK')
        data={'start_sample':4500,'end_sample':4800,'leads':[{'lead':'V1','mean':.01,'maximum':3.0}]}
        obs=[{'evidence_id':'a:w','path':path,'value':value} for path,value in [('/start_sample',4500),('/end_sample',4800),('/leads/0/lead','V1'),('/leads/0/mean',.01),('/leads/0/maximum',3.0)]]
        out={'analysis_id':'a','status':'completed_draft','evidence':{'a:w':{'analysis_id':'a','data':data}},
             'draft':{'answer':'ok','evidence_ids':['a:w'],'knowledge_ids':[],'observations':obs},
             'trace':[{'stage':'tool','tool':'inspect_error_window','arguments':{},'ok':True,'evidence_id':'a:w'}]}
        self.assertTrue(score(out,ref)['automatic_task_pass'])
        data['leads'][0]['lead']='V2';obs[2]['value']='V2'
        self.assertTrue(score(out,ref)['structure_pass'])
        self.assertFalse(score(out,ref)['automatic_task_pass'])

    def test_plan_gateway_integration(self):
        import json
        from unittest.mock import patch
        from src.evaluation.complex_plan_v1 import run_plan
        class Executor:
            def __init__(self,*args):pass
            def execute(self,name,args):
                data={'input':{'num_samples':4800,'sampling_rate':500}} if name=='get_analysis_summary' else {'start_sample':4200}
                return {'ok':True,'analysis_id':'a','evidence_id':'a:'+name,'data':data}
        class Gateway:
            def __init__(self):self.calls=0
            def complete(self,messages,tools):
                self.calls+=1
                if self.calls==1:
                    plan={'steps':[{'id':'w','tool':'inspect_error_window','args':{'lead':'V2','start_sample':4200,'end_sample':4800}}]}
                    value={'plan_json':json.dumps(plan)};name='submit_plan'
                else:
                    value={'answer':'4200','evidence_ids':['a:inspect_error_window'],'knowledge_ids':[],
                        'observations':[{'evidence_id':'a:inspect_error_window','path':'/start_sample','value':4200}]};name='submit_answer'
                return {'finish_reason':'tool_calls','tool_calls':[{'type':'function','function':{'name':name,'arguments':json.dumps(value)}}]}
        with patch('src.tools.executor.ECGToolExecutor',Executor):
            gateway=Gateway();out=run_plan(None,'a','query',gateway)
        self.assertEqual(out['status'],'completed_draft',out)
        self.assertEqual(gateway.calls,2);self.assertEqual(out['supplemental_queries'],1)

if __name__=='__main__':unittest.main()
