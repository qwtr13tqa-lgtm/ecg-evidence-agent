import unittest
from copy import deepcopy
from src.review.reliability_closeout import missing_plan,request_plan
from src.review.collection_runtime import run_collection
from evaluation.reliability_review import judge_payload,map_review,reliability_metrics
from evaluation.cross_record_boundary_v2 import fixture,SyntheticHistory

class Fake:
    def __init__(self,items):self.items=iter(items);self.calls=[]
    def complete(self,messages,tools):
        self.calls.append(deepcopy((messages,tools)));item=next(self.items)
        if isinstance(item,Exception):raise item
        return item

class ReliabilityTests(unittest.TestCase):
    def test_missing_full_match(self):
        self.assertIsNotNone(missing_plan('列出漏报中重构项缺失的所有记录，按索引升序。'))
        self.assertIsNotNone(missing_plan('筛选category=fn且reconstruction_error缺失的记录，按index升序列出全部索引。'))
    def test_extra_conditions_declined(self):
        for q in ['列出漏报中重构项缺失的所有记录，按索引升序，且score>0。','只看前5条：列出漏报中重构项缺失的所有记录，按索引升序。','列出漏报中重构项缺失的所有记录，按索引降序。','比较FN与TP']:
            self.assertIsNone(missing_plan(q))
    def test_missing_real_execution_no_gateway(self):
        data=fixture();g=Fake([])
        out=run_collection(SyntheticHistory(data),g,'test','列出漏报中重构项缺失的所有记录，按索引升序。',data['report'],'hash',None)
        self.assertEqual(out['status'],'completed_draft');self.assertEqual(out['model_calls'],0)
        rows=next(iter(out['evidence'].values()))['data']['rows'];self.assertEqual([r['index'] for r in rows],[1,2])
    def test_timeout_one_recovery_same_input(self):
        g=Fake([TimeoutError(),{'finish_reason':'tool_calls'}]);o={'model_calls':0,'trace':[]}
        request_plan(g,[{'content':'all original filters'}],[],o,6)
        self.assertEqual(o['model_calls'],2);self.assertEqual(g.calls[0],g.calls[1])
    def test_timeout_stops_twice(self):
        g=Fake([TimeoutError(),TimeoutError()]);o={'model_calls':0,'trace':[]}
        with self.assertRaisesRegex(ValueError,'GATEWAY_TIMEOUT'):request_plan(g,[],[],o,6)
        self.assertEqual(o['model_calls'],2)
    def test_insufficient_budget_no_retry(self):
        g=Fake([TimeoutError()]);o={'model_calls':0,'trace':[]}
        with self.assertRaisesRegex(ValueError,'GATEWAY_TIMEOUT'):request_plan(g,[],[],o,2)
        self.assertEqual(o['model_calls'],1)
    def test_other_failure_not_retried(self):
        g=Fake([ValueError('bad')]);o={'model_calls':0,'trace':[]}
        with self.assertRaisesRegex(ValueError,'GATEWAY_REQUEST_FAILED'):request_plan(g,[],[],o,6)
        self.assertEqual(o['model_calls'],1)
    def test_short_projection_preserves_values(self):
        r={'case':{'question':'q'},'reference_workflow':{},'output':{'draft':{'answer':'long-evidence-id','evidence_ids':['long-evidence-id']},'evidence':{'long-evidence-id':{'data':{'rows':[{'v':None},{'v':2}]},'observation_paths':['/rows/1/v']}},'trace':[]}}
        original=deepcopy(r);p,m=judge_payload(r)
        self.assertEqual(r,original);self.assertEqual(p['evidence']['E1']['data'],r['output']['evidence']['long-evidence-id']['data'])
        self.assertEqual(map_review({'claims':[{'evidence_ids':['E1']}]},m)['claims'][0]['evidence_ids'],['long-evidence-id'])
    def test_unknown_short_id_rejected(self):
        for ids in [['E2'],['E01'],['long-id'],[None]]:
            with self.assertRaisesRegex(ValueError,'JUDGE_UNKNOWN_SHORT_REFERENCE'):map_review({'claims':[{'evidence_ids':ids}]},{'E1':'long-id'})
    def test_metrics_first_vs_recovered_missing(self):
        def r(trace,passed):return {'output':{'trace':trace},'automatic':{'automatic_acceptance':passed},'seconds':1,'requests':[{}]}
        m=reliability_metrics([r([],True),r([{'stage':'planning_recovery'}],True),r([],False)],4)
        self.assertEqual(m['first_pass_rate'],.25);self.assertEqual(m['eventual_pass_rate'],.5);self.assertEqual(m['total_model_requests'],3)
if __name__=='__main__':unittest.main()
