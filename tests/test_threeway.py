import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.evaluation.threeway_routes import route
from src.evaluation.threeway_metrics import outcome,summarize,load_batch,save_review,conversation_outcomes
from src.evaluation.records import RunRecord


class ThreewayTests(unittest.TestCase):
    def test_route_window(self):
        self.assertEqual(route('查看V2最后1.2秒，给出均值')[0],[('inspect_recent_error',{'lead':'V2','duration_seconds':1.2})])
    def test_route_rr(self):
        self.assertEqual(route('原始RR中最长的是哪一个')[0][0][0],'inspect_rr_intervals')
    def test_route_alignment(self):
        self.assertEqual(route('V1最后0.6秒与多少RR重叠')[0][0][0],'inspect_recent_rr_alignment')
    def test_route_unknown_no_guess(self):
        self.assertEqual(route('比较V1和V2最后1秒')[0],[])
    def test_route_conditional_no_fake_planner(self):
        self.assertEqual(route('如果误差更大则向前扩窗')[0],[])
    def test_pending_not_success(self):
        r={'output':{'status':'completed_draft'},'automatic':{'metrics':{'task_observations':True}}}
        self.assertEqual(outcome(r,{}),'pending')
    def test_auto_failure_overrides_manual(self):
        r={'output':{'status':'completed_draft'},'automatic':{'metrics':{'task_observations':False}}}
        self.assertEqual(outcome(r,dict.fromkeys(('task_correct','evidence_support','text_complete'),'pass')),'fail')
    def test_timeout_failure(self):
        self.assertEqual(outcome({'output':{'status':'failed'}},{}),'fail')
    def test_sixth_turn_shared_resolver(self):
        from src.agent.window_memory import resolve_reference
        eid='a:window';w={'start_sample':4200,'end_sample':4800,'leads':[{'lead':'V2','peak_sample':4765}]}
        out={'analysis_id':'a','evidence':{eid:{'analysis_id':'a','evidence_id':eid,'ok':True,'data':w}},'trace':[{'stage':'tool','tool':'inspect_recent_error','arguments':{'lead':'V2'},'ok':True,'evidence_id':eid}]}
        session={'analysis_id':'a','turns':[{'analysis_id':'a','output':out}]+[{'analysis_id':'a','output':{'analysis_id':'a'}} for _ in range(4)]}
        before=copy.deepcopy(session)
        res=resolve_reference(session,'a','回到第一轮那个窗口')
        self.assertEqual(route('回到第一轮那个窗口',res)[0],[('inspect_error_window',{'lead':'V2','start_sample':4200,'end_sample':4800})])
        self.assertEqual(session,before)
        with self.assertRaises(ValueError):resolve_reference(session,'b','第一轮窗口')
    def test_ambiguous_clarifies(self):
        self.assertEqual(route('第一轮窗口',{'status':'needs_clarification','message':'两个窗口'})[0],[])
    def fixture(self,root):
        cfg={'batch_id':'batch','protocol':'threeway-1.0','case_id':'c','scheme':'rules','repetition':1}
        (root/'batch.manifest.json').write_text(json.dumps({'config':cfg,'tasks':[{'case_id':'c','scheme':'rules','repetition':1,'turn_count':2}]}))
        record=RunRecord(root,{'turn_index':1,'question':'q','expectation':{'kind':'window'}},cfg)
        record.finish({'status':'completed_draft','evidence':{'x':{}},'trace':[{'stage':'tool','source':'bootstrap','ok':True},{'stage':'tool','tool':'inspect_error_window','arguments':{},'ok':True}], 'draft':{'answer':'text'}},{},{'metrics':{'structure_values_references':True,'task_observations':True},'request_measurements':[{'status':'returned','elapsed_seconds':1}]},wall_seconds=2)
        return record.path/'result.json'
    def test_manifest_metadata_and_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp));m,rows,w=load_batch(tmp,'batch')
            self.assertEqual(rows[0]['scheme'],'rules');self.assertEqual(rows[0]['query_calls'],1)
            self.assertTrue(w);self.assertEqual(summarize(rows)[0]['pending'],1)
            self.assertEqual(conversation_outcomes(m,rows)[0]['outcome'],'pending')
    def test_review_preserves_result_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.fixture(Path(tmp));before=path.read_bytes()
            save_review(path,'tester',dict.fromkeys(('task_correct','evidence_support','text_complete'),'pass'),'checked')
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(load_batch(tmp,'batch')[1][0]['outcome'],'pass')
            path.write_bytes(before+b' ')
            _,rows,w=load_batch(tmp,'batch');self.assertEqual(rows[0]['outcome'],'pending');self.assertTrue(w)
    def test_baselines_use_real_validator_and_one_call(self):
        from src.evaluation.threeway_baselines import run_fixed
        class Executor:
            def __init__(self,*a):pass
            def execute(self,name,args):
                data={'input':{'num_samples':4800}} if name=='get_analysis_summary' else {'start_sample':4200,'end_sample':4800,'leads':[{'lead':'V2','mean':.25,'maximum':.5}]}
                return {'analysis_id':'a','evidence_id':'a:'+name,'ok':True,'data':data}
        class Gateway:
            def __init__(self):self.calls=0
            def complete(self,messages,tools):
                self.calls+=1
                payload=json.loads(messages[1]['content']);e=payload['evidence'];eid='a:inspect_recent_error' if 'a:inspect_recent_error' in e else 'a:get_analysis_summary'
                path='/leads/0/mean' if eid.endswith('recent_error') else '/input/num_samples'
                value=.25 if eid.endswith('recent_error') else 4800
                answer={'answer':'test','evidence_ids':[eid],'knowledge_ids':[],'observations':[{'evidence_id':eid,'path':path,'value':value}]}
                return {'finish_reason':'stop','content':json.dumps(answer),'tool_calls':[]}
        with patch('src.evaluation.threeway_baselines.ECGToolExecutor',Executor):
            for scheme,n in [('summary',1),('rules',2)]:
                gw=Gateway();out=run_fixed(None,'a','V2最后1.2秒均值',scheme,gw)
                self.assertEqual(out['status'],'completed_draft');self.assertEqual(gw.calls,1)
                self.assertEqual(len([x for x in out['trace'] if x['stage']=='tool']),n)

if __name__=='__main__':unittest.main()
