import copy
import importlib.util
import json
import unittest
from unittest.mock import patch
from src.agent.efficient_queries import prepare,complete,policy_trace
from src.agent.tool_protocol import TOOLS
from src.agent.answer_validator import validate_answer


class Empty:
    knowledge_available=False
class Enabled:pass


class PolicyTests(unittest.TestCase):
    def setUp(self):self.messages=[{'role':'system','content':'original safety'},{'role':'user','content':'existing evidence'}]
    def names(self,retriever):return {x['function']['name'] for x in prepare(self.messages,TOOLS,retriever)[1]}
    def test_existing_summary_removed(self):self.assertNotIn('get_analysis_summary',self.names(Enabled()))
    def test_disabled_search_removed(self):self.assertNotIn('search_knowledge',self.names(Empty()))
    def test_unknown_retriever_not_disabled(self):self.assertIn('search_knowledge',self.names(Enabled()))
    def test_decision_and_window_remain(self):
        self.assertTrue({'get_model_decision','inspect_error_window','inspect_recent_error','inspect_rr_intervals','submit_answer'}<=self.names(Empty()))
    def test_no_mutation(self):
        before=copy.deepcopy((self.messages,TOOLS));prepare(self.messages,TOOLS,Empty());self.assertEqual((self.messages,TOOLS),before)
    def test_keeps_original_safety(self):self.assertTrue(prepare(self.messages,TOOLS,Empty())[0][0]['content'].startswith('original safety'))
    def test_one_gateway_request_no_auto_reply(self):
        class Gateway:
            count=0
            def complete(self,*args):self.count+=1;return {'finish_reason':'stop','content':'provider_reply'}
        g=Gateway();self.assertEqual(complete(g,self.messages,TOOLS,Empty())['content'],'provider_reply');self.assertEqual(g.count,1)
    def test_wrong_scalar_still_fails(self):
        with self.assertRaises(ValueError):validate_answer({'answer':'x','evidence_ids':['e'],'knowledge_ids':[],'observations':[{'evidence_id':'e','path':'/value','value':2}]},{'e':{'data':{'value':1}}},{})
    def test_policy_metadata_not_tool(self):self.assertEqual(policy_trace(Empty())['stage'],'query_policy')


@unittest.skipUnless(importlib.util.find_spec('langgraph'),'Requires project langgraph dependency')
class AgentIntegrationTests(unittest.TestCase):
    def run_agent(self,wrong=False,window=False):
        from src.agent.evidence_agent import ECGEvidenceAgent
        class Store:
            def get_result(self,aid):return object()
        class Executor:
            def __init__(self,*a):pass
            def execute(self,name,args):
                data={'input':{'num_samples':4800}} if name=='get_analysis_summary' else {'start_sample':4200,'end_sample':4800,'leads':[{'lead':'V2','mean':.25,'maximum':.5}]}
                return {'analysis_id':'a','ok':True,'evidence_id':'a:'+name,'data':data}
        class Gateway:
            count=0
            def complete(self,messages,tools):
                self.count+=1
                assert 'get_analysis_summary' not in {t['function']['name'] for t in tools}
                if window and self.count==1:
                    return {'finish_reason':'tool_calls','tool_calls':[{'id':'one','type':'function','function':{'name':'inspect_recent_error','arguments':json.dumps({'lead':'V2','duration_seconds':1.2})}}]}
                eid='a:inspect_recent_error' if window else 'a:get_analysis_summary'
                path='/leads/0/mean' if window else '/input/num_samples'
                value=999 if wrong else .25 if window else 4800
                answer={'answer':'software fixture','evidence_ids':[eid],'knowledge_ids':[],'observations':[{'evidence_id':eid,'path':path,'value':value}]}
                return {'finish_reason':'tool_calls','tool_calls':[{'id':'answer','type':'function','function':{'name':'submit_answer','arguments':json.dumps(answer)}}]}
        gateway=Gateway()
        with patch('src.agent.evidence_agent.ECGToolExecutor',Executor),patch('src.agent.evidence_agent.build_interpretation',lambda _: {'limitations':[]}):
            result=ECGEvidenceAgent(Store(),Empty(),gateway).run('a','V2最后1.2秒' if window else '输入有多少点',allow_external=True)
        return result,gateway.count
    def test_initial_evidence_one_request(self):
        out,n=self.run_agent();self.assertEqual(out['status'],'completed_draft');self.assertEqual(n,1)
        self.assertEqual(out['tool_calls'],1) # submit only; policy trace consumes no budget
    def test_missing_evidence_still_queries(self):
        out,n=self.run_agent(window=True);self.assertEqual(out['status'],'completed_draft');self.assertEqual(n,2);self.assertEqual(out['tool_calls'],2)
    def test_bad_value_cannot_pass(self):
        out,n=self.run_agent(wrong=True);self.assertEqual(out['status'],'failed')

if __name__=='__main__':unittest.main()
