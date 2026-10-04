import json
import unittest
from test_cross_record_agent import History,REPORT,Gateway,call
from src.review.cross_record_agent import run

class Watching(Gateway):
    def __init__(self,queries=None,wrong=False):
        super().__init__(queries,wrong);self.requests=[]
    def complete(self,messages,tools):
        self.requests.append((messages,tools))
        return super().complete(messages,tools)

class BudgetTests(unittest.TestCase):
    def test_full_budget_still_submits(self):
        g=Watching([call('get_analysis_summary',{'sample_index':0},'a')])
        o=run(History(),g,'root','比较',REPORT,'hash',max_tool_calls=1)
        self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['tool_calls'],1)
        self.assertEqual([t['function']['name'] for t in g.requests[-1][1]],['submit_answer'])
    def test_oversized_batch_preserves_protocol_and_evidence(self):
        g=Watching([call('get_analysis_summary',{'sample_index':i},str(i)) for i in (0,1)])
        o=run(History(),g,'root','比较',REPORT,'hash',max_tool_calls=1)
        self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['tool_calls'],1)
        replies=[json.loads(m['content']) for m in g.requests[-1][0] if m['role']=='tool']
        self.assertEqual(len(replies),2);self.assertTrue(replies[0]['ok'])
        self.assertFalse(replies[1]['executed']);self.assertEqual(len(o['evidence']),2)
    def test_last_model_call_reserved(self):
        g=Watching([call('get_analysis_summary',{'sample_index':0},'a')])
        o=run(History(),g,'root','比较',REPORT,'hash',max_model_calls=2)
        self.assertEqual(o['status'],'completed_draft')
        self.assertEqual([t['function']['name'] for t in g.requests[1][1]],['submit_answer'])
    def test_one_model_call_can_answer_bootstrap(self):
        g=Watching();o=run(History(),g,'root','比较',REPORT,'hash',max_model_calls=1)
        self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['tool_calls'],0)
        self.assertEqual(len(g.requests[0][1]),1)
    def test_noncompliant_final_call_does_not_execute_or_fake_success(self):
        g=Watching([call('get_analysis_summary',{'sample_index':0},'a')])
        o=run(History(),g,'root','比较',REPORT,'hash',max_model_calls=1)
        self.assertEqual(o['status'],'failed');self.assertEqual(o['tool_calls'],0)
        self.assertEqual(o['draft'],{});self.assertEqual(o['model_calls'],1)
    def test_wrong_value_still_fails_after_budget(self):
        g=Watching([call('get_analysis_summary',{'sample_index':0},'a')],wrong=True)
        o=run(History(),g,'root','比较',REPORT,'hash',max_tool_calls=1)
        self.assertEqual(o['error'],'OBSERVATION_VALUE_MISMATCH')
    def test_budget_denials_auditable(self):
        g=Watching([call('get_analysis_summary',{'sample_index':i},str(i)) for i in (0,1)])
        o=run(History(),g,'root','比较',REPORT,'hash',max_tool_calls=1)
        self.assertEqual(sum(t['stage']=='tool_rejected' for t in o['trace']),1)
        self.assertTrue(any(t.get('finalization_only') for t in o['trace']))
if __name__=='__main__':unittest.main()
