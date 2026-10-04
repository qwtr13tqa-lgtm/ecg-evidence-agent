import json
import unittest
from copy import deepcopy
from src.review.cross_record_agent import _run_agent
from src.review.submission_diagnostics import describe_failure
from src.agent.answer_validator import AnswerValidationError
from test_cross_record_agent import History, REPORT, call

class RepairTests(unittest.TestCase):
    def run_case(self, modes, budget=3, plain=False):
        class Gateway:
            def __init__(self):self.requests=[]
            def complete(self,messages,tools):
                n=len(self.requests);self.requests.append((deepcopy(messages),deepcopy(tools)))
                mode=modes[min(n,len(modes)-1)]
                first=json.loads(messages[1]['content'])
                eid=(first['working_evidence'][0]['evidence_id'] if 'working_evidence' in first else first['evidence_projection'][0]['evidence_id'] if 'evidence_projection' in first else first['evidence_id'])
                obs={'evidence_id':eid,'path':'/cohort_counts/fn','value':2}
                if mode=='wrong':obs['value']=999
                if mode=='type':obs['value']=2.0
                if mode=='path':obs['path']='/groups/999/median'
                if mode=='query':return {'finish_reason':'tool_calls','tool_calls':[call('get_analysis_summary',{'sample_index':0},'q'+str(n))]}
                if mode=='length':return {'finish_reason':'length','tool_calls':[],'content':'incomplete'}
                answer={'answer':'本批漏报2条，不能据此推断因果。','evidence_ids':[eid],'knowledge_ids':[],'observations':[obs]}
                if plain:return {'finish_reason':'stop','content':json.dumps(answer),'tool_calls':[]}
                return {'finish_reason':'tool_calls','tool_calls':[call('submit_answer',answer,'s'+str(n))]}
        g=Gateway();o=_run_agent(History(),g,'root','比较FN与TP',REPORT,'hash',max_model_calls=budget)
        return o,g
    def test_spontaneous_submit_repaired(self):
        o,g=self.run_case(['wrong','ok']);self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['model_calls'],2);self.assertEqual(o['tool_calls'],0)
        self.assertEqual([x['function']['name'] for x in g.requests[1][1]],['submit_answer'])
        self.assertIn('working_evidence',g.requests[1][0][1]['content'])
    def test_plain_json_repaired(self):self.assertEqual(self.run_case(['wrong','ok'],plain=True)[0]['status'],'completed_draft')
    def test_diagnostic_values(self):
        o,_=self.run_case(['wrong','ok']);d=next(t for t in o['trace'] if t['stage']=='submission_validation')
        self.assertEqual((d['submitted_value'],d['expected_value']),(999,2));self.assertTrue(d['path_exists']);self.assertIn('evidence_id',d)
    def test_second_wrong_fails(self):
        o,g=self.run_case(['wrong']);self.assertEqual(o['error'],'OBSERVATION_VALUE_MISMATCH');self.assertEqual(o['model_calls'],2);self.assertEqual(o['draft'],{})
    def test_type_still_strict(self):self.assertEqual(self.run_case(['type'])[0]['error'],'OBSERVATION_VALUE_MISMATCH')
    def test_invalid_path_repair(self):
        o,g=self.run_case(['path','ok']);self.assertEqual(o['status'],'completed_draft');d=next(t for t in o['trace'] if t['stage']=='submission_validation');self.assertFalse(d['path_exists'])
    def test_no_budget(self):
        o,g=self.run_case(['wrong'],budget=1);self.assertEqual(o['model_calls'],1);self.assertEqual(o['error'],'OBSERVATION_VALUE_MISMATCH')
    def test_query_after_repair_not_executed(self):
        o,g=self.run_case(['wrong','query','ok']);self.assertEqual(o['tool_calls'],0);self.assertEqual(o['model_calls'],2);self.assertEqual(o['error'],'FINALIZATION_QUERY_REJECTED');self.assertTrue(any(t['stage']=='tool_rejected' for t in o['trace']))
    def test_no_repair_chain_after_length(self):
        o,g=self.run_case(['length','wrong']);self.assertEqual(o['model_calls'],2);self.assertEqual(o['error'],'OBSERVATION_VALUE_MISMATCH')
    def test_no_repair_chain_after_validation(self):
        o,g=self.run_case(['wrong','length']);self.assertEqual(o['model_calls'],2);self.assertEqual(o['error'],'MODEL_OUTPUT_TRUNCATED')
    def test_success_no_extra_request(self):self.assertEqual(self.run_case(['ok'])[0]['model_calls'],1)
    def test_original_evidence_unchanged(self):
        e={'E':{'data':{'groups':[{'median':1.5}]}}};old=deepcopy(e)
        a={'observations':[{'evidence_id':'E','path':'/groups/0/median','value':2.5}]}
        d=describe_failure(AnswerValidationError('OBSERVATION_VALUE_MISMATCH',0,'/groups/0/median'),a,e)
        self.assertEqual(d['expected_value'],1.5);self.assertEqual(e,old)

if __name__=='__main__':unittest.main()
