import json,unittest
from copy import deepcopy
from src.review.working_evidence_view import build_working_messages
from src.agent.answer_validator import validate_answer
from test_cross_record_agent import History,REPORT,call

class ContextTests(unittest.TestCase):
 def test_original_paths_and_values(self):
  e={'E':{'data':{'rows':[{'index':i,'score':float(i),'category':'fn'} for i in range(100)]}}};old=deepcopy(e)
  m,t=build_working_messages('policy','q',e,[],[]);v=json.loads(m[1]['content'])['working_evidence'][0]
  row=next(r for r in v['row_excerpts'] if r['data']['index']==99)
  a={'answer':'example','evidence_ids':['E'],'knowledge_ids':[],'observations':[{'evidence_id':'E','path':row['original_path']+'/score','value':99.0}]}
  self.assertTrue(validate_answer(a,e,{})['passed']);self.assertEqual(e,old);self.assertEqual(t['rows_omitted'],92)
 def test_small_results_complete(self):
  m,t=build_working_messages('p','q',{'E':{'data':{'rows':[{'index':1},{'index':2}]}}},[],[]);self.assertEqual(t['rows_omitted'],0)
 def test_group_statistics_complete(self):
  groups=[{'category':'fn','valid_n':16,'missing_n':0,'median':.3}]
  m,t=build_working_messages('p','q',{'E':{'data':{'groups':groups}}},[],[])
  self.assertEqual(json.loads(m[1]['content'])['working_evidence'][0]['data']['groups'],groups)
 def test_actions_preserved_and_no_orphan_messages(self):
  trace=[{'stage':'tool','tool':'x','arguments':{'index':1},'ok':False,'error':'bad'}]
  m,t=build_working_messages('p','q',{},[{'role':'tool','content':'OLD_PAYLOAD'}],trace)
  self.assertEqual(json.loads(m[1]['content'])['executed_actions'],trace);self.assertNotIn('OLD_PAYLOAD',str(m));self.assertEqual([x['role'] for x in m],['system','user','user'])
 def test_region_and_measurement_retained(self):
  row={'index':1,'region_coordinates':[{'start':41,'end':109}],'measurement_status':'unvalidated'}
  m,t=build_working_messages('p','q',{'E':{'data':{'rows':[row]}}},[],[])
  self.assertEqual(json.loads(m[1]['content'])['working_evidence'][0]['row_excerpts'][0]['data'],row)

class RuntimeTests(unittest.TestCase):
 def run_case(self,modes,snapshot=None,budget=4):
  from src.review.cross_record_agent import _run_agent
  class Gateway:
   def __init__(self):self.requests=[]
   def complete(self,messages,tools):
    n=len(self.requests);self.requests.append((deepcopy(messages),deepcopy(tools)));mode=modes[min(n,len(modes)-1)]
    view=json.loads(messages[1]['content'])['working_evidence'];eid=view[0]['evidence_id']
    if mode=='timeout':raise TimeoutError()
    if mode=='length':return {'finish_reason':'length','tool_calls':[],'content':'discard'}
    if mode=='query':return {'finish_reason':'tool_calls','tool_calls':[call('get_analysis_summary',{'sample_index':0},'q'+str(n))]}
    a={'answer':'FN有2条','evidence_ids':[eid],'knowledge_ids':[],'observations':[{'evidence_id':eid,'path':'/cohort_counts/fn','value':999 if mode=='wrong' else 2}]}
    return {'finish_reason':'tool_calls','tool_calls':[call('submit_answer',a,'s'+str(n))]}
  g=Gateway();o=_run_agent(History(),g,'root','比较',REPORT,'hash',max_model_calls=budget,snapshot=snapshot);return o,g
 def test_every_request_projected(self):
  o,g=self.run_case(['query','ok']);self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['tool_calls'],1)
  self.assertTrue(all('working_evidence' in json.loads(m[1]['content']) for m,_ in g.requests))
  self.assertIn('get_analysis_summary',g.requests[1][0][1]['content'])
 def test_validation_repair_still_works(self):
  o,g=self.run_case(['wrong','ok']);self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['model_calls'],2);self.assertIn('validation_feedback',str(g.requests[1]))
 def test_wrong_value_still_fails(self):self.assertEqual(self.run_case(['wrong'])[0]['error'],'OBSERVATION_VALUE_MISMATCH')
 def test_timeout_retains_evidence(self):
  o,g=self.run_case(['query','timeout']);self.assertEqual(o['error'],'GATEWAY_TIMEOUT');self.assertEqual(len(o['evidence']),2)
  d=next(t for t in o['trace'] if t.get('failure_phase')=='gateway');self.assertEqual(d['error_type'],'TimeoutError');self.assertGreater(d['context_characters'],0)
 def test_snapshot_retry_one_request_no_query(self):
  failed,_=self.run_case(['query','timeout']);o,g=self.run_case(['ok'],snapshot=failed,budget=1)
  self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['evidence'],failed['evidence']);self.assertEqual(o['tool_calls'],0);self.assertEqual(o['model_calls'],1)
  self.assertEqual([t['function']['name'] for t in g.requests[0][1]],['submit_answer'])
 def test_truncation_repair(self):self.assertEqual(self.run_case(['length','ok'])[0]['status'],'completed_draft')
if __name__=='__main__':unittest.main()
