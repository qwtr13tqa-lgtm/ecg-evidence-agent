import json,os,unittest
from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace
from src.review.query_reuse import find_reusable
from src.review.collection_ops import execute,evidence
from src.review.working_evidence_view import build_working_messages
from src.review.cross_record_agent import _run_agent
from test_cross_record_agent import History,REPORT,call

def operation(**kw):
 d=dict(kind='select',filters=[dict(field='category',op='eq',value='fn')],order_by='score',direction='asc',limit=40,offset=0,columns=['score','reconstruction_error'],group_by=None,metrics=[]);d.update(kw);return d

def fixture():
 rows=[dict(index=i,category='fn',score=float(i),reconstruction_error=-float(i),source_analysis_id='s',history_status='matched') for i in range(40)]
 e=evidence('root','hash',execute(rows,operation()));return {e['evidence_id']:e},e['evidence_id']

class CacheTests(unittest.TestCase):
 def test_subset_reuses_original_page(self):
  es,eid=fixture();old=deepcopy(es);op=operation(columns=['score'],offset=10,limit=5)
  hit=find_reusable(es,op,'root','hash');self.assertEqual(hit,(eid,list(range(10,15))));self.assertEqual(es,old)
  trace=[dict(stage='evidence_reuse',evidence_id=eid,row_indices=hit[1]),dict(stage='tool',tool='query_batch_collection',ok=True,cache_hit=True,evidence_id=eid)]
  m,_=build_working_messages('p','q',es,[],trace);v=json.loads(m[1]['content'])['working_evidence'][0]
  self.assertEqual(v['row_excerpts'][0]['original_path'],'/rows/10');self.assertEqual(len(v['row_excerpts']),5)
 def test_scope_rejected(self):
  es,_=fixture();self.assertIsNone(find_reusable(es,operation(),'other','hash'));self.assertIsNone(find_reusable(es,operation(),'root','other'))
 def test_missing_field_new_sort_and_filter_not_reused(self):
  es,_=fixture()
  for op in [operation(columns=['rr_cv']),operation(direction='desc'),operation(filters=[])]:self.assertIsNone(find_reusable(es,op,'root','hash'))
 def test_incomplete_page_not_reused(self):
  es,eid=fixture();es[eid]['data']['has_more']=True
  self.assertIsNone(find_reusable(es,operation(limit=50),'root','hash'))
 def test_null_string_stays_invalid(self):
  with self.assertRaisesRegex(ValueError,'SORT_INVALID'):find_reusable({},operation(order_by='null'),'root','hash')
 def test_explicit_result_not_resampled(self):
  es,eid=fixture();m,_=build_working_messages('p','q',es,[],[dict(stage='tool',tool='query_batch_collection',ok=True,evidence_id=eid)])
  self.assertEqual(len(json.loads(m[1]['content'])['working_evidence'][0]['row_excerpts']),40)

class Gateway:
 timeout=90
 def __init__(self,modes):self.modes=modes;self.requests=[]
 def complete_with_timeout(self,messages,tools,timeout):
  n=len(self.requests);self.requests.append((deepcopy(messages),deepcopy(tools),timeout));mode=self.modes[min(n,len(self.modes)-1)]
  if mode=='timeout':raise TimeoutError()
  if mode=='connection':raise ConnectionError()
  if mode=='query':return dict(finish_reason='tool_calls',tool_calls=[call('get_analysis_summary',dict(sample_index=0),'q'+str(n))])
  if mode=='reuse':return dict(finish_reason='tool_calls',tool_calls=[call('query_batch_collection',operation(columns=['score'],offset=10,limit=5),'q'+str(n))])
  eid=json.loads(messages[1]['content'])['working_evidence'][0]['evidence_id']
  answer=dict(answer='FN 2条',evidence_ids=[eid],knowledge_ids=[],observations=[dict(evidence_id=eid,path='/cohort_counts/fn',value=999 if mode=='wrong' else 2)])
  return dict(finish_reason='tool_calls',tool_calls=[call('submit_answer',answer,'s'+str(n))])

class RuntimeTests(unittest.TestCase):
 def run_case(self,modes,budget=3,**kw):
  g=Gateway(modes);o=_run_agent(History(),g,'root','比较',REPORT,'hash',max_model_calls=budget,**kw);return o,g
 def test_final_timeout_once_then_success(self):
  o,g=self.run_case(['query','timeout','ok']);self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['tool_calls'],1);self.assertEqual(o['model_calls'],3)
  self.assertEqual([x['function']['name'] for x in g.requests[1][1]],['submit_answer']);self.assertLessEqual(g.requests[1][2],120)
  self.assertEqual(sum(t['stage']=='timeout_repair' for t in o['trace']),1)
 def test_second_timeout_fails_keeps_evidence(self):
  o,g=self.run_case(['query','timeout']);self.assertEqual(o['error'],'GATEWAY_TIMEOUT');self.assertEqual(len(g.requests),3);self.assertEqual(len(o['evidence']),2);self.assertEqual(o['draft'],{})
 def test_nonfinal_timeout_no_retry(self):
  o,g=self.run_case(['timeout']);self.assertEqual(o['error'],'GATEWAY_TIMEOUT');self.assertEqual(len(g.requests),1)
 def test_connection_not_retried(self):
  o,g=self.run_case(['connection']);self.assertEqual(o['error'],'GATEWAY_REQUEST_FAILED');self.assertEqual(len(g.requests),1)
 def test_no_budget_no_retry(self):
  o,g=self.run_case(['timeout'],budget=1);self.assertEqual(len(g.requests),1)
 def test_no_repair_chain(self):
  o,g=self.run_case(['query','timeout','wrong'],budget=3);self.assertEqual(o['error'],'OBSERVATION_VALUE_MISMATCH');self.assertEqual(len(g.requests),3)
 def test_reuse_no_collection_execution(self):
  es,eid=fixture()
  with patch('src.review.cross_record_agent.collection_execute',side_effect=AssertionError('must reuse')):
   o,g=self.run_case(['reuse','ok'],collection_seed=list(es.values()))
  self.assertEqual(o['status'],'completed_draft');self.assertEqual(sum(t.get('cache_hit',False) for t in o['trace']),1)
  self.assertIn('/rows/10',g.requests[1][0][1]['content']);self.assertEqual(len(o['evidence']),2)
 def test_expired_budget_no_gateway_request(self):
  with patch('src.review.cross_record_agent.time.perf_counter',side_effect=[0,241,242]):
   o,g=self.run_case(['ok'])
  self.assertEqual(o['error'],'TASK_TIME_BUDGET_EXHAUSTED');self.assertEqual(len(g.requests),0)
 def test_total_budget_caps_request(self):
  with patch.dict(os.environ,{'ECG_BATCH_TOTAL_SECONDS':'10'}):o,g=self.run_case(['ok'])
  self.assertLessEqual(g.requests[0][2],10)

class AdapterTests(unittest.TestCase):
 def test_timeout_override_local_to_request(self):
  from src.agent.gateway import ToolGateway
  g=ToolGateway.__new__(ToolGateway);g.timeout=90;g.model='m';g.max_tokens=6000;g.base_url='unused';captured=[]
  g.diagnostics=SimpleNamespace(begin=lambda *a:None,finish=lambda *a:None,response=lambda *a:None)
  def create(**kw):captured.append(kw);return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{}',tool_calls=[]))])
  g.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  g.complete_with_timeout([],[],120);g.complete([],[])
  self.assertEqual([c['timeout'] for c in captured],[120,90]);self.assertEqual(g.timeout,90)
if __name__=='__main__':unittest.main()
