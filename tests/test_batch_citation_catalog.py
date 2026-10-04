import json,unittest
from copy import deepcopy
from unittest.mock import patch
from src.review.citation_catalog import catalog_messages,expand_answer
from src.review.working_evidence_view import build_working_messages
from src.agent.answer_validator import validate_answer,AnswerValidationError
from src.review.cross_record_agent import _run_agent
from test_cross_record_agent import History,REPORT,call

def make(aid='root'):
 eid=aid+':overview'
 groups=[{'category':'fn','field':'shape_error','median':0.04284530443449815},
         {'category':'tp','field':'score','median':-0.8660851816336314},
         {'category':'tp','field':'reconstruction_error','median':-0.8889720936616261}]
 e={eid:{'analysis_id':aid,'scope':'explicit_batch_report','data':{'groups':groups,'rows':[{'index':i,'score':float(i)} for i in range(100)]}}}
 m,_=build_working_messages('p','q',e,[],[]);return e,*catalog_messages(m,e)

class CatalogTests(unittest.TestCase):
 def test_correct_bindings_not_array_counting(self):
  e,m,reg=make();refs=json.loads(m[1]['content'])['working_evidence'][0]['reference_catalog']['groups']
  ids=[refs[i]['median'][0] for i in (1,2)]
  a,_=expand_answer({'answer':'TP分数{{'+ids[0]+'}}','citation_ids':ids},reg)
  self.assertEqual([o['path'] for o in a['observations']],['/groups/1/median','/groups/2/median'])
  self.assertTrue(validate_answer(a,e,{})['passed']);self.assertIn('-0.866085',a['answer'])
 def test_regression_tp_values_bind_to_24_and_26(self):
  eid='regression:overview';groups=[{'category':'unused','median':0.0} for _ in range(32)]
  for i,category,field,value in [(3,'fn','shape_error',0.04284530443449815),(7,'fn','rr_cv',0.024172920608987084),(24,'tp','score',-0.8660851816336314),(26,'tp','reconstruction_error',-0.8889720936616261)]:
   groups[i]={'category':category,'field':field,'median':value}
  e={eid:{'scope':'explicit_batch_report','data':{'groups':groups}}};m,_=build_working_messages('p','q',e,[],[]);m,r=catalog_messages(m,e)
  g=json.loads(m[1]['content'])['working_evidence'][0]['reference_catalog']['groups']
  a,_=expand_answer({'answer':'TP对比','citation_ids':[g[24]['median'][0],g[26]['median'][0]]},r)
  self.assertEqual([o['path'] for o in a['observations']],['/groups/24/median','/groups/26/median']);self.assertTrue(validate_answer(a,e,{})['passed'])
 def test_unknown_ref_rejected(self):
  e,m,r=make()
  with self.assertRaisesRegex(AnswerValidationError,'CITATION_UNKNOWN'):expand_answer({'answer':'x','citation_ids':['bogus']},r)
 def test_cross_run_ref_rejected(self):
  e,m,r=make();r2=make('other')[2]
  with self.assertRaisesRegex(AnswerValidationError,'CITATION_UNKNOWN'):expand_answer({'answer':'x','citation_ids':[next(iter(r))]},r2)
 def test_duplicate_rejected(self):
  e,m,r=make();ref=next(iter(r))
  with self.assertRaisesRegex(AnswerValidationError,'CITATION_DUPLICATE'):expand_answer({'answer':'x','citation_ids':[ref,ref]},r)
 def test_placeholder_must_be_cited(self):
  e,m,r=make();ids=list(r)
  with self.assertRaisesRegex(AnswerValidationError,'CITATION_PLACEHOLDER_UNBOUND'):expand_answer({'answer':'{{'+ids[1]+'}}','citation_ids':[ids[0]]},r)
 def test_no_mixed_payload(self):
  with self.assertRaisesRegex(AnswerValidationError,'CITATION_SCHEMA_INVALID'):expand_answer({'answer':'x','citation_ids':['x'],'observations':[]},{})
 def test_evidence_unchanged_and_original_row_indices(self):
  e,m,r=make();old=deepcopy(e);obs=next(v for v in r.values() if v['path']=='/rows/99/score');self.assertEqual(obs['value'],99.0)
  self.assertEqual(e,old);self.assertFalse(any('/row_excerpts' in v['path'] for v in r.values()))
 def test_legacy_wrong_value_still_rejected(self):
  e,m,r=make();obs=deepcopy(next(iter(r.values())));obs['value']=999
  a,_=expand_answer(dict(answer='x',evidence_ids=[obs['evidence_id']],knowledge_ids=[],observations=[obs]),r)
  with self.assertRaisesRegex(AnswerValidationError,'OBSERVATION_VALUE_MISMATCH'):validate_answer(a,e,{})
 def test_refs_stable_if_order_changes(self):
  e,m,r=make();m2=deepcopy(m)
  # identical evidence projection deterministically produces identical references
  msg,_=build_working_messages('p','q',e,[],[]);self.assertEqual(r,catalog_messages(msg,e)[1])

class RuntimeTests(unittest.TestCase):
 def run_case(self,modes,plain=False):
  class Gateway:
   def __init__(self):self.n=0
   def complete(self,messages,tools):
    mode=modes[min(self.n,len(modes)-1)];self.n+=1
    params=next(t['function']['parameters'] for t in tools if t['function']['name']=='submit_answer')
    assert params['required']==['answer','citation_ids']
    root=json.loads(messages[1]['content'])['working_evidence'][0]['reference_catalog']
    ref=root['cohort_counts']['fn'][0]
    a={'answer':'FN数量{{'+ref+'}}','citation_ids':['bad'] if mode=='bad' else [ref]}
    if plain:return {'finish_reason':'stop','content':json.dumps(a),'tool_calls':[]}
    return {'finish_reason':'tool_calls','tool_calls':[call('submit_answer',a,'s'+str(self.n))]}
  g=Gateway();o=_run_agent(History(),g,'root','比较',REPORT,'hash',max_model_calls=3);return o
 def test_catalog_tool_submission(self):
  o=self.run_case(['ok']);self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['draft']['answer'],'FN数量2');self.assertEqual(o['model_calls'],1)
 def test_catalog_plain_submission(self):self.assertEqual(self.run_case(['ok'],True)['status'],'completed_draft')
 def test_unknown_once_repaired(self):
  o=self.run_case(['bad','ok']);self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['model_calls'],2)
 def test_second_bad_stops(self):
  o=self.run_case(['bad']);self.assertEqual(o['error'],'CITATION_UNKNOWN');self.assertEqual(o['draft'],{});self.assertEqual(o['model_calls'],2)
if __name__=='__main__':unittest.main()
