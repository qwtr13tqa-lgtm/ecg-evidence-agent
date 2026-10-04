import unittest,json
from copy import deepcopy
from src.review.final_evidence_view import project,build_final_messages,selected_rows
from src.agent.answer_validator import validate_answer
class FinalViewTests(unittest.TestCase):
 def fixture(self):
  return {'E':{'data':{'threshold':-0.9,'groups':[{'valid_n':50,'missing_n':0,'median':1.2}], 'rows':[{'index':i,'category':'fn' if i<50 else 'tp','score':i/100,'reconstruction_error':i,'shape_error':100-i,'long_unused':'x'*200} for i in range(100)]}}}
 def test_smaller(self):
  e=self.fixture();old=[{'content':json.dumps(e)}]*2;m,t=build_final_messages('policy','compare',e,old);self.assertLess(t['context_after'],t['context_before']);self.assertEqual(t['rows_retained'],12)
 def test_original_preserved(self):
  e=self.fixture();original=deepcopy(e);project(e);self.assertEqual(e,original)
 def test_original_pointers_validate(self):
  e=self.fixture();v,_=project(e);o=next(x for x in v[0]['scalars'] if x['path']=='/rows/99/score')
  a={'answer':'test','evidence_ids':['E'],'knowledge_ids':[],'observations':[{'evidence_id':'E',**o}]};self.assertTrue(validate_answer(a,e,{})['passed'])
 def test_group_statistics_preserved(self):
  v,_=project(self.fixture());self.assertIn({'path':'/groups/0/missing_n','value':0},v[0]['scalars'])
 def test_selection_both_groups_extremes(self):
  indices=selected_rows(self.fixture()['E']['data']['rows']);self.assertTrue({0,49,50,99}<=set(indices))
 def test_small_selection_not_cut(self):self.assertEqual(set(selected_rows([{'index':0},{'index':1}])),{0,1})
 def test_null_preserved(self):
  v,_=project({'E':{'data':{'unknown':None}}});self.assertEqual(v[0]['scalars'],[{'path':'/unknown','value':None}])
 def test_no_orphan_tool_messages(self):
  m,t=build_final_messages('p','q',self.fixture(),[{'role':'tool','tool_call_id':'old','content':'x'}]);self.assertEqual([x['role'] for x in m],['system','user','user']);self.assertEqual(m[-1]['content'],'q')
 def test_omission_explicit(self):
  v,t=project(self.fixture());self.assertEqual(v[0]['row_selection']['omitted_rows'],88)
if __name__=='__main__':unittest.main()

class RuntimeTests(unittest.TestCase):
 def run_case(self,modes):
  from src.review.cross_record_agent import _run_agent
  from test_cross_record_agent import History,REPORT,call
  class Gateway:
   def __init__(self):self.n=0;self.requests=[]
   def complete(self,messages,tools):
    self.requests.append(messages);mode=modes[min(self.n,len(modes)-1)];self.n+=1
    if mode=='length':return {'finish_reason':'length','tool_calls':[],'content':'PARTIAL_UNTRUSTED'}
    body=json.loads(messages[1]['content']);view=body.get('working_evidence',body.get('evidence_projection'));root=next(e for e in view if e['scope']=='explicit_batch_report')
    scalar={'path':'/record_count','value':root['reference_catalog']['record_count'][1]}
    answer={'answer':'本批记录数','evidence_ids':[root['evidence_id']],'knowledge_ids':[], 'observations':[{'evidence_id':root['evidence_id'],**scalar}]}
    if mode=='wrong':answer['observations'][0]['value']=999
    return {'finish_reason':'tool_calls','tool_calls':[call('submit_answer',answer,'submit')]}
  g=Gateway();o=_run_agent(History(),g,'root','共性',REPORT,'hash',max_model_calls=3);return o,g
 def test_runtime_repair(self):
  o,g=self.run_case(['length','ok']);self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['model_calls'],2);self.assertEqual(o['tool_calls'],0);self.assertNotIn('PARTIAL_UNTRUSTED',str(g.requests));self.assertTrue(any(t['stage'] in ('final_evidence_projection','working_evidence_projection') for t in o['trace']))
 def test_runtime_second_cut_fails(self):
  o,g=self.run_case(['length']);self.assertEqual(o['error'],'MODEL_OUTPUT_TRUNCATED');self.assertEqual(o['model_calls'],2);self.assertEqual(o['draft'],{})
 def test_runtime_wrong_value_fails(self):
  o,g=self.run_case(['length','wrong']);self.assertEqual(o['error'],'OBSERVATION_VALUE_MISMATCH')
