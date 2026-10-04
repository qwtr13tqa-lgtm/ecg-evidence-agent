import ast
import importlib.util
import json
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import time
import unittest
from src.agent.truncation_repair import diagnostics,repair_messages
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'src/agent'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v=load('answer_validator')
class TruncationTests(unittest.TestCase):
 def setup_node(self,replies,budget=4,tools=6):
  self.requests=[]
  def complete(gateway,messages,available,retriever):
   self.requests.append((deepcopy(messages),deepcopy(available)));return replies.pop(0)
  node=next(n for n in ast.walk(ast.parse((ROOT/'src/agent/evidence_agent.py').read_text(encoding='utf-8-sig'))) if isinstance(n,ast.FunctionDef) and n.name=='decide')
  env={'__package__':'src.agent','self':SimpleNamespace(max_model_calls=budget,max_tool_calls=tools,gateway=None,retriever=None),
   'json':json,'time':time,'deepcopy':deepcopy,'efficient_complete':complete,
   'TOOLS':[{'function':{'name':n}} for n in ('submit_answer','inspect_error_window')],
   'failed':lambda c:{'status':'failed','error':c,'route':'stop','draft':{},'validation':{}},
   'strict_json':json.loads,'parse_answer':json.loads,'validate_answer':v.validate_answer,
   'check_submission':lambda *a:None,'reference':{},'AnswerValidationError':v.AnswerValidationError,
   'AnswerParseError':type('AnswerParseError',(Exception,),{}),'WindowReferenceError':type('WindowReferenceError',(Exception,),{})}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_decide','exec'),env)
  self.decide=env['decide']
  self.state={'messages':[{'role':'system','content':'policy'},{'role':'user','content':'question'}],
   'model_calls':0,'tool_calls':0,'trace':[],'call_ids':[],'evidence':{'E':{'data':{'score':1.25}}},'knowledge':{}}
 def cut(self):return {'finish_reason':'length','tool_calls':[],'content':'INCOMPLETE_SECRET','usage':{'completion_tokens':6000},'request_metadata':{'max_tokens':6000}}
 def answer(self,value=1.25):return {'answer':'score','evidence_ids':['E'],'knowledge_ids':[],'observations':[{'evidence_id':'E','path':'/score','value':value}]}
 def submit(self):return {'finish_reason':'tool_calls','tool_calls':[{'id':'new','type':'function','function':{'name':'submit_answer','arguments':json.dumps(self.answer())}}]}
 def test_repair_then_valid_submit(self):
  self.setup_node([self.cut(),self.submit()]);before=deepcopy(self.state['evidence']);self.state.update(self.decide(self.state));r=self.decide(self.state)
  self.assertEqual(r['status'],'completed_draft');self.assertEqual(r['model_calls'],2);self.assertEqual(self.state['evidence'],before)
  self.assertNotIn('INCOMPLETE_SECRET',json.dumps(self.requests));self.assertEqual([t['function']['name'] for t in self.requests[1][1]],['submit_answer'])
 def test_second_truncation_stops(self):
  self.setup_node([self.cut(),self.cut()]);self.state.update(self.decide(self.state));r=self.decide(self.state);self.assertEqual(r['error'],'MODEL_OUTPUT_TRUNCATED');self.assertEqual(r['route'],'stop')
 def test_no_model_budget_no_retry(self):
  self.setup_node([self.cut()],budget=1);r=self.decide(self.state);self.assertEqual(r['error'],'MODEL_OUTPUT_TRUNCATED');self.assertEqual(len(self.requests),1)
 def test_tool_budget_plain_answer(self):
  self.setup_node([self.cut(),{'finish_reason':'stop','tool_calls':[],'content':json.dumps(self.answer())}]);self.state['tool_calls']=6
  self.state.update(self.decide(self.state));r=self.decide(self.state);self.assertEqual(r['status'],'completed_draft');self.assertEqual(self.requests[1][1],[]);self.assertEqual(self.state['tool_calls'],6)
 def test_repair_query_blocked(self):
  self.setup_node([self.cut(),{'finish_reason':'tool_calls','tool_calls':[{'function':{'name':'inspect_error_window'}}]}]);self.state.update(self.decide(self.state));r=self.decide(self.state);self.assertEqual(r['error'],'FINALIZATION_QUERY_REJECTED')
 def test_wrong_value_not_accepted(self):
  self.setup_node([self.cut(),{'finish_reason':'stop','tool_calls':[],'content':json.dumps(self.answer(2))}]);self.state.update(self.decide(self.state));r=self.decide(self.state);self.assertEqual(r['error'],'OBSERVATION_VALUE_MISMATCH')
 def test_non_length_not_retried(self):
  self.setup_node([{'finish_reason':'content_filter','tool_calls':[]}]);r=self.decide(self.state);self.assertEqual(r['error'],'MODEL_PROTOCOL_INVALID');self.assertEqual(r['trace'][-1]['finish_reason'],'content_filter')
 def test_partial_tool_not_executed(self):
  cut=self.cut();cut['tool_calls']=[{'function':{'name':'inspect_error_window','arguments':'{'}}];self.setup_node([cut]);r=self.decide(self.state);self.assertEqual(r['route'],'decide');self.assertNotIn('pending',r)
 def test_metadata_sanitized(self):
  r=self.cut();r['usage']['secret']='private';self.assertNotIn('secret',str(diagnostics(r)));self.assertEqual(diagnostics(r)['request_metadata']['max_tokens'],6000)
 def test_messages_not_mutated(self):
  m=[{'role':'system','content':'x'}];self.assertNotEqual(repair_messages(m),m);self.assertEqual(m,[{'role':'system','content':'x'}])
class GatewayTests(unittest.TestCase):
 def test_usage_and_length_preserved(self):
  from src.agent.gateway import ToolGateway
  response=SimpleNamespace(choices=[SimpleNamespace(finish_reason='length',message=SimpleNamespace(tool_calls=None,content='partial'))],usage=SimpleNamespace(prompt_tokens=20,completion_tokens=10,total_tokens=30))
  result=ToolGateway._adapt(response)
  self.assertEqual(result['finish_reason'],'length');self.assertEqual(result['usage']['completion_tokens'],10)
 def test_empty_tools_omitted(self):
  from src.agent.gateway import ToolGateway
  g=ToolGateway.__new__(ToolGateway);g.model='test';g.max_tokens=6000;g.timeout=90;g.base_url='test'
  captured={}
  def create(**kw):
   captured.update(kw)
   return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(tool_calls=None,content='{}'))],usage=None)
  g.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  g.diagnostics=SimpleNamespace(begin=lambda *a:{},response=lambda *a:None,finish=lambda *a:None)
  r=g.complete([{'role':'user','content':'中文'}],[])
  self.assertNotIn('tools',captured);self.assertNotIn('tool_choice',captured);self.assertEqual(r['request_metadata']['max_tokens'],6000)
if __name__=='__main__':unittest.main()
