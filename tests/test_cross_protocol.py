import json
import unittest
from test_cross_record_agent import History,REPORT,Gateway,call
from src.review.cross_record_agent import run

class ReplyGateway(Gateway):
    def __init__(self,mode):super().__init__();self.mode=mode;self.requests=[]
    def complete(self,messages,tools):
        self.requests.append((messages,tools))
        base=super().complete(messages,tools)
        answer=base['tool_calls'][0]['function']['arguments']
        if self.mode=='plain':return {'finish_reason':'stop','content':answer}
        if self.mode=='fence':return {'finish_reason':'stop','tool_calls':None,'content':'```json\n'+answer+'\n```'}
        if self.mode=='wrong':
            a=json.loads(answer);a['observations'][0]['value']=999
            return {'finish_reason':'stop','content':json.dumps(a)}
        if self.mode=='repair' and self.n>1:return base
        if self.mode in ('repair','prose'):return {'finish_reason':'stop','content':'初步可能有差异，需要复核。'}
        if self.mode=='length':return {'finish_reason':'length','content':answer}
        if self.mode=='badlist':return {'finish_reason':'tool_calls','tool_calls':{}}
        if self.mode=='empty':return {'finish_reason':'tool_calls','tool_calls':[]}
        if self.mode=='stopcall':base['finish_reason']='stop';return base

class ProtocolTests(unittest.TestCase):
    def go(self,mode,**kw):
        g=ReplyGateway(mode);return run(History(),g,'root','比较',REPORT,'hash',**kw),g
    def test_plain_json(self):self.assertEqual(self.go('plain')[0]['status'],'completed_draft')
    def test_fenced_json(self):self.assertEqual(self.go('fence')[0]['status'],'completed_draft')
    def test_scalar_still_validated(self):self.assertEqual(self.go('wrong')[0]['error'],'OBSERVATION_VALUE_MISMATCH')
    def test_one_bounded_format_repair(self):
        o,g=self.go('repair');self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['model_calls'],2)
        self.assertEqual([t['function']['name'] for t in g.requests[1][1]],['submit_answer'])
        self.assertEqual(o['tool_calls'],0)
    def test_prose_not_marked_success(self):
        o,_=self.go('prose');self.assertEqual(o['error'],'ANSWER_JSON_INVALID');self.assertEqual(o['model_calls'],2);self.assertEqual(o['draft'],{})
    def test_last_request_no_retry(self):self.assertEqual(self.go('prose',max_model_calls=1)[0]['model_calls'],1)
    def test_truncated_even_valid_json_rejected(self):self.assertEqual(self.go('length')[0]['error'],'MODEL_OUTPUT_TRUNCATED')
    def test_bad_list(self):self.assertEqual(self.go('badlist')[0]['error'],'TOOL_CALLS_NOT_LIST')
    def test_empty_calls(self):self.assertEqual(self.go('empty')[0]['error'],'EMPTY_TOOL_CALLS')
    def test_stop_with_calls(self):self.assertEqual(self.go('stopcall')[0]['status'],'completed_draft')
    def test_trace_has_protocol_metadata(self):
        o,_=self.go('plain');t=next(t for t in o['trace'] if t['stage']=='model')
        self.assertEqual(t['finish_reason'],'stop');self.assertGreater(t['content_length'],0)
if __name__=='__main__':unittest.main()
