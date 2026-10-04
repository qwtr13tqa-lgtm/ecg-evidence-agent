import unittest
from test_cross_record_agent import History,REPORT,Gateway,call
from src.review.cross_record_agent import run
class Sequence(Gateway):
    def __init__(self,modes):super().__init__();self.modes=modes;self.requests=[]
    def complete(self,messages,tools):
        self.requests.append((messages,tools));reply=super().complete(messages,tools)
        mode=self.modes[min(self.n-1,len(self.modes)-1)]
        if mode=='length':return {'finish_reason':'length','content':'PARTIAL_MUST_NOT_REUSE','tool_calls':[call('get_analysis_summary',{'sample_index':0},'partial')]}
        if mode=='query':return {'finish_reason':'tool_calls','tool_calls':[call('get_analysis_summary',{'sample_index':0},'q')]}
        if mode=='wrong':
            import json
            a=json.loads(reply['tool_calls'][0]['function']['arguments']);a['observations'][0]['value']=999
            reply['tool_calls'][0]['function']['arguments']=json.dumps(a)
        return reply
class OutputTests(unittest.TestCase):
    def go(self,modes,**kwargs):
        g=Sequence(modes);return run(History(),g,'root','共性',REPORT,'hash',**kwargs),g
    def test_truncation_recovers(self):
        o,g=self.go(['length','ok']);self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['model_calls'],2)
        self.assertEqual(o['tool_calls'],0);self.assertEqual([t['function']['name'] for t in g.requests[-1][1]],['submit_answer'])
        self.assertNotIn('PARTIAL_MUST_NOT_REUSE',str(g.requests[-1][0]))
    def test_repeated_truncation_stops(self):
        o,g=self.go(['length']);self.assertEqual(o['error'],'MODEL_OUTPUT_TRUNCATED');self.assertEqual(o['model_calls'],2);self.assertEqual(o['draft'],{})
    def test_no_budget_no_recovery(self):
        o,g=self.go(['length'],max_model_calls=1);self.assertEqual(o['error'],'MODEL_OUTPUT_TRUNCATED');self.assertEqual(o['model_calls'],1)
    def test_preserves_evidence_without_requery(self):
        o,g=self.go(['query','length','ok']);self.assertEqual(o['status'],'completed_draft');self.assertEqual(o['tool_calls'],1);self.assertEqual(len(o['evidence']),2)
    def test_invalid_scalar_still_rejected(self):self.assertEqual(self.go(['length','wrong'])[0]['error'],'OBSERVATION_VALUE_MISMATCH')
    def test_trace_records_recovery(self):
        o,g=self.go(['length','ok']);self.assertEqual(sum(t['stage']=='truncation_repair' for t in o['trace']),1)
if __name__=='__main__':unittest.main()
