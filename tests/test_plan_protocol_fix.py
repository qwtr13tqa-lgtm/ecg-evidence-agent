import json
import unittest
from src.evaluation.complex_plan_v1 import parse_plan_reply,response_metadata
from evaluation.run_complex_v1 import Meter

def call(name='submit_plan',arguments=None):
    return {'type':'function','function':{'name':name,'arguments':json.dumps({'plan_json':json.dumps({'steps':[]})}) if arguments is None else arguments}}
class ProtocolTests(unittest.TestCase):
    def test_native_submit(self):
        self.assertEqual(parse_plan_reply({'finish_reason':'tool_calls','tool_calls':[call()]}),{'steps':[]})
    def test_plain_json(self):
        self.assertEqual(parse_plan_reply({'finish_reason':'stop','content':'{"steps":[]}'}),{'steps':[]})
    def test_wrong_tool(self):
        with self.assertRaisesRegex(ValueError,'PLAN_WRONG_TOOL'):parse_plan_reply({'finish_reason':'tool_calls','tool_calls':[call('inspect_rr_intervals')]})
    def test_truncated(self):
        with self.assertRaisesRegex(ValueError,'PLAN_OUTPUT_TRUNCATED'):parse_plan_reply({'finish_reason':'length','tool_calls':[]})
    def test_multiple(self):
        with self.assertRaisesRegex(ValueError,'PLAN_SUBMIT_MUST_BE_ALONE'):parse_plan_reply({'finish_reason':'tool_calls','tool_calls':[call(),call()]})
    def test_null_calls(self):
        with self.assertRaisesRegex(ValueError,'PLAN_CALLS_NOT_LIST'):parse_plan_reply({'finish_reason':'stop','tool_calls':None})
    def test_missing_calls(self):
        with self.assertRaisesRegex(ValueError,'PLAN_CALLS_MISSING'):parse_plan_reply({'finish_reason':'tool_calls','tool_calls':[]})
    def test_invalid_wrapper(self):
        with self.assertRaisesRegex(ValueError,'PLAN_WRAPPER_INVALID'):parse_plan_reply({'finish_reason':'tool_calls','tool_calls':[call(arguments='[]')]})
    def test_metadata_no_payload(self):
        m=response_metadata({'finish_reason':'stop','content':'secret-body','tool_calls':[call(arguments='secret-args')]})
        self.assertNotIn('secret',json.dumps(m));self.assertEqual(m['tool_call_count'],1)
    def test_meter_records_before_reject(self):
        class G:
            def complete(self,m,t):return {'finish_reason':'tool_calls','tool_calls':[call('inspect_rr_intervals')]}
        meter=Meter(G())
        with self.assertRaisesRegex(ValueError,'TOOL_NOT_OFFERED_IN_THIS_PHASE'):
            meter.complete([], [{'function':{'name':'submit_plan'}}])
        self.assertEqual(meter.requests[0]['response_protocol']['tools'][0]['name'],'inspect_rr_intervals')
        self.assertEqual(meter.requests[0]['status'],'failed')
    def test_meter_accepts_submit(self):
        class G:
            def complete(self,m,t):return {'finish_reason':'tool_calls','tool_calls':[call()]}
        meter=Meter(G());meter.complete([], [{'function':{'name':'submit_plan'}}])
        self.assertEqual(meter.queries,0)
if __name__=='__main__':unittest.main()
