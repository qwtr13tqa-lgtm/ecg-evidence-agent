import unittest
from src.evaluation.plan_validation_v12 import validate_plan,PlanValidationError
from src.evaluation.complex_plan_v1 import execute_plan
class Executor:
    def __init__(self):self.calls=[]
    def execute(self,name,args):
        self.calls.append((name,args));return {'ok':True,'evidence_id':'x','data':{'n':2}}
class ValidationTests(unittest.TestCase):
    def test_collect_errors_before_queries(self):
        p={'steps':[{'id':'a','tool':'get_analysis_summary','args':{}},{'id':'b','value':{'ref':'a'}},{'id':'c','value':{'op':'get','args':[1],'args2':None}},{'id':'d','value':{'tool':'get_analysis_summary','args':{}}}]}
        e=Executor();errors=validate_plan(p)
        self.assertTrue({'REFERENCE_SCHEMA','OP_ARITY','OP_SCHEMA','NESTED_TOOL_NOT_ALLOWED'}<={r['code'] for r in errors})
        with self.assertRaises(PlanValidationError):execute_plan(p,e,{}, {},[])
        self.assertEqual(e.calls,[])
    def test_when_false_skips_bad_data_path(self):
        p={'steps':[{'id':'a','tool':'inspect_error_window','args':{'start_sample':{'ref':'summary','path':'/missing'},'end_sample':10},'when':False}]}
        e=Executor();trace=[];env=execute_plan(p,e,{}, {},trace)
        self.assertIsNone(env['a']);self.assertEqual(e.calls,[]);self.assertEqual(trace[0]['stage'],'plan_skip')
    def test_when_true(self):
        e=Executor();execute_plan({'steps':[{'id':'a','tool':'get_analysis_summary','args':{},'when':True}]},e,{}, {},[])
        self.assertEqual(len(e.calls),1)
    def test_runtime_boolean_required(self):
        p={'steps':[{'id':'a','tool':'get_analysis_summary','args':{},'when':{'ref':'summary','path':'/n'}}]}
        with self.assertRaisesRegex(ValueError,'WHEN_MUST_BE_BOOLEAN'):execute_plan(p,Executor(),{'n':1},{},[])
    def test_forward_and_unknown_arg(self):
        p={'steps':[{'id':'a','tool':'get_analysis_summary','args':{'analysis_id':'other'},'when':{'ref':'future','path':''}}]}
        self.assertEqual({r['code'] for r in validate_plan(p)},{'UNEXPECTED_ARGUMENT','UNKNOWN_OR_FORWARD_REFERENCE'})
    def test_unselected_bad_syntax_still_rejected(self):
        p={'steps':[{'id':'a','value':{'op':'if','args':[True,1,{'ref':'summary'}]}}]}
        self.assertTrue(validate_plan(p))
    def test_guard_skipped_result(self):
        p={'steps':[{'id':'a','tool':'get_analysis_summary','args':{},'when':False},{'id':'b','value':{'op':'if','args':[False,{'ref':'a','path':'/n'},'missing']}}]}
        self.assertEqual(execute_plan(p,Executor(),{}, {},[])['b'],'missing')
if __name__=='__main__':unittest.main()
