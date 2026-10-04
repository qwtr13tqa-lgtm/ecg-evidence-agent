import unittest
from copy import deepcopy
from src.review.plan_bindings import validate_bound_operation,resolve_operation,guard_plan,ambiguity_reason
from src.review.collection_runtime import run_collection,checked_plan
from src.review.collection_ops import execute,enrich
from evaluation.cross_record_boundary_v2 import fixture,SyntheticHistory
from evaluation.cross_record_eval_v2 import expected,suite,grade
from evaluation.cross_record_scoring_v2 import normalized_output,consistency
import json

def op(kind='aggregate',cat='tp'):
    return {'kind':kind,'filters':[{'field':'category','op':'eq','value':cat}], 'order_by':None if kind=='aggregate' else 'shape_error','direction':'desc','limit':500,'offset':0,'columns':[] if kind=='aggregate' else ['index','shape_error'],'group_by':None,'metrics':['shape_error'] if kind=='aggregate' else []}
def plan():
    first=op();second=op('select','fn');second['filters'].append({'field':'shape_error','op':'gt','value':{'from_step':0,'path':'/groups/0/metrics/shape_error/median'}})
    return {'mode':'query','operations':[first,second],'assumptions':[],'clarification':''}
def reply(p):return {'finish_reason':'tool_calls','tool_calls':[{'id':'x','type':'function','function':{'name':'plan_batch_task','arguments':json.dumps(p)}}]}
class Gateway:
    def __init__(self,plans):self.plans=plans;self.n=0
    def complete(self,*a,**kw):p=self.plans[min(self.n,len(self.plans)-1)];self.n+=1;return reply(p)

class V2Tests(unittest.TestCase):
    def setUp(self):self.data=fixture();self.history=SyntheticHistory(self.data);self.rows=enrich(self.data['report'],self.history)[0]
    def run_plan(self,plans,no_tp=False,q='先计算TP形状项中位数，再筛选高于该值的FN'):
        g=Gateway(plans);out=run_collection(SyntheticHistory(self.data,no_tp),g,'test',q,self.data['report'],'hash',None,max_model_calls=3);return out,g
    def test_backward_reference_accepted(self):self.assertEqual(checked_plan(reply(plan())),plan())
    def test_forward_reference_rejected(self):
        p=plan();p['operations'][1]['filters'][1]['value']['from_step']=1
        with self.assertRaisesRegex(ValueError,'DEPENDENCY_NOT_PREVIOUS_STEP'):checked_plan(reply(p))
    def test_wrong_metric_rejected(self):
        p=plan();p['operations'][1]['filters'][1]['field']='score'
        with self.assertRaisesRegex(ValueError,'METRIC_MISMATCH'):checked_plan(reply(p))
    def test_no_arbitrary_pointer(self):
        p=plan();p['operations'][1]['filters'][1]['value']['path']='/rows/0/score'
        with self.assertRaisesRegex(ValueError,'PATH_NOT_AGGREGATE'):checked_plan(reply(p))
    def test_zero_placeholder_rejected(self):
        p=plan();p['operations'][1]['filters'][1]['value']=0
        with self.assertRaisesRegex(ValueError,'BINDING_REQUIRED'):guard_plan('高于TP中位数的FN',p)
    def test_actual_zero_statistic_valid(self):
        p=plan();r,b=resolve_operation(p['operations'][1],[{'evidence_id':'e','data':{'groups':[{'metrics':{'shape_error':{'median':0}}}]}}]);self.assertEqual(r['filters'][1]['value'],0);self.assertEqual(b[0]['source_evidence_id'],'e')
    def test_positive_conditional_runtime(self):
        out,g=self.run_plan([plan()]);self.assertEqual(out['status'],'completed_draft');self.assertEqual(g.n,1)
        ev=list(out['evidence'].values());self.assertEqual([r['index'] for r in ev[-1]['data']['rows']],[0]);self.assertEqual(ev[-1]['data']['operation']['filters'][1]['value'],.4);self.assertEqual(ev[-1]['data']['resolved_bindings'][0]['source_evidence_id'],ev[0]['evidence_id'])
    def test_missing_upstream_not_zero(self):
        out,g=self.run_plan([plan()],True);self.assertEqual(out['query_status'],'insufficient_evidence');self.assertEqual(out['local_query_calls'],1);self.assertEqual(out['status'],'completed_draft')
    def test_binding_repair_once(self):
        bad=plan();bad['operations'][1]['filters'][1]['value']=0
        out,g=self.run_plan([bad,plan()]);self.assertEqual(out['status'],'completed_draft');self.assertEqual(g.n,2)
    def test_invalid_repair_no_execution(self):
        bad=plan();bad['operations'][1]['filters'][1]['value']=0
        out,g=self.run_plan([bad,bad]);self.assertEqual(out['status'],'failed');self.assertEqual(out['local_query_calls'],0);self.assertEqual(g.n,2)
    def test_ambiguous_no_default_severity(self):
        out,g=self.run_plan([plan()],q='找出漏报中最值得关注的几条');self.assertEqual(g.n,0);self.assertTrue(any(t['stage']=='clarification' for t in out['trace']))
    def test_explicit_priority_allowed(self):self.assertIsNone(ambiguity_reason('按分数最低选择最值得关注的5条'))
    def test_equivalent_labels_aggregate(self):
        o=op();o['filters']=[{'field':'label','op':'eq','value':1},{'field':'prediction','op':'eq','value':1}]
        out={'evidence':{'e':{'data':execute(self.rows,o)}}};n=normalized_output(out);self.assertEqual(n['evidence']['e']['data']['groups'][0]['group_value'],'tp');self.assertEqual(out['evidence']['e']['data']['groups'][0]['group_value'],'all')
    def test_restricted_aggregate_not_whole_group(self):
        o=op();o['filters'].append({'field':'shape_error','op':'gt','value':.5});out={'evidence':{'e':{'data':execute(self.rows,o)}}};self.assertEqual(normalized_output(out)['evidence']['e']['data']['groups'][0]['group_value'],'all')
    def test_conflicting_category_not_inferred(self):
        o=op();o['filters'] += [{'field':'label','op':'eq','value':0}];out={'evidence':{'e':{'data':execute(self.rows,o)}}};self.assertEqual(normalized_output(out)['evidence']['e']['data']['groups'][0]['group_value'],'all')
    def test_critical_contradiction_no(self):self.assertEqual(consistency({'task_complete':'yes','claims':[{'task_critical':True,'verdict':'contradicted'}]})['effective_task_complete'],'no')
    def test_legacy_contradiction_uncertain(self):self.assertEqual(consistency({'task_complete':'yes','claims':[{'verdict':'contradicted'}]})['effective_task_complete'],'uncertain')
    def test_critical_missing_uncertain(self):self.assertEqual(consistency({'task_complete':'yes','claims':[{'task_critical':True,'verdict':'insufficient'}]})['effective_task_complete'],'uncertain')
    def test_boundary_nonempty_missing_and_ties(self):
        cases={c['family']:c for c in suite()};self.assertEqual(expected(cases['CONDITIONAL'],self.rows)['indices'],[0]);self.assertEqual(expected(cases['MISSING'],self.rows)['indices'],[1,2]);self.assertEqual(expected(cases['FN_TOP5'],self.rows)['indices'],[0,1,2,7,8])
    def test_conditional_grade_true(self):
        out,_=self.run_plan([plan()]);c=next(c for c in suite() if c['family']=='CONDITIONAL');self.assertTrue(grade(out,expected(c,self.rows))['automatic_acceptance'])
if __name__=='__main__':unittest.main()
