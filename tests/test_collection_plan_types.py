import copy
import unittest
from src.review.collection_ops import normalize_operation,validate_operation,CollectionValidationError,FILTER_SCHEMA,NUMERIC,TEXT
from src.review.collection_runtime import checked_plan
from src.review.cross_record_agent import run
from test_batch_collection import select,plan,Gateway,report

class PlanTypeTests(unittest.TestCase):
    def test_quoted_numbers_normalized_locally(self):
        op=select(filters=[{'field':'label','op':'eq','value':'1'},{'field':'prediction','op':'eq','value':'0'}],limit=2)
        out=run(None,Gateway(plan(operations=[op])),'aid','漏报最低2条',report(),'batch')
        self.assertEqual(out['status'],'completed_draft',out)
        self.assertEqual(out['model_calls'],1);self.assertEqual(out['tool_calls'],0)
        d=next(iter(out['evidence'].values()))['data'];self.assertEqual([r['index'] for r in d['rows']],[6,7])
        self.assertTrue(any(t['stage']=='plan_normalization' for t in out['trace']))
    def test_original_plan_and_normalized_plan_both_saved(self):
        trace=[];p=checked_plan(plan(operations=[select(filters=[{'field':'score','op':'lt','value':'-0.5'}])]),trace)
        self.assertEqual(trace[0]['plan']['operations'][0]['filters'][0]['value'],'-0.5')
        self.assertEqual(p['operations'][0]['filters'][0]['value'],-.5)
    def test_in_list(self):
        op,changes=normalize_operation(select(filters=[{'field':'index','op':'in','value':['5','6']}]))
        validate_operation(op);self.assertEqual(op['filters'][0]['value'],[5,6]);self.assertEqual(len(changes),2)
    def test_bad_values_not_guessed(self):
        for v in (None,True,'最低','正常','','nan','inf','1 bpm','1,000','1e999','1e-999'):
            with self.subTest(value=v):
                op,_=normalize_operation(select(filters=[{'field':'score','op':'eq','value':v}]))
                with self.assertRaises(CollectionValidationError):validate_operation(op)
    def test_failure_has_operation_field_and_path(self):
        op=select(filters=[{'field':'score','op':'lt','value':None}])
        out=run(None,Gateway(plan(operations=[op])),'aid','问题',report(),'batch')
        self.assertEqual(out['error'],'FINITE_NUMERIC_REQUIRED');self.assertEqual(out['local_query_calls'],0)
        t=out['trace'][-1];self.assertEqual(t['requested_path'],'/operations/0/filters/0/value');self.assertEqual(t['field'],'score');self.assertEqual(t['received_type'],'NoneType')
        self.assertTrue(any(t['stage']=='collection_plan_received' for t in out['trace']))
    def test_no_mutation(self):
        op=select(filters=[{'field':'score','op':'gt','value':'0.1'}]);old=copy.deepcopy(op);normalize_operation(op);self.assertEqual(op,old)
    def test_unknown_extra_field_not_removed(self):
        op=select();op['invented']='drop me'
        with self.assertRaises(ValueError):checked_plan(plan(operations=[op]))
    def test_null_stays_null_for_missing_operator(self):
        op=select(filters=[{'field':'region_count','op':'is_missing','value':None}]);new,changes=normalize_operation(op);validate_operation(new);self.assertEqual(changes,[])
    def test_schema_couples_numeric_and_text_values(self):
        branches=FILTER_SCHEMA['anyOf']
        self.assertEqual(branches[0]['properties']['value']['type'],'number')
        self.assertEqual(set(branches[0]['properties']['field']['enum']),set(NUMERIC))
        self.assertEqual(set(branches[2]['properties']['field']['enum']),set(TEXT))
        self.assertEqual(branches[-1]['properties']['value']['type'],'null')
    def test_later_bad_step_prevents_earlier_execution(self):
        ops=[select(),select(filters=[{'field':'score','op':'lt','value':'lowest'}])]
        out=run(None,Gateway(plan(operations=ops)),'aid','问题',report(),'batch')
        self.assertEqual(out['evidence'],{});self.assertEqual(out['local_query_calls'],0)

if __name__=='__main__':unittest.main()
