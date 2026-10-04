import copy
import unittest
from test_batch_collection import (select,aggregate,plan,Gateway,History,report,submit_from_evidence)
from src.review.collection_runtime import checked_plan,PlanConflictError
from src.review.cross_record_agent import run

TEXT='比较FN与TP在重构项、形状项和区域数量上的分布，并列出支持记录与反例。'
ASSUMPTIONS=['FN共16条、TP共34条，分别按category过滤统计','支持记录与反例通过按重构项降序取全部FN与TP记录进行对照','所有统计为描述性，不证明漏检原因']
def four_operations():
    return [aggregate(filters=[{'field':'category','op':'eq','value':c}],limit=500,metrics=['reconstruction_error','shape_error','region_count']) for c in ('fn','tp')]+[
        select(filters=[{'field':'category','op':'eq','value':c}],order_by='reconstruction_error',direction='desc',limit=n,columns=['index','category','reconstruction_error','shape_error','region_count']) for c,n in [('fn',16),('tp',34)]]
def original():return plan('analyze',four_operations(),TEXT,ASSUMPTIONS)
def corrected():return plan('analyze',four_operations(),'',ASSUMPTIONS)
class ConflictTests(unittest.TestCase):
    def test_exact_four_operations_are_valid_when_field_corrected(self):
        p=checked_plan(corrected());self.assertEqual(len(p['operations']),4)
    def test_specific_error_for_user_plan(self):
        with self.assertRaises(PlanConflictError) as ctx:checked_plan(original())
        self.assertEqual(ctx.exception.code,'PLAN_CLARIFICATION_CONFLICT');self.assertEqual(ctx.exception.details['requested_path'],'/clarification')
    def test_repair_then_four_operations_and_answer(self):
        gw=Gateway(original(),corrected(),submit_from_evidence)
        out=run(History(),gw,'aid','比较FN与TP，支持与反例',report(),'batch',max_model_calls=3)
        self.assertEqual(out['status'],'completed_draft',out);self.assertEqual(out['model_calls'],3);self.assertEqual(out['local_query_calls'],4)
        self.assertEqual(sum(t['stage']=='plan_repair' for t in out['trace']),1)
        self.assertTrue(any(t.get('stage')=='model' and t.get('call')==3 for t in out['trace']))
    def test_whitespace_no_repair(self):
        gw=Gateway(plan('query',[select()], ' \n\t'))
        out=run(None,gw,'aid','查询',report(),'batch');self.assertEqual(out['status'],'completed_draft');self.assertEqual(len(gw.calls),1)
    def test_cannot_change_filters_during_repair(self):
        changed=four_operations();changed[2]['limit']=1
        gw=Gateway(original(),plan('analyze',changed,'',ASSUMPTIONS));h=History()
        out=run(h,gw,'aid','比较',report(),'batch');self.assertEqual(out['error'],'PLAN_REPAIR_CHANGED_TASK');self.assertEqual(h.loads,[]);self.assertEqual(out['evidence'],{})
    def test_real_ambiguity_can_be_returned_to_user(self):
        gw=Gateway(original(),plan('clarify',[],'需要按哪项指标选择优先复核记录？'))
        out=run(None,gw,'aid','比较',report(),'batch');self.assertEqual(out['status'],'completed_draft');self.assertEqual(out['local_query_calls'],0)
        self.assertTrue(any(t['stage']=='clarification' for t in out['trace']))
    def test_repair_only_once(self):
        gw=Gateway(original(),original());out=run(None,gw,'aid','比较',report(),'batch')
        self.assertEqual(out['error'],'PLAN_CLARIFICATION_CONFLICT');self.assertEqual(len(gw.calls),2);self.assertEqual(out['evidence'],{})
    def test_reserve_final_answer_request(self):
        gw=Gateway(original());out=run(None,gw,'aid','比较',report(),'batch',max_model_calls=2)
        self.assertEqual(out['error'],'PLAN_CLARIFICATION_CONFLICT');self.assertEqual(len(gw.calls),1)
    def test_repair_timeout_has_no_execution(self):
        gw=Gateway(original(),TimeoutError());out=run(None,gw,'aid','比较',report(),'batch')
        self.assertEqual(out['error'],'GATEWAY_REQUEST_FAILED');self.assertEqual(out['model_calls'],2);self.assertEqual(out['evidence'],{})
    def test_empty_query_has_distinct_error(self):
        with self.assertRaises(PlanConflictError) as ctx:checked_plan(plan('query',[]))
        self.assertEqual(ctx.exception.code,'PLAN_QUERY_OPERATIONS_REQUIRED')
if __name__=='__main__':unittest.main()
