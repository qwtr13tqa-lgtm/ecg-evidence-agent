import importlib.util
from pathlib import Path
import tempfile
import sqlite3
import unittest

spec=importlib.util.spec_from_file_location('eval_v1',Path(__file__).resolve().parents[1]/'evaluation/cross_record_eval_v1.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

class EvalTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(index=i,category=c,score=s,reconstruction_error=r,shape_error=h,region_count=n) for i,c,s,r,h,n in [(0,'fn',-.9,None,3,0),(1,'fn',-.95,1,1,2),(2,'tp',-.7,2,2,1),(3,'tp',-.6,4,4,2),(4,'fp',-.5,4,1,0),(5,'fn',-.95,1,5,0)]]
    def case(self,f):return next(c for c in e.suite() if c['family']==f)
    def out(self,data,terminal=True):return {'status':'completed_draft' if terminal else 'failed','validation':{'passed':terminal},'draft':{'evidence_ids':['x']},'evidence':{'x':{'ok':True,'scope':'batch_collection','data':data}}}
    def test_suite_distinct(self):
        self.assertEqual(len(e.suite()),14);self.assertEqual(len({c['id'] for c in e.suite()}),14)
    def test_stable_tie(self):self.assertEqual(e.expected(self.case('FN_TOP5'),self.rows)['indices'],[1,5,0])
    def test_missing_not_zero(self):
        o=e.expected(self.case('COMPARE'),self.rows)['groups']['fn']['reconstruction_error'];self.assertEqual(o,{'valid_n':2,'missing_n':1,'median':1.0})
    def test_condition_strict(self):
        o=e.expected(self.case('CONDITIONAL'),self.rows);self.assertEqual(o['boundary'],3);self.assertEqual(o['indices'],[5])
    def test_condition_missing(self):
        for r in self.rows:r['shape_error']=None
        self.assertEqual(e.expected(self.case('CONDITIONAL'),self.rows)['kind'],'insufficient')
    def test_failed_not_pass(self):
        o=e.expected(self.case('FP_SORT'),self.rows);d={'category':'fp','rows':[{'index':4,'score':-.5}]};g=e.grade(self.out(d,False),o);self.assertTrue(g['table_contract_pass']);self.assertFalse(g['automatic_acceptance'])
    def test_wrong_order(self):
        o=e.expected(self.case('FN_TOP5'),self.rows);d={'category':'fn','rows':[{'index':0},{'index':1},{'index':5}]};self.assertFalse(e.grade(self.out(d),o)['table_contract_pass'])
    def test_bootstrap_not_answer(self):
        o=e.expected(self.case('FP_SORT'),self.rows);out=self.out({'category':'fp','rows':[{'index':4,'score':-.5}]});out['evidence']={'a:batch-overview:b':out['evidence']['x']};self.assertFalse(e.grade(out,o)['table_contract_pass'])
    def test_wrong_scope_empty(self):
        o=e.expected(self.case('MISSING'),[]);self.assertFalse(e.grade(self.out({'category':'tp','rows':[]}),o)['table_contract_pass'])
    def test_missing_median_key_not_null(self):
        o={'kind':'aggregate','groups':{'fn':{'shape_error':{'median':None}}}};self.assertFalse(e.grade(self.out({'groups':[{'group_value':'fn','metrics':{}}]}),o)['table_contract_pass'])
    def test_query_cost_unknown(self):
        self.assertEqual(e.costs({'trace':[{'stage':'tool','ok':True}]})['execution_flag_unknown'],1)
    def test_read_history_readonly(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'h.db'
            with sqlite3.connect(p) as db:db.execute('create table t(x)')
            with e.ReadHistory(p).connect() as db:
                with self.assertRaises(sqlite3.OperationalError):db.execute('insert into t values (1)')
    def test_top5_allows_remaining_rows(self):
        o=e.expected(self.case('FN_TOP5'),self.rows)
        data={'category':'fn','rows':[{'index':i,'score':v} for i,v in zip(o['indices'],o['values'])],'has_more':True}
        self.assertTrue(e.grade(self.out(data),o)['table_contract_pass'])
    def test_all_records_rejects_truncation(self):
        o=e.expected(self.case('FP_SORT'),self.rows)
        self.assertFalse(e.grade(self.out({'category':'fp','rows':[{'index':4,'score':-.5}],'has_more':True}),o)['table_contract_pass'])
    def test_cited_startup_statistics_can_be_reused(self):
        o={'kind':'aggregate','groups':{'fn':{'shape_error':{'median':3}}}}
        out=self.out({});eid='a:batch-overview:b';out['draft']['evidence_ids']=[eid];out['evidence']={eid:{'ok':True,'scope':'explicit_batch_report','data':{'groups':[{'category':'fn','field':'shape_error','median':3}]}}}
        self.assertTrue(e.grade(out,o)['table_contract_pass'])
    def test_counts(self):self.assertEqual(e.expected(self.case('COUNTS'),self.rows)['counts'],{'fn':3,'fp':1,'tp':2})
    def test_clarification(self):self.assertTrue(e.grade(self.out({'query_status':'needs_clarification'}),{'kind':'clarify'})['automatic_acceptance'])
if __name__=='__main__':unittest.main()
