import copy
import json
import tempfile
import unittest
from pathlib import Path
from src.evaluation.fine_metrics import analyze,aggregate,calls_for,save_review,read_review


def fixture():
    return {'run_id':'r','scheme':'agent','case':'window','turn':1,'model_calls':2,'latency':3,
        'record':{'run_id':'r','config':{'scheme':'agent','protocol':'threeway-1.0'},'case':{'expectation':{'kind':'window'}},
        'automatic':{'field_checks':{'a':True,'b':False}},'output':{'evidence':{'s':{'data':{'input':{}}},'k':{'data':{'status':'no_match'}}},'trace':[
            {'stage':'tool','tool':'get_analysis_summary','arguments':{},'ok':True,'evidence_id':'s'},
            {'stage':'tool','tool':'get_model_decision','arguments':{},'ok':True},
            {'stage':'tool','tool':'search_knowledge','arguments':{},'ok':True,'evidence_id':'k'}]}}}


class FineTests(unittest.TestCase):
    def test_partial_coverage(self):self.assertEqual(analyze(fixture())['field_coverage'],.5)
    def test_missing_fields_not_zero_or_pass(self):
        r=fixture();r['record']['automatic']['field_checks']={}
        self.assertIsNone(analyze(r)['field_coverage'])
    def test_bootstrap_duplicate_included(self):
        x=analyze(fixture());self.assertEqual(x['redundant_calls'],1);self.assertEqual(x['pending_calls'],2)
    def test_empty_search_not_automatically_redundant(self):
        self.assertEqual(calls_for(fixture()['record'])[2]['suggestion'],'pending')
    def test_initial_bootstrap_excluded(self):
        r=fixture();r['record']['output']['trace'][0]['source']='bootstrap'
        self.assertEqual(analyze(r)['query_requests'],2)
    def test_failed_not_double_counted(self):
        r=fixture();r['record']['output']['trace'][0]['ok']=False
        x=analyze(r);self.assertEqual(x['failed_calls'],1);self.assertEqual(x['redundant_calls'],0)
    def test_repeated_arguments(self):
        r=fixture();r['record']['output']['trace'].append(copy.deepcopy(r['record']['output']['trace'][1]))
        self.assertEqual(analyze(r)['redundant_calls'],2)
    def test_manual_override_with_provenance(self):
        x=analyze(fixture(),{'calls':{'1':{'verdict':'redundant','note':'not requested'}}})
        self.assertEqual(x['redundant_calls'],2);self.assertEqual(x['calls'][1]['origin'],'human')
    def test_no_claims_missing(self):self.assertIsNone(analyze(fixture())['claim_support_rate'])
    def test_partial_claims_not_full_accuracy(self):
        x=analyze(fixture(),{'claims':[{'verdict':'supported'},{'verdict':'pending'}]})
        self.assertEqual(x['claim_support_rate'],1);self.assertFalse(x['claims_complete']);self.assertEqual(x['pending_claims'],1)
    def test_aggregate_micro_coverage(self):
        a=analyze(fixture());b=copy.deepcopy(a);b.update(field_total=4,field_correct=4)
        self.assertEqual(aggregate([a,b])[0]['field_coverage'],5/6)
    def test_review_hash_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'result.json';p.write_text(json.dumps(fixture()['record']));before=p.read_bytes()
            save_review(p,'tester',{'1':{'verdict':'redundant','note':'only window requested'}},[],False)
            self.assertEqual(read_review(p)['reviewer'],'tester');self.assertEqual(p.read_bytes(),before)
            p.write_bytes(before+b' ')
            with self.assertRaises(ValueError):read_review(p)
    def test_invalid_trace_index_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'result.json';p.write_text(json.dumps(fixture()['record']))
            with self.assertRaises(ValueError):save_review(p,'t',{'99':{'verdict':'redundant','note':'x'}},[],False)

if __name__=='__main__':unittest.main()
