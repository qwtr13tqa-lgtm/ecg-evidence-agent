import copy
import unittest
from unittest.mock import patch
from src.evaluation.checks_v2 import score_output

def fixture():
    rr={'candidate_beat_count':10,'mean_rr_seconds':.9860000610351562,'heart_rate_bpm':60.85192321084519,
        'total_intervals':9,'retained_count':9,'excluded_count':0}
    facts={'status':'verified','analysis_id':'A','scope':'stored_rr_arithmetic_only','candidate_peak_count':10,
           'mean_rr_seconds':.9860000000000001,'heart_rate_bpm':60.85192697768762}
    data={k:rr[k] for k in ('total_intervals','retained_count','excluded_count')};data['calculation_facts']=facts
    obs=[{'evidence_id':'E','path':'/'+k,'value':data[k]} for k in data if k!='calculation_facts']
    obs += [{'evidence_id':'E','path':'/calculation_facts/'+k,'value':facts[k]} for k in ('candidate_peak_count','mean_rr_seconds','heart_rate_bpm')]
    out={'evidence':{'E':{'analysis_id':'A','ok':True,'data':data}},'draft':{'observations':obs,'evidence_ids':['E']},
         'trace':[{'stage':'tool','tool':'inspect_rr_intervals','ok':True,'evidence_id':'E'}]}
    return out,{'analysis_id':'A','rr':rr}
class Tests(unittest.TestCase):
    def score(self,o,r,valid=True):
        with patch('src.evaluation.checks_v2.score_v1',return_value={'metrics':{'structure_values_references':valid,'analysis_binding':True}}):
            return score_output({'expectation':{'kind':'rr'}},o,r)['metrics']['task_observations']
    def test_alias(self):
        o,r=fixture();self.assertTrue(self.score(o,r))
    def test_wrong_value(self):
        o,r=fixture();o['evidence']['E']['data']['calculation_facts']['mean_rr_seconds']=.8
        self.assertFalse(self.score(o,r))
    def test_not_exact(self):
        o,r=fixture();o['draft']['observations'][4]['value']=.986
        self.assertFalse(self.score(o,r))
    def test_uncited(self):
        o,r=fixture();o['draft']['evidence_ids']=[];self.assertFalse(self.score(o,r))
    def test_wrong_binding(self):
        o,r=fixture();o['evidence']['E']['data']['calculation_facts']['analysis_id']='B';self.assertFalse(self.score(o,r))
    def test_invalid_structure(self):
        o,r=fixture();self.assertFalse(self.score(o,r,False))
    def test_missing_field(self):
        o,r=fixture();o['draft']['observations'].pop();self.assertFalse(self.score(o,r))
    def test_no_mutation(self):
        o,r=fixture();before=copy.deepcopy(o);self.score(o,r);self.assertEqual(o,before)
if __name__=='__main__':unittest.main()
