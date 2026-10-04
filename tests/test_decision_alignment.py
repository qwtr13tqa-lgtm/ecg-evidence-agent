import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
from src.analysis.model_decision import evaluate_decision, freeze_decision, get_model_decision
from src.tools.window_alignment import inspect_recent_rr_alignment, inspect_window_rr_alignment
from src.agent.failure_help import failure_help, DiagnosticGateway
from src.tools.executor import ECGToolExecutor
from src.analysis.store import AnalysisStore
from evaluation.select_anomaly_threshold import select_threshold
from tests.test_rr_facts import fixture


def signal_fixture():
    r=fixture((332,791,1268,1759,2246,2751,3261,3775,4277,4769))
    r.rhythm.lead_index=1;r.input.num_leads=12
    r.model=S(anomaly_score=-.9,error_map=np.arange(4800*12,dtype=float).reshape(4800,12))
    r.provenance={'crop_start_sample':100,'checkpoint_sha256':'hash'}
    return r


class TestDecisionAlignment(unittest.TestCase):
    def config(self):
        records={'split':'validation','split_id':'v1','profile':{'version':'a'},
            'dataset_sha256':'data','labels_sha256':'labels','rows':[
                {'score':-1.0,'label':0,'sample_index':0},{'score':-.8,'label':1,'sample_index':1}]}
        return select_threshold(records)

    def test_unconfigured(self):
        d=evaluate_decision(-.9,None,None);self.assertIsNone(d['prediction'])
        self.assertEqual(d['status'],'unconfigured')
    def test_threshold_equal_and_negative(self):
        c=self.config();p=c['profile'];t=c['threshold']
        self.assertEqual(evaluate_decision(t,p,c)['prediction'],'model_anomaly')
        self.assertEqual(evaluate_decision(-1.0,p,c)['prediction'],'model_normal')
        self.assertIsNone(evaluate_decision(t,p,c)['probability'])
    def test_mismatch(self):
        c=self.config();self.assertEqual(evaluate_decision(0,{'version':'b'},c)['status'],'invalid')
    def test_normalized_threshold_rejected(self):
        c=self.config();c['score_space']='minmax'
        self.assertEqual(evaluate_decision(0,c['profile'],c)['status'],'invalid')
    def test_nonfinite(self):
        c=self.config();c['threshold']=None
        self.assertEqual(evaluate_decision(0,c['profile'],c)['status'],'invalid')
        self.assertEqual(evaluate_decision(float('nan'),None,None)['status'],'invalid')
    def test_test_split_rejected(self):
        c=self.config();c['selection']['split']='test'
        self.assertEqual(evaluate_decision(0,c['profile'],c)['status'],'invalid')
    def test_frozen_configuration(self):
        r=signal_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'threshold.json';p.write_text('{"enabled":false}')
            freeze_decision(r,S(mask_ratio_time=30,mask_ratio_spec=20),p)
            p.write_text('broken')
            self.assertEqual(get_model_decision(r)['status'],'unconfigured')
            freeze_decision(r,S(mask_ratio_time=30,mask_ratio_spec=20),p)
            self.assertEqual(get_model_decision(r)['status'],'invalid')
    def test_last_point_six_overlap(self):
        a=inspect_recent_rr_alignment(signal_fixture(),.6,'V1')
        self.assertEqual(a['window']['start_sample'],4500)
        self.assertEqual(a['overlapping_rr_count'],1)
        row=a['rr_intervals'][0]
        self.assertEqual(row['rr_index'],8)
        self.assertAlmostEqual(row['rr_seconds'],.984)
        self.assertAlmostEqual(row['overlap_seconds'],.538)
        self.assertFalse(row['fully_contained']);self.assertFalse(a['same_lead'])
        self.assertEqual(row['original_left_peak_sample'],4377)
    def test_boundary_no_overlap(self):
        a=inspect_window_rr_alignment(signal_fixture(),4769,4800,'II')
        self.assertEqual(a['overlapping_rr_count'],0)
    def test_missing_rr(self):
        r=signal_fixture();r.rhythm=None
        self.assertEqual(inspect_recent_rr_alignment(r,.6)['rr_status'],'unavailable')
    def test_tampered_rr(self):
        r=signal_fixture();r.rhythm.heart_rate=999
        self.assertEqual(inspect_recent_rr_alignment(r,.6)['rr_status'],'invalid')
    def test_invalid_window(self):
        with self.assertRaises(ValueError):inspect_recent_rr_alignment(signal_fixture(),99)
        with self.assertRaises(ValueError):inspect_window_rr_alignment(signal_fixture(),-1,2)
    def test_bound_executor(self):
        r=signal_fixture();s=AnalysisStore();aid=s.begin();r.analysis_id=aid;s.complete(aid,r)
        e=ECGToolExecutor(s,aid)
        self.assertTrue(e.execute('get_model_decision',{})['ok'])
        self.assertTrue(e.execute('inspect_recent_rr_alignment',{'duration_seconds':.6})['ok'])
        self.assertFalse(e.execute('get_model_decision',{'analysis_id':'foreign'})['ok'])
    def test_unknown_failure_not_invented(self):
        self.assertEqual(failure_help({'status':'failed','trace':[]})['error_type'],'unknown')
    def test_timeout_explained(self):
        self.assertIn('等待时间',failure_help({'trace':[{'error_type':'APITimeoutError'}]})['message'])
    def test_safe_exception_metadata(self):
        class G:
            def complete(self,m,t):raise ValueError('secret-token')
        g=DiagnosticGateway(G())
        with self.assertRaises(ValueError):g.complete([],[])
        self.assertNotIn('secret',json.dumps(g.last_error))

if __name__=='__main__':unittest.main()
