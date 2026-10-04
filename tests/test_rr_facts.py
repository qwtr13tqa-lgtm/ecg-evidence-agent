import copy
import json
import unittest
from types import SimpleNamespace as S
from src.analysis.rr_facts import build_rr_facts, render_rr_facts
from src.tools.ecg_tools import inspect_rr_intervals

def fixture(peaks=(100,600,1100)):
    raw=[(b-a)/500 for a,b in zip(peaks,peaks[1:])]
    mask=[60/220<=x<=2 for x in raw]
    kept=[x for x,k in zip(raw,mask) if k]
    mean=sum(kept)/len(kept) if kept else None
    r=S(num_beats=len(peaks),r_peaks=list(peaks),mean_rr=mean,heart_rate=60/mean if mean else None,
        rr_details={'raw_rr_seconds':raw,'valid_mask':mask,'retained_rr_seconds':kept,
        'exclusion_reasons':[None if k else 'below_configured_min_rr' if x<60/220 else 'above_configured_max_rr' for x,k in zip(raw,mask)],
        'parameters':{'sampling_rate':500,'min_hr':30,'max_hr':220,'min_rr_seconds':60/220,'max_rr_seconds':2}})
    return S(analysis_id='test',rhythm=r,input=S(sampling_rate=500,num_samples=4800),provenance={})

class TestRRFacts(unittest.TestCase):
    def test_retained(self):
        f=build_rr_facts(fixture()); self.assertEqual(f['status'],'verified')
        self.assertEqual((f['candidate_peak_count'],f['total_intervals'],f['excluded_count']),(3,2,0))
        self.assertIn('没有任何',render_rr_facts(f))
    def test_partial(self):
        f=build_rr_facts(fixture((100,150,650)))
        self.assertEqual((f['retained_count'],f['excluded_count'],f['heart_rate_bpm']),(1,1,60))
    def test_all_excluded(self):
        f=build_rr_facts(fixture((100,150,200))); self.assertIsNone(f['heart_rate_bpm'])
        self.assertIn('全部间隔',render_rr_facts(f))
    def test_zero_single(self):
        for p in ((),(100,)):
            f=build_rr_facts(fixture(p)); self.assertEqual(f['status'],'verified'); self.assertIsNone(f['mean_rr_seconds'])
    def test_missing(self):
        r=fixture(); r.rhythm=None; self.assertEqual(build_rr_facts(r)['status'],'unavailable')
    def test_tampering(self):
        for field,value in [('raw_rr_seconds',[.9,1]),('valid_mask',[False,True]),
                            ('exclusion_reasons',['bad',None]),('retained_rr_seconds',[1]),
                            ('raw_rr_seconds',[float('nan'),1])]:
            r=fixture(); r.rhythm.rr_details[field]=value
            self.assertEqual(build_rr_facts(r)['status'],'invalid')
            with self.assertRaises(ValueError): inspect_rr_intervals(r)
    def test_summary_mismatch(self):
        r=fixture(); r.rhythm.heart_rate=75
        self.assertEqual(build_rr_facts(r)['status'],'invalid')
    def test_peak_mismatch(self):
        r=fixture(); r.rhythm.r_peaks[1]=100
        self.assertEqual(build_rr_facts(r)['status'],'invalid')
    def test_tolerance(self):
        r=fixture((100,593,1086)); r.rhythm.mean_rr=.9860000610351562; r.rhythm.heart_rate=60/r.rhythm.mean_rr
        self.assertEqual(build_rr_facts(r)['status'],'verified')
    def test_pagination(self):
        p=inspect_rr_intervals(fixture(),1,1)
        self.assertEqual(len(p['intervals']),1); self.assertEqual(p['calculation_facts']['total_intervals'],2)
    def test_no_mutation(self):
        r=fixture(); before=copy.deepcopy(r); f=build_rr_facts(r); render_rr_facts(f)
        self.assertEqual(r,before); json.dumps(f,allow_nan=False)
if __name__=='__main__': unittest.main()
