import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
from src.review.batch_statistics import collect
from src.ui.batch_statistics_ui import navigate

class History:
    def __init__(self):
        self.contexts={0:{'model':{'shape_error':2.,'reconstruction_error':0.},'evidence':{'temporal_regions':[],
            'lead_evidence':[{'lead':'V2','rank':1}]},'signal_features':{'rhythm':{'heart_rate_bpm':60.,'rr_cv':float('nan'),'measurement_status':'unvalidated'}}},
            1:{'model':{'shape_error':4.},'evidence':{'temporal_regions':[{'start':1,'end':3}]}}}
    def list_analyses(self,limit,offset):return [{'sample_index':i,'analysis_id':str(i)} for i in self.contexts][offset:offset+limit]
    def load_analysis(self,aid):return NS(analysis_id=aid,to_llm_context=lambda:self.contexts[int(aid)]),None,None

REPORT={'threshold':.5,'rows':[{'index':i,'score':0. if i in (0,2) else 1.,'label':1,'prediction':int(i==1),'input_sha256':str(i)*64} for i in range(3)]}
class Tests(unittest.TestCase):
    def setUp(self):self.guard=patch('src.review.batch_statistics.compatible',return_value=True);self.guard.start()
    def tearDown(self):self.guard.stop()
    def test_missing_and_zero_distinct(self):
        r=collect(REPORT,History(),['fn','tp']);self.assertEqual(r['rows'][0]['region_count'],0);self.assertIsNone(r['rows'][2]['region_count'])
        g=next(g for g in r['groups'] if g['category']=='fn' and g['field']=='region_count')
        self.assertEqual((g['valid_n'],g['missing_n'],g['median']),(1,1,0))
    def test_report_scores_include_missing_history(self):
        r=collect(REPORT,History(),['fn']);g=next(g for g in r['groups'] if g['field']=='score')
        self.assertEqual(g['valid_n'],2);self.assertEqual(g['matched_n'],1)
    def test_nonfinite_missing(self):
        r=collect(REPORT,History(),['fn']);self.assertIsNone(r['rows'][0]['rr_cv'])
    def test_scope(self):self.assertEqual([r['index'] for r in collect(REPORT,History(),['tp'])['rows']],[1])
    def test_wrong_version_excluded(self):
        with patch('src.review.batch_statistics.compatible',return_value=False):
            r=collect(REPORT,History(),['tp']);self.assertEqual(r['rows'][0]['history_status'],'no_compatible_history');self.assertIsNone(r['rows'][0]['shape_error'])
    def test_group_quartiles(self):
        r=collect(REPORT,History(),['tp']);g=next(g for g in r['groups'] if g['field']=='shape_error');self.assertEqual((g['q1'],g['median'],g['q3']),(4.,4.,4.))
    def test_navigation_callback(self):
        st=NS(session_state={});seen=[];navigate(st,seen.append,'aid');self.assertEqual(seen,['aid']);self.assertEqual(st.session_state['workspace_entry'],'历史记录')
    def test_navigation_failure_does_not_switch(self):
        st=NS(session_state={})
        def fail(aid):raise ValueError('deleted')
        with self.assertRaises(ValueError):navigate(st,fail,'aid')
        self.assertNotIn('workspace_entry',st.session_state)
if __name__=='__main__':unittest.main()
