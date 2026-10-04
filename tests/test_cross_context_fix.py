import json
import unittest
from copy import deepcopy
from types import SimpleNamespace as NS
from src.review.cross_context import model_evidence,recent_context
from src.ui.cross_navigation import open_source,return_origin,render_sources

class Tests(unittest.TestCase):
    def fixture(self):
        return {'scope':'explicit_batch_report','evidence_id':'root:e','analysis_id':'root','data':{'cohort_counts':{'fn':16},
            'rows':[{'index':i,'category':'fn','history_status':'matched','top_lead':'V2','source_analysis_id':str(i),'evidence':{'long':'x'*1000}} for i in range(100)]}}
    def test_projection_small_and_no_mutation(self):
        e=self.fixture();before=deepcopy(e);p=model_evidence(e)
        self.assertEqual(e,before);self.assertLess(len(json.dumps(p)),len(json.dumps(e))/4)
        self.assertEqual(p['data']['cohort_counts'],e['data']['cohort_counts']);self.assertEqual(len(p['data']['rows']),100)
    def test_remaining_row_paths_resolve_to_original(self):
        e=self.fixture();p=model_evidence(e)
        for i,r in enumerate(p['data']['rows']):
            for k,v in r.items():self.assertEqual(v,e['data']['rows'][i][k])
    def test_tool_evidence_values_preserved(self):
        e={'scope':'batch_record_query','data':{'sample_index':2,'leads':[{'mean':.23}],'source_analysis_id':'a'}}
        self.assertEqual(model_evidence(e)['data'],e['data'])
    def test_recent_prose_bounded(self):
        r=recent_context([{'question':'a'*4000,'unverified_previous_answer':'b'*10000} for _ in range(8)])
        self.assertEqual(len(r),3);self.assertLess(len(json.dumps(r)),8000)
    def test_return_to_original_conversation(self):
        st=NS(session_state={'workspace_entry':'历史记录'},query_params={'analysis':'original','conversation':'chat'})
        def open_record(aid):st.query_params.update(analysis=aid,conversation='new')
        open_source(st,open_record,'sample0');self.assertEqual(st.query_params['analysis'],'sample0')
        return_origin(st);self.assertEqual(st.query_params,{'analysis':'original','conversation':'chat'})
    def test_return_to_batch_page(self):
        st=NS(session_state={'workspace_entry':'批量复核'},query_params={})
        open_source(st,lambda aid:st.query_params.update(analysis=aid),'a');return_origin(st)
        self.assertEqual(st.session_state['workspace_entry'],'批量复核');self.assertEqual(st.query_params,{})
    def test_one_source_selector_for_one_hundred_records(self):
        class UI:
            def __init__(self):self.selectors=0;self.buttons=0;self.expanders=0
            def expander(self,*a,**k):self.expanders+=1;return self
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def caption(self,*a):pass
            def json(self,*a):pass
            def selectbox(self,l,options,**kw):self.selectors+=1;return options[0]
            def button(self,*a,**kw):self.buttons+=1;assert 'on_click' in kw
        st=UI();render_sources(st,{'evidence':{'e':self.fixture()}},lambda a:None,'turn')
        self.assertEqual((st.selectors,st.buttons,st.expanders),(1,1,1))
if __name__=='__main__':unittest.main()
