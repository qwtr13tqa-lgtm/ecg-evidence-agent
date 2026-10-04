import json
import unittest
from types import SimpleNamespace as NS
import numpy as np
from src.review.cross_record_agent import run,bootstrap,query_record,tools

class Result:
    def __init__(self,i,score):
        self.analysis_id='saved-'+str(i);self.input=NS(num_samples=1000,sampling_rate=500)
        self.model=NS(anomaly_score=score,error_map=np.full((1000,12),float(i)))
        self.provenance={'input_sha256':str(i)*64,'checkpoint_sha256':'c','review_adapter_sha256':'a',
            'sample_index':i,'crop_start_sample':100,'model_decision':{'status':'configured','threshold':.5,'prediction':'model_normal' if score<.5 else 'model_anomaly'}}
    def to_llm_context(self):
        return {'input':{'num_samples':1000,'sampling_rate':500},'model':{'anomaly_score':self.model.anomaly_score,'reconstruction_error':.1,'shape_error':.2},
            'evidence':{'temporal_regions':[],'lead_evidence':[{'rank':1,'lead':'V2'}]},'signal_features':{}}
class History:
    def __init__(self):self.results={'saved-0':Result(0,.1),'saved-1':Result(1,.9)}
    def list_analyses(self,limit,offset):return [{'analysis_id':aid,'sample_index':r.provenance['sample_index']} for aid,r in self.results.items()][offset:offset+limit]
    def load_analysis(self,aid):return self.results[aid],None,None
REPORT={'threshold':.5,'checkpoint_sha256':'c','adapter_sha256':'a','rows':[{'index':i,'label':1,'prediction':int(s>=.5),'score':s,'input_sha256':str(i)*64} for i,s in enumerate((.1,.9,.2))]}
def call(name,args,identifier='call'):
    return {'id':identifier,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}
class Gateway:
    def __init__(self,queries=None,wrong=False):self.n=0;self.queries=queries;self.wrong=wrong
    def complete(self,messages,tools):
        self.n+=1
        if self.n==1 and self.queries:return {'finish_reason':'tool_calls','tool_calls':self.queries}
        first=json.loads(messages[1]['content']);eid=first['evidence_id']
        answer={'answer':'FN分数低于阈值；现有数据不足以确定深层原因。','evidence_ids':[eid],'knowledge_ids':[],
            'observations':[{'evidence_id':eid,'path':'/cohort_counts/fn','value':999 if self.wrong else 2}]}
        return {'finish_reason':'tool_calls','tool_calls':[call('submit_answer',answer,'submit')]}
class Tests(unittest.TestCase):
    def test_bootstrap_fn_tp_counts_and_missing(self):
        e,rows=bootstrap(REPORT,History(),'root','hash');self.assertEqual(e['data']['cohort_counts']['fn'],2)
        self.assertEqual(rows[2]['history_status'],'missing')
    def test_cross_record_window_source_binding(self):
        h=History();_,rows=bootstrap(REPORT,h,'root','hash')
        a=query_record(h,REPORT,rows,'root','inspect_error_window',{'sample_index':0,'start_sample':100,'end_sample':200,'lead':'V2'})
        b=query_record(h,REPORT,rows,'root','inspect_error_window',{'sample_index':1,'start_sample':100,'end_sample':200,'lead':'V2'})
        self.assertEqual(a['data']['leads'][0]['mean'],0.);self.assertEqual(b['data']['leads'][0]['mean'],1.)
        self.assertNotEqual(a['evidence_id'],b['evidence_id']);self.assertEqual(a['analysis_id'],'root')
        self.assertEqual(b['data']['source_analysis_id'],'saved-1')
    def test_outside_scope_rejected(self):
        h=History();_,rows=bootstrap(REPORT,h,'root','hash')
        with self.assertRaises(ValueError):query_record(h,REPORT,rows,'root','get_analysis_summary',{'sample_index':99})
    def test_missing_analysis_rejected(self):
        h=History();_,rows=bootstrap(REPORT,h,'root','hash')
        with self.assertRaises(ValueError):query_record(h,REPORT,rows,'root','get_analysis_summary',{'sample_index':2})
    def test_rechecks_version(self):
        h=History();_,rows=bootstrap(REPORT,h,'root','hash');h.results['saved-0'].provenance['checkpoint_sha256']='changed'
        with self.assertRaises(ValueError):query_record(h,REPORT,rows,'root','get_analysis_summary',{'sample_index':0})
    def test_actual_tool_loop(self):
        g=Gateway([call('get_analysis_summary',{'sample_index':0},'a'),call('get_analysis_summary',{'sample_index':1},'b')])
        out=run(History(),g,'root','漏报共性',REPORT,'hash');self.assertEqual(out['status'],'completed_draft');self.assertEqual(out['tool_calls'],2)
        self.assertEqual(len(out['evidence']),3)
    def test_wrong_scalar_rejected(self):
        out=run(History(),Gateway(wrong=True),'root','共性',REPORT,'hash');self.assertEqual(out['error'],'OBSERVATION_VALUE_MISMATCH')
        self.assertEqual(out['draft'],{})
    def test_no_forced_tail_query(self):
        out=run(History(),Gateway(),'root','概览',REPORT,'hash');self.assertEqual(out['tool_calls'],0)
    def test_gateway_failure_preserved(self):
        class Bad:
            def complete(self,*args):raise TimeoutError()
        out=run(History(),Bad(),'root','概览',REPORT,'hash');self.assertEqual(out['error'],'GATEWAY_REQUEST_FAILED');self.assertEqual(out['model_calls'],1)
    def test_all_record_tools_require_index(self):
        for t in tools():
            if t['function']['name']!='submit_answer':self.assertIn('sample_index',t['function']['parameters']['required'])
    def test_duplicate_call_counted_without_execution(self):
        g=Gateway([call('get_analysis_summary',{'sample_index':0},'a'),call('get_analysis_summary',{'sample_index':0},'b')])
        out=run(History(),g,'root','概览',REPORT,'hash');self.assertTrue(any(t.get('error')=='REPEATED_TOOL_CALL' for t in out['trace']))
if __name__=='__main__':unittest.main()
