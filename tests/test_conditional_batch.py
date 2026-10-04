import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from src.review.conditional_batch import prepare,run,Session,save

class Result:
    def __init__(self,index,regions):
        self.analysis_id='analysis-'+str(index)
        self.input=NS(sampling_rate=500,num_samples=1000)
        self.model=NS(anomaly_score=1.,error_map=np.arange(12000,dtype=float).reshape(1000,12))
        self.provenance={'input_sha256':str(index)*64,'checkpoint_sha256':'c','review_adapter_sha256':'a',
            'sample_index':index,'crop_start_sample':100,'model_decision':{'status':'configured','threshold':.5,'prediction':'model_anomaly'}}
        self.context={'input':{'num_samples':1000,'sampling_rate':500},'model':{'anomaly_score':1.},
            'evidence':{'temporal_regions':regions,'lead_evidence':[{'lead':'V2','rank':1}]},'signal_features':{}}
    def to_llm_context(self):return copy.deepcopy(self.context)

class History:
    def __init__(self):self.results={0:Result(0,[]),1:Result(1,[{'start':900,'end':1000}])}
    def list_analyses(self,limit,offset):return [{'sample_index':i,'analysis_id':r.analysis_id} for i,r in self.results.items()][offset:offset+limit]
    def load_analysis(self,aid):return next(r for r in self.results.values() if r.analysis_id==aid),None,{}

def fixture():
    report={'threshold':.5,'checkpoint_sha256':'c','adapter_sha256':'a','rows':[
        {'index':i,'label':0,'prediction':1,'score':1.,'input_sha256':str(i)*64} for i in range(3)]}
    return report,History()

class Gateway:
    def __init__(self,steps):self.steps=iter(steps);self.calls=0
    def complete(self,messages,tools):
        self.calls+=1;step=next(self.steps)
        if isinstance(step,Exception):raise step
        return {'finish_reason':'tool_calls','tool_calls':[{'id':str(self.calls)+'-'+str(i),'type':'function',
            'function':{'name':name,'arguments':json.dumps(args)}} for i,(name,args) in enumerate(step)]}

class BatchConditionalTests(unittest.TestCase):
    def setup(self):
        report,history=fixture();return prepare(report,history,'fp')
    def test_matching_and_missing_states(self):
        s,_=self.setup();self.assertEqual([r['branch'] for r in s['rows']],['supplement','keep_regions','blocked'])
        self.assertEqual(s['rows'][2]['state'],'needs_local_analysis')
    def test_fixed_exact_window_and_coverage(self):
        s,o=self.setup();r=run(s,o)
        self.assertEqual(r['status'],'completed');self.assertEqual(r['metrics']['record_coverage'],1)
        self.assertEqual(r['metrics']['matched_evidence_rate'],2/3)
        d=r['rows'][0]['window']['data'];self.assertEqual((d['start_sample'],d['end_sample']),(700,1000))
        self.assertEqual(d['leads'][0]['lead'],'V2')
        self.assertEqual(d['leads'][0]['maximum'],float(o[0].model.error_map[700:,7].max()))
        self.assertEqual(r['metrics']['window_computations'],1)
    def test_agent_same_results_as_fixed(self):
        s,o=self.setup();g=Gateway([[('collect_batch_evidence',{})],[('inspect_tail_window',{'index':0})],[('submit_batch_review',{})]])
        a=run(s,o,'agent',g);b=run(s,o)
        self.assertEqual(a['rows'],b['rows']);self.assertEqual(a['metrics']['model_requests'],3)
        self.assertEqual(a['metrics']['branch_decision_accuracy'],1)
    def test_early_submit_rejected_then_recover(self):
        s,o=self.setup();g=Gateway([[('collect_batch_evidence',{})],[('submit_batch_review',{})],
            [('inspect_tail_window',{'index':0})],[('submit_batch_review',{})]])
        r=run(s,o,'agent',g);self.assertEqual(r['status'],'completed');self.assertEqual(r['metrics']['rejected_calls'],1)
    def test_out_of_scope_and_nonempty_rejected(self):
        s,o=self.setup();session=Session(s,o);session.execute('collect_batch_evidence',{})
        for index in (1,2,999):self.assertFalse(session.execute('inspect_tail_window',{'index':index})['ok'])
        self.assertFalse(session.attempted)
    def test_duplicate_does_not_compute_again(self):
        s,o=self.setup();session=Session(s,o);session.execute('collect_batch_evidence',{})
        self.assertTrue(session.execute('inspect_tail_window',{'index':0})['ok'])
        self.assertFalse(session.execute('inspect_tail_window',{'index':0})['ok']);self.assertEqual(len(session.attempted),1)
    def test_missing_metadata_not_empty(self):
        report,h=fixture();del h.results[0].context['evidence']['temporal_regions']
        s,_=prepare(report,h,'fp');self.assertEqual(s['rows'][0]['branch'],'blocked')
    def test_wrong_version_not_loaded(self):
        report,h=fixture();h.results[0].provenance['review_adapter_sha256']='wrong'
        s,_=prepare(report,h,'fp');self.assertEqual(s['rows'][0]['state'],'needs_local_analysis')
    def test_wrong_sample_not_loaded(self):
        report,h=fixture();h.results[0].provenance['sample_index']=2
        s,_=prepare(report,h,'fp');self.assertEqual(s['rows'][0]['state'],'needs_local_analysis')
    def test_bad_map_reports_partial_failure(self):
        s,o=self.setup();o[0].model.error_map=np.zeros((2,2));r=run(s,o)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['metrics']['supplement_completion'],0)
        self.assertEqual(r['rows'][0]['query_state'],'failed')
    def test_gateway_timeout_saved_as_failure(self):
        s,o=self.setup();r=run(s,o,'agent',Gateway([TimeoutError()]))
        self.assertEqual(r['error'],'GATEWAY_REQUEST_FAILED');self.assertEqual(r['metrics']['model_requests'],1)
        self.assertEqual(r['metrics']['record_coverage'],0)
    def test_budget_ends_without_success(self):
        s,o=self.setup();r=run(s,o,'agent',Gateway([[('collect_batch_evidence',{})]]),max_requests=1)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['error'],'MODEL_BUDGET_EXHAUSTED')
    def test_no_mutation(self):
        s,o=self.setup();old=copy.deepcopy(s);before=o[0].model.error_map.copy();run(s,o)
        self.assertEqual(old,s);np.testing.assert_array_equal(before,o[0].model.error_map)
    def test_save_unique_and_original_preserved(self):
        s,o=self.setup()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'original.json';p.write_text('original');a=save(d,run(s,o));b=save(d,run(s,o))
            self.assertNotEqual(a,b);self.assertEqual(p.read_text(),'original')
    def test_fingerprint_tracks_error_map(self):
        report,h=fixture();a,_=prepare(report,h,'fp');h.results[0].model.error_map[0,0]+=1;b,_=prepare(report,h,'fp')
        self.assertNotEqual(a['fingerprint'],b['fingerprint'])
    def test_unknown_lead_is_blocked(self):
        report,h=fixture();h.results[0].context['evidence']['lead_evidence'][0]['lead']='X'
        s,_=prepare(report,h,'fp');self.assertEqual(s['rows'][0]['branch'],'blocked')
    def test_wrong_branch_is_visible_even_after_recovery(self):
        s,o=self.setup();g=Gateway([[('collect_batch_evidence',{})],[('inspect_tail_window',{'index':1})],
            [('inspect_tail_window',{'index':0})],[('submit_batch_review',{})]])
        r=run(s,o,'agent',g);self.assertEqual(r['status'],'completed')
        self.assertEqual(r['metrics']['unnecessary_query_attempts'],1)
        self.assertEqual(r['metrics']['branch_decision_accuracy'],.5)
    def test_requires_explicit_gateway(self):
        s,o=self.setup()
        with self.assertRaises(ValueError):run(s,o,'agent')

if __name__=='__main__':unittest.main()
