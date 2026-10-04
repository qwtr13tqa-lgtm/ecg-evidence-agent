"""Offline capability benchmark contracts; no weights, API key or network."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.analysis.result import ECGAnalysisResult,ECGInputInfo,ECGModelOutput,ECGEvidenceSummary
from src.analysis.store import AnalysisStore
from src.features.rhythm import RhythmFeatures
from src.evaluation.capability import baseline_evidence,run_fixed,reference,score,FIXED_SCHEDULE
from src.evaluation.records import RunRecord,atomic_json
from evaluation.run_capability import load_suite
from evaluation.summarize_capability import collect


def fixture():
    store=AnalysisStore();aid=store.begin()
    peaks=[100,500,1000,1550,2000,2500,3000,3500,4000,4700]
    rr=[(b-a)/500 for a,b in zip(peaks,peaks[1:])];mean=sum(rr)/len(rr)
    details={'raw_rr_seconds':rr,'valid_mask':[True]*9,'retained_rr_seconds':rr,
        'exclusion_reasons':[None]*9,'parameters':{'sampling_rate':500,'lead_index':1,
        'min_hr':30.,'max_hr':220.,'min_rr_seconds':60/220,'max_rr_seconds':2.}}
    rhythm=RhythmFeatures(60/mean,mean,float(np.median(rr)),float(np.std(rr)),float(np.std(rr)/mean),10,peaks,None,500,rr_details=details)
    error=np.arange(4800*12,dtype=float).reshape(4800,12)/1000
    result=ECGAnalysisResult(ECGInputInfo(4800,12),ECGModelOutput(-.9,-.95,.05,error_map=error),
        ECGEvidenceSummary(),rhythm=rhythm,analysis_id=aid,provenance={'crop_start_sample':100})
    store.complete(aid,result);return store,result


def answer_output(aid,data,paths):
    eid=aid+':test';obs=[]
    for path in paths:
        value=data
        for token in path[1:].split('/'):
            value=value[int(token)] if isinstance(value,list) else value[token]
        obs.append({'evidence_id':eid,'path':path,'value':value})
    return {'analysis_id':aid,'status':'completed_draft','draft':{'answer':'测试回答',
        'evidence_ids':[eid],'knowledge_ids':[],'observations':obs},
        'evidence':{eid:{'analysis_id':aid,'evidence_id':eid,'ok':True,'data':data}},'knowledge':{},'trace':[]}


class CapabilityTests(unittest.TestCase):
    def setUp(self):self.store,self.result=fixture();self.aid=self.result.analysis_id
    def test_ablations_do_not_leak_professional_fields(self):
        for scheme,expected in [('pure_llm',{'input'}),('sgrf',{'input','model','evidence','decision'}),
                                ('sgrf_rhythm',{'input','model','evidence','decision','signal_features'})]:
            pool,_=baseline_evidence(self.store,self.aid,scheme)
            self.assertEqual(set(next(iter(pool.values()))['data']),expected)
    def test_fixed_schedule_uses_real_tools(self):
        pool,trace=baseline_evidence(self.store,self.aid,'fixed_tools')
        self.assertEqual([t['tool'] for t in trace],[n for n,a in FIXED_SCHEDULE])
        self.assertTrue(all(t['ok'] for t in trace));self.assertEqual(len(pool),5)
    def test_mock_submission_and_no_reference_leak(self):
        captured=[]
        def complete(messages,tools):
            captured.extend(messages);payload=json.loads(messages[1]['content']);eid=next(iter(payload['evidence']))
            draft={'answer':'仅有输入元信息，无法判断。','evidence_ids':[eid],'knowledge_ids':[],
                'observations':[{'evidence_id':eid,'path':'/input/num_samples','value':4800}]}
            return {'finish_reason':'tool_calls','tool_calls':[{'type':'function','function':{'name':'submit_answer','arguments':json.dumps(draft)}}]}
        out=run_fixed(self.store,self.aid,'测试问题','pure_llm',SimpleNamespace(complete=complete))
        self.assertEqual(out['status'],'completed_draft')
        self.assertNotIn('reference',json.dumps(captured));self.assertNotIn('rubric',json.dumps(captured))
    def test_gateway_failure_retained(self):
        def complete(*args):raise TimeoutError()
        out=run_fixed(self.store,self.aid,'测试','pure_llm',SimpleNamespace(complete=complete))
        self.assertEqual(out['error'],'GATEWAY_REQUEST_FAILED');self.assertEqual(out['model_calls'],1)
    def test_direct_reference(self):
        ref=reference(self.result)
        self.assertEqual(ref['window']['start_sample'],4500)
        self.assertEqual(ref['window']['peak_sample'],4799)
        self.assertEqual(ref['rr']['longest_index'],8)
        self.assertEqual(ref['alignment']['overlapping_rr_count'],1)
        self.assertEqual(reference(self.result,duration=.601)['window']['start_sample'],4499)
        with self.assertRaises(ValueError):reference(self.result,duration=20)
    def test_rr_equivalent_paths(self):
        from src.tools.ecg_tools import inspect_rr_intervals
        data=inspect_rr_intervals(self.result)
        paths=['/total_intervals','/retained_count','/excluded_count']+['/calculation_facts/'+k for k in ('candidate_peak_count','mean_rr_seconds','heart_rate_bpm')]
        out=answer_output(self.aid,data,paths)
        self.assertTrue(score('rr',out,reference(self.result))['metrics']['task_observations'])
    def test_wrong_copied_value_rejected(self):
        out=answer_output(self.aid,{'total_intervals':9},['/total_intervals'])
        out['draft']['observations'][0]['value']=10
        self.assertFalse(score('rr',out,reference(self.result))['metrics']['structure_values_references'])
    def test_foreign_analysis_rejected(self):
        out=answer_output(self.aid,{'total_intervals':9},['/total_intervals'])
        next(iter(out['evidence'].values()))['analysis_id']='foreign'
        self.assertFalse(score('rr',out,reference(self.result))['metrics']['analysis_binding'])
    def test_window_and_wrong_lead(self):
        from src.tools.ecg_tools import inspect_recent_error
        data=inspect_recent_error(self.result,.6,'V1')
        paths=['/start_sample','/end_sample','/leads/0/mean','/leads/0/maximum']
        out=answer_output(self.aid,data,paths);ref=reference(self.result)
        self.assertTrue(score('window',out,ref)['metrics']['task_observations'])
        data['leads'][0]['lead']='I'
        self.assertFalse(score('window',out,ref)['metrics']['task_observations'])
    def test_longest_must_come_from_same_interval(self):
        from src.tools.ecg_tools import inspect_rr_intervals
        data=inspect_rr_intervals(self.result);paths=['/intervals/8/'+k for k in ('rr_index','rr_seconds','left_peak_sample','right_peak_sample')]
        out=answer_output(self.aid,data,paths);ref=reference(self.result)
        self.assertTrue(score('longest',out,ref)['metrics']['task_observations'])
        data['intervals'][8]['rr_index']=0;out=answer_output(self.aid,data,paths)
        self.assertFalse(score('longest',out,ref)['metrics']['task_observations'])
    def test_alignment_reference(self):
        from src.tools.window_alignment import inspect_recent_rr_alignment
        data=inspect_recent_rr_alignment(self.result,.6,'V1')
        out=answer_output(self.aid,data,['/overlapping_rr_count','/window/start_sample','/window/end_sample'])
        self.assertTrue(score('alignment',out,reference(self.result))['metrics']['task_observations'])
    def test_no_threshold_does_not_invent_prediction(self):
        from src.analysis.model_decision import get_model_decision
        out=answer_output(self.aid,get_model_decision(self.result),['/score','/threshold','/prediction','/status'])
        self.assertTrue(score('decision',out,reference(self.result))['metrics']['task_observations'])
    def test_suite_and_long_history_probe(self):
        cases=load_suite(Path(__file__).resolve().parents[1]/'evaluation/capability_cases.jsonl')
        self.assertEqual(len(cases),48);self.assertEqual(sum(len(c['turns']) for c in cases),78)
        memory=next(c for c in cases if c['id']=='CAP_MEMORY_S0')
        self.assertEqual(len(memory['turns']),6);self.assertEqual(memory['reference_window'],{'lead':'V2','duration':1.2})
    def test_failure_missing_and_unreviewed_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch='batch'
            atomic_json(root/(batch+'.manifest.json'),{'tasks':[{'case_id':'C','scheme':'agent','repetition':1,'turn_count':3}]})
            for i,status in [(1,'completed_draft'),(2,'failed')]:
                r=RunRecord(root,{'id':'C','turn_index':i,'question':'q','rubric':[]},{'batch_id':batch,'scheme':'agent','repetition':1})
                r.finish({'status':status,'draft':{},'error':'TIMEOUT'}, {}, {'metrics':{'task_observations':False}},wall_seconds=1)
            report,packet=collect(root,batch);g=report['groups']['agent']
            self.assertEqual((g['planned'],g['recorded'],g['completed'],g['missing']),(3,2,1,1))
            self.assertEqual(g['manual']['task_correct']['unscored'],3)
            path=root/packet[0]['run_id']/'result.json';path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):collect(root,batch)

if __name__=='__main__':unittest.main()
