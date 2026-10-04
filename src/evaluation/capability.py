"""Capability ablations. Never pass evaluator references or task kinds to models."""
import copy
import json
import math
import re
import time
import numpy as np
from src.agent.answer_validator import validate_answer, scalar_paths
from src.agent.answer_parser import parse_answer
from src.agent.tool_protocol import TOOLS, strict_json
from src.tools.executor import ECGToolExecutor

SCHEMES=('pure_llm','sgrf','sgrf_rhythm','fixed_tools','agent','agent_rag')
FIXED_SCHEDULE=(('get_analysis_summary',{}),('get_model_decision',{}),
    ('inspect_rr_intervals',{'offset':0,'limit':50}),
    ('inspect_recent_error',{'duration_seconds':0.6,'lead':'V1'}),
    ('inspect_recent_rr_alignment',{'duration_seconds':0.6,'lead':'V1'}))

class EmptyRetriever:
    knowledge_available = False  # efficient-queries-1.0
    def search(self,**kwargs):return []


def baseline_evidence(store,aid,scheme):
    if scheme not in SCHEMES[:4]:raise ValueError('Not a fixed baseline')
    result=store.get_result(aid);context=copy.deepcopy(result.to_llm_context())
    data={'input':context['input']}
    if scheme in ('sgrf','sgrf_rhythm'):
        from src.analysis.model_decision import get_model_decision
        data.update(model=context['model'],evidence=context['evidence'],decision=get_model_decision(result))
    if scheme=='sgrf_rhythm':data['signal_features']=context['signal_features']
    pool={};trace=[]
    if scheme=='fixed_tools':
        executor=ECGToolExecutor(store,aid)
        for name,args in FIXED_SCHEDULE:
            tick=time.perf_counter();item=executor.execute(name,args)
            trace.append({'stage':'tool','tool':name,'arguments':args,'ok':item['ok'],
                'evidence_id':item.get('evidence_id'),'error':item.get('error'),
                'elapsed_seconds':time.perf_counter()-tick})
            if item['ok']:
                item['data'].pop('provenance',None)
                item['observation_paths']=scalar_paths(item['data'])
                pool[item['evidence_id']]=item
    else:
        eid=aid+':capability:'+scheme
        pool[eid]={'analysis_id':aid,'evidence_id':eid,'ok':True,'data':data,'observation_paths':scalar_paths(data)}
    return pool,trace


def run_fixed(store,aid,question,scheme,gateway):
    evidence,trace=baseline_evidence(store,aid,scheme)
    output={'analysis_id':aid,'status':'failed','error':'','draft':{},'validation':{},
        'evidence':evidence,'knowledge':{},'trace':trace,'model_calls':1,'tool_calls':len(trace),'requires_review':True}
    # Three messages keep the shared ConversationGateway insertion position compatible with Agent.
    messages=[{'role':'system','content':
        '你是ECG研究辅助问答系统。仅依据给定数据回答，不编造测量或诊断。分数不是概率，峰测量未验证。'
        '本轮不能补查。信息不足时说明缺少什么。调用submit_answer提交中文答案及原样复制的标量observations。'
        '至少引用一条已提供证据，缺少专业数据时可引用输入元信息并说明无法判断。历史回答不是本轮证据。'},
        {'role':'user','content':json.dumps({'analysis_id':aid,'evidence':evidence},ensure_ascii=False,allow_nan=False)},
        {'role':'user','content':question}]
    tick=time.perf_counter();phase='gateway'
    try:
        reply=gateway.complete(messages,[t for t in TOOLS if t['function']['name']=='submit_answer'])
        phase='protocol';calls=reply.get('tool_calls',[])
        if reply.get('finish_reason') not in ('stop','tool_calls') or not isinstance(calls,list):raise ValueError('Invalid reply')
        if calls:
            if len(calls)!=1 or calls[0].get('type')!='function' or calls[0]['function']['name']!='submit_answer':raise ValueError('Unexpected tool')
            phase='answer_json';answer=strict_json(calls[0]['function']['arguments'])
        else:phase='answer_json';answer=parse_answer(reply.get('content'))
        phase='validation';validation=validate_answer(answer,evidence,{})
        output.update(status='completed_draft',draft=answer,validation=validation)
        output['trace'].append({'stage':'model','call':1,'elapsed_seconds':time.perf_counter()-tick})
        if calls:output['trace'].append({'stage':'submission','tool':'submit_answer','ok':True});output['tool_calls']+=1
    except Exception as exc:
        output['error']={'gateway':'GATEWAY_REQUEST_FAILED','protocol':'MODEL_PROTOCOL_INVALID','answer_json':'ANSWER_JSON_INVALID','validation':'ANSWER_VALIDATION_FAILED'}[phase]
        output['trace'].append({'stage':'model','error_type':type(exc).__name__,'failure_phase':phase,'elapsed_seconds':time.perf_counter()-tick})
    output['elapsed_seconds']=time.perf_counter()-tick
    return output


def reference(result,lead='V1',duration=.6):
    """Direct array/coordinate computation, not tool output or LLM text."""
    n=result.input.num_samples;fs=result.input.sampling_rate;start=n-int(math.floor(duration*fs+.5))
    lead_index=('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6').index(lead)
    if start<0 or start>=n:raise ValueError('Invalid reference window')
    values=np.asarray(result.model.error_map)[start:n,lead_index]
    if not np.isfinite(values).all():raise ValueError('Nonfinite reference values')
    d=result.provenance.get('model_decision') or {}
    threshold=d.get('threshold') if d.get('status')=='configured' else None
    prediction=('model_anomaly' if result.model.anomaly_score>=threshold else 'model_normal') if threshold is not None else None
    ref={'analysis_id':result.analysis_id,'window':{'start_sample':start,'end_sample':n,'lead':lead,
        'mean':float(values.mean()),'maximum':float(values.max()),'peak_sample':start+int(values.argmax())},
        'decision':{'score':result.model.anomaly_score,'threshold':threshold,'prediction':prediction,
                    'status':d.get('status','unconfigured')},'rr':None,'alignment':None}
    rhythm=result.rhythm
    if rhythm is not None and rhythm.rr_details is not None:
        peaks=[int(x) for x in rhythm.r_peaks];raw=[(b-a)/fs for a,b in zip(peaks,peaks[1:])]
        p=rhythm.rr_details['parameters'];mask=[60/p['max_hr']<=x<=60/p['min_hr'] for x in raw]
        retained=[x for x,k in zip(raw,mask) if k];mean=math.fsum(retained)/len(retained) if retained else None
        longest=int(np.argmax(raw)) if raw else None
        ref['rr']={'candidate_peak_count':len(peaks),'total_intervals':len(raw),
            'retained_count':sum(mask),'excluded_count':len(raw)-sum(mask),'mean_rr_seconds':mean,
            'heart_rate_bpm':60/mean if mean is not None else None,'longest_index':longest,
            'longest_seconds':raw[longest] if longest is not None else None,
            'left_peak_sample':peaks[longest] if longest is not None else None,
            'right_peak_sample':peaks[longest+1] if longest is not None else None}
        ref['alignment']={'overlapping_rr_count':sum(a<n and b>start for a,b in zip(peaks,peaks[1:])),
                          'start_sample':start,'end_sample':n}
    return ref


def score(kind,out,ref):
    aid=ref.get('analysis_id');evidence=out.get('evidence') or {};answer=out.get('draft') or {}
    complete=out.get('status')=='completed_draft';valid=False
    binding=bool(aid and out.get('analysis_id')==aid and all(
        e.get('analysis_id')==aid and eid.startswith(aid+':') and e.get('evidence_id')==eid for eid,e in evidence.items()))
    try:validate_answer(answer,evidence,out.get('knowledge') or {});valid=complete and binding
    except Exception:pass
    obs=answer.get('observations',[]) if valid else []
    def equal(a,b):
        if type(a) in (int,float) and type(b) in (int,float):return math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7)
        return type(a) is type(b) and a==b
    def has(paths,value,condition=lambda data:True):
        return any(o['path'] in paths and equal(o['value'],value) and condition(evidence[o['evidence_id']]['data']) for o in obs)
    def correct_rr_row(o):
        match=re.fullmatch(r'/(intervals|rr_intervals)/(0|[1-9][0-9]*)/[^/]+',o['path'])
        if not match:return False
        data=evidence[o['evidence_id']]['data']
        row=data[match[1]][int(match[2])]
        rr=ref['rr']
        return (data.get('coordinate_reference')=='input_segment' and
            row.get('rr_index')==rr['longest_index'] and
            row.get('left_peak_sample')==rr['left_peak_sample'] and
            row.get('right_peak_sample')==rr['right_peak_sample'])
    def window_ok(data):
        w=data.get('window',data);r=ref['window']
        return w.get('start_sample')==r['start_sample'] and w.get('end_sample')==r['end_sample'] and bool(w.get('leads')) and w['leads'][0].get('lead')==r['lead']
    fields={}
    if kind=='decision':
        for key,value in ref['decision'].items():fields[key]=has(['/'+key,'/decision/'+key],value)
    elif kind in ('window','window_peak'):
        keys=('start_sample','end_sample','mean','maximum') if kind=='window' else ('peak_sample',)
        for key in keys:
            suffix=('/leads/0/'+key) if key in ('mean','maximum','peak_sample') else '/'+key
            fields[key]=has([suffix,'/window'+suffix],ref['window'][key],window_ok)
    elif kind=='rr' and ref.get('rr'):
        for key in ('candidate_peak_count','total_intervals','retained_count','excluded_count','mean_rr_seconds','heart_rate_bpm'):
            paths=['/'+key,'/calculation_facts/'+key,'/signal_features/rhythm/'+('candidate_beat_count' if key=='candidate_peak_count' else key)]
            fields[key]=has(paths,ref['rr'][key])
    elif kind=='longest' and ref.get('rr') and ref['rr']['longest_index'] is not None:
        rr=ref['rr']
        for key,expected in [('rr_index',rr['longest_index']),('rr_seconds',rr['longest_seconds']),('left_peak_sample',rr['left_peak_sample']),('right_peak_sample',rr['right_peak_sample'])]:
            fields[key]=any(o['path'].endswith('/'+key) and equal(o['value'],expected)
                and correct_rr_row(o) for o in obs)
    elif kind=='alignment' and ref.get('alignment'):
        fields['overlapping_rr_count']=has(['/overlapping_rr_count'],ref['alignment']['overlapping_rr_count'],window_ok)
        for key in ('start_sample','end_sample'):fields[key]=has(['/window/'+key],ref['window'][key],window_ok)
    errors=[t for t in out.get('trace',[]) if t.get('error_type')]
    return {'version':'capability-checks-1.0','metrics':{'completed_draft':complete,'structure_values_references':valid,
        'analysis_binding':binding,'task_observations':all(fields.values()) if fields else None},
        'field_checks':fields,'error_type':errors[-1]['error_type'] if errors else None,
        'manual_required':['task_correct','evidence_support','text_complete'],
        'scope':'structured_observations_only; prose, abstention and clinical correctness not automatically graded'}
