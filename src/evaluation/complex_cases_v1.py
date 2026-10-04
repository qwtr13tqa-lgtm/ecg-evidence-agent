"""Frozen questions and array-based references, never sent to either planner."""
import math
import numpy as np

QUESTIONS={
 'SELECT_LEAD':'比较V1和V2最后1.2秒的模型分数最大值。选择最大值较高的导联（相同选择V1），再查询该导联最后1.8秒的均值和最大值。报告两次比较值、所选导联、最终窗口起止采样点、均值和最大值。所有坐标使用输入片段坐标。',
 'RR_WINDOW':'找出全部保留RR中最长的间隔（相同选择rr_index最小的）。给出rr_index、左右峰采样点，再查询V1在这两个峰之间半开窗口的模型分数均值和最大值。使用输入片段坐标，若保留RR为空请说明无法查询。',
 'REGION_FALLBACK':'如果摘要有候选时间区域，选择score最高的区域（相同选择列表中第一个），查询V1在该区域的模型分数均值和最大值；如果区域为空，则查询V1最后0.6秒。说明使用了哪个分支，给出导联、起止采样点、均值和最大值。窗口不是病变定位。使用输入片段坐标。'
}


def reference(result,kind):
    values=np.asarray(result.model.error_map);n=result.input.num_samples;fs=result.input.sampling_rate
    if values.shape!=(n,12) or not np.isfinite(values).all(): raise ValueError('INVALID_ERROR_MAP')
    windows=[];targets=[]
    def window(lead,start,end,fields):
        if not 0<=start<end<=n: raise ValueError('INVALID_REFERENCE_WINDOW')
        x=values[start:end,6 if lead=='V1' else 7]
        w={'lead':lead,'start_sample':int(start),'end_sample':int(end)}
        windows.append(w)
        expected={**w,'mean':float(np.mean(x)),'maximum':float(np.max(x))}
        for field in fields: targets.append({'name':f'{len(windows)}.{field}','window':w,'field':field,'value':expected[field]})
        return expected
    tail=lambda sec:n-int(math.floor(sec*fs+.5))
    if kind=='SELECT_LEAD':
        a=window('V1',tail(1.2),n,['maximum']);b=window('V2',tail(1.2),n,['maximum'])
        lead='V1' if a['maximum']>=b['maximum'] else 'V2'
        window(lead,tail(1.8),n,['lead','start_sample','end_sample','mean','maximum']);branch=lead
    elif kind=='RR_WINDOW':
        r=result.rhythm
        if r is None or not r.rr_details: raise ValueError('RR_UNAVAILABLE')
        peaks=[int(p) for p in r.r_peaks];mask=r.rr_details['valid_mask']
        if len(peaks)-1!=len(mask) or len(mask)>100: raise ValueError('RR_OUT_OF_PILOT_SCOPE')
        rows=[(i,peaks[i],peaks[i+1]) for i,keep in enumerate(mask) if keep]
        if not rows: raise ValueError('NO_RETAINED_RR_PILOT_INELIGIBLE')
        i,left,right=max(rows,key=lambda row:row[2]-row[1])
        for key,val in [('rr_index',i),('left_peak_sample',left),('right_peak_sample',right)]:
            targets.append({'name':key,'rr_index':i,'field':key,'value':val})
        window('V1',left,right,['mean','maximum']);branch='longest_retained_rr'
    elif kind=='REGION_FALLBACK':
        regions=result.to_llm_context()['evidence']['temporal_regions']
        if regions:
            row=max(regions,key=lambda r:r['score']);start,end=int(row['start']),int(row['end']);branch='region'
        else:start,end=tail(.6),n;branch='fallback'
        window('V1',start,end,['lead','start_sample','end_sample','mean','maximum'])
    else: raise ValueError('UNKNOWN_CASE')
    return {'analysis_id':result.analysis_id,'branch':branch,'windows':windows,'targets':targets}


def score(output,ref):
    from src.agent.answer_validator import validate_answer
    from src.evaluation.complex_plan_v1 import pointer
    evidence=output.get('evidence',{});draft=output.get('draft',{});observations=draft.get('observations',[])
    structure=False
    try:
        validate_answer(draft,evidence,{})
        structure=output.get('status')=='completed_draft' and output.get('analysis_id')==ref['analysis_id'] and all(
            x.get('analysis_id')==ref['analysis_id'] for x in evidence.values())
    except Exception: pass
    checks={}
    for t in ref['targets']:
        found=False
        for obs in observations if structure else []:
            data=evidence.get(obs.get('evidence_id'),{}).get('data',{});path=obs.get('path','')
            if 'window' in t:
                w=t['window']
                if (data.get('start_sample'),data.get('end_sample'))!=(w['start_sample'],w['end_sample']):continue
                rows=data.get('leads',[])
                leadidx=next((i for i,r in enumerate(rows) if r.get('lead')==w['lead']),None)
                if leadidx is None:continue
                expected_path='/'+t['field'] if t['field'] in ('start_sample','end_sample') else f'/leads/{leadidx}/'+t['field']
            else:
                idx=next((i for i,r in enumerate(data.get('intervals',[])) if r.get('rr_index')==t['rr_index'] and r.get('retained') is True),None)
                if idx is None:continue
                expected_path=f'/intervals/{idx}/'+t['field']
            if path!=expected_path:continue
            val=pointer(data,path);want=t['value']
            if type(want) is float: found=type(val) in (int,float) and math.isclose(val,want,rel_tol=1e-6,abs_tol=1e-7)
            else:found=type(val) is type(want) and val==want
            if found:break
        checks[t['name']]=found
    # Query coverage is separate from cited-value coverage. Query extras are heuristic, not full semantic redundancy.
    queried=[];trace=output.get('trace',[])
    for event in trace:
        if event.get('stage')!='tool' or not event.get('ok'):continue
        d=evidence.get(event.get('evidence_id'),{}).get('data',{})
        for r in d.get('leads',[]): queried.append({'lead':r['lead'],'start_sample':d.get('start_sample'),'end_sample':d.get('end_sample')})
    execution=all(w in queried for w in ref['windows'])
    tool_events=[t for t in trace if t.get('stage')=='tool' and t.get('source')!='bootstrap']
    identities=[str((t.get('tool'),sorted((t.get('arguments') or {}).items()))) for t in tool_events]
    return {'structure_pass':structure,'field_checks':checks,'target_covered':sum(checks.values()),'target_total':len(checks),
        'required_windows_queried':execution,'automatic_task_pass':structure and all(checks.values()) and execution,
        'extra_window_count':sum(w not in ref['windows'] for w in queried),'duplicate_query_count':len(identities)-len(set(identities)),
        'supplemental_queries':len(tool_events),'semantic_review':'pending',
        'scope':'automatic is structured field + object + required query coverage; prose branch explanation and full claim support require human review'}
