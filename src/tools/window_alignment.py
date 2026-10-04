"""Time overlap, not independent validation or fusion diagnosis."""
from src.analysis.rr_facts import build_rr_facts
from src.tools.ecg_tools import inspect_recent_error, inspect_error_window, LEADS


def inspect_window_rr_alignment(result, start_sample, end_sample, lead='V1'):
    window=inspect_error_window(result,start_sample,end_sample,lead)
    facts=build_rr_facts(result)
    output={'window':window,'rr_status':facts['status'],'rr_errors':facts['errors'],
            'rr_intervals':[],'overlapping_rr_count':0,'returned_rr_count':0,
            'rr_measurement_status':'unvalidated','fusion_validated':False,
            'overlap_rule':'left_peak < window_end and right_peak > window_start',
            'scope':'time_overlap_only_not_independent_validation'}
    if facts['status']!='verified':return output
    fs=result.input.sampling_rate;crop=result.provenance.get('crop_start_sample',0)
    rows=[]
    for row in facts['rr_intervals']:
        left,right=row['left_peak_sample'],row['right_peak_sample']
        if left<end_sample and right>start_sample:
            lo,hi=max(left,start_sample),min(right,end_sample)
            rows.append({**row,'original_left_peak_sample':left+crop,'original_right_peak_sample':right+crop,
                'overlap_start_sample':lo,'overlap_end_sample':hi,'overlap_seconds':(hi-lo)/fs,
                'fully_contained':left>=start_sample and right<=end_sample})
    output.update(rr_intervals=rows[:50],overlapping_rr_count=len(rows),returned_rr_count=min(50,len(rows)),
        truncated=len(rows)>50,rhythm_lead=LEADS[result.rhythm.lead_index],
        same_lead=lead==LEADS[result.rhythm.lead_index],
        retained_overlap_count=sum(r['retained'] for r in rows),
        excluded_overlap_count=sum(not r['retained'] for r in rows),
        note='完整RR时长不是重叠时长。边界外未形成的RR不推断。不同导联仅按同步时间对齐；不能据此确定或排除疾病。')
    return output


def inspect_recent_rr_alignment(result,duration_seconds,lead='V1'):
    w=inspect_recent_error(result,duration_seconds,lead)
    out=inspect_window_rr_alignment(result,w['start_sample'],w['end_sample'],lead)
    out['window']=w
    return out
