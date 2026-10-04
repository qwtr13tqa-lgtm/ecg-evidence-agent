"""Descriptive statistics over report-bound matching histories; no inference/LLM."""
import math
import hashlib
import json
from datetime import datetime,timezone
import numpy as np
from src.review.batch import query,validate
from src.review.open_analysis import compatible

FIELDS={'score':'模型分数','score_margin':'分数减阈值','reconstruction_error':'重构项','shape_error':'形状项',
        'region_count':'候选区域数','heart_rate_bpm':'估计心率','mean_rr_seconds':'平均RR秒','rr_cv':'RR变异系数'}

def number(value):
    return float(value) if type(value) in (int,float) and math.isfinite(value) else None

def collect(report,history,categories):
    validate(report)
    if not categories or not set(categories)<=set(('fn','tp','fp','tn')):raise ValueError('Invalid categories')
    selected={r['index']:(c,r) for c in sorted(set(categories)) for r in query(report,c)}
    records={};offset=0
    while True:
        page=history.list_analyses(limit=100,offset=offset)
        for e in page:records.setdefault(e.get('sample_index'),[]).append(e)
        if len(page)<100:break
        offset+=len(page)
    rows=[]
    for index,(category,source) in sorted(selected.items()):
        row={'index':index,'category':category,'label':source['label'],'prediction':source['prediction'],
             'threshold':report['threshold'],'analysis_id':None,'history_status':'missing',
             'history_candidates':len(records.get(index,[])),'incompatible_count':0,'unreadable_count':0,
             'score':number(source['score']),'score_margin':source['score']-report['threshold'],
             'measurement_status':None,'lead_evidence':None,'temporal_regions':None,'missing_reasons':{}}
        for key in FIELDS:
            row.setdefault(key,None)
        found=None
        for entry in records.get(index,[]):
            try:
                result,_,_=history.load_analysis(entry['analysis_id'])
                if compatible(result,report,source):found=result;break
                row['incompatible_count']+=1
            except (ValueError,KeyError,OSError,TypeError):row['unreadable_count']+=1
        if found is not None:
            ctx=found.to_llm_context();model=ctx.get('model') or {};evidence=ctx.get('evidence') or {}
            rhythm=(ctx.get('signal_features') or {}).get('rhythm') or {}
            row.update(analysis_id=found.analysis_id,history_status='matched',measurement_status=rhythm.get('measurement_status'),
                lead_evidence=evidence.get('lead_evidence'),temporal_regions=evidence.get('temporal_regions'))
            for key in ('reconstruction_error','shape_error'):row[key]=number(model.get(key))
            for key in ('heart_rate_bpm','mean_rr_seconds','rr_cv'):row[key]=number(rhythm.get(key))
            if isinstance(row['temporal_regions'],list):row['region_count']=len(row['temporal_regions'])
        elif row['history_candidates']:row['history_status']='no_compatible_history'
        for key in FIELDS:
            if row[key] is None:row['missing_reasons'][key]='无匹配分析' if found is None else '字段缺失或非有限数值'
        rows.append(row)
    groups=[];lead_groups=[]
    for category in sorted(set(categories)):
        group=[r for r in rows if r['category']==category]
        for key,label in FIELDS.items():
            values=[r[key] for r in group if r[key] is not None]
            groups.append({'category':category,'field':key,'metric':label,'total_n':len(group),'valid_n':len(values),
                'missing_n':len(group)-len(values),'matched_n':sum(r['history_status']=='matched' for r in group),
                'median':float(np.median(values)) if values else None,
                'q1':float(np.percentile(values,25)) if values else None,'q3':float(np.percentile(values,75)) if values else None,
                'source':'批次报告（全部所选记录）' if key in ('score','score_margin') else '版本匹配历史（逐字段有效值）'})
        leads=[]
        for r in group:
            entries=r['lead_evidence']
            top=[e.get('lead') for e in entries if isinstance(e,dict) and e.get('rank')==1 and isinstance(e.get('lead'),str)] if isinstance(entries,list) else []
            if len(top)==1:leads.append(top[0])
        for lead in sorted(set(leads)):
            lead_groups.append({'category':category,'lead':lead,'count':leads.count(lead),'valid_n':len(leads),
                                'missing_n':len(group)-len(leads),'fraction':leads.count(lead)/len(leads)})
    return {'version':'batch-descriptive-1','report_sha256':hashlib.sha256(json.dumps(report,sort_keys=True,allow_nan=False).encode()).hexdigest(),
            'quantile_method':'numpy percentile linear, valid finite values only','created_at':datetime.now(timezone.utc).isoformat(),
            'categories':sorted(set(categories)),'rows':rows,'groups':groups,'top_leads':lead_groups,
            'scope':'描述性统计；不同字段有效样本可能不同；不推断漏报原因；未验证节律仅为估计'}
