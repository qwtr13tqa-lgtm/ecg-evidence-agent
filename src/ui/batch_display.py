"""Display-only projections. Never mutate stored evidence or merge unlike queries."""
import json
LABELS={'index':'样本','category':'类别','label':'标签','prediction':'预测','score':'模型分数','threshold':'阈值','score_margin':'距阈值', 'reconstruction_error':'重构项','shape_error':'形状项','region_count':'区域数','rr_cv':'RR变异系数','valid_n':'有效数','missing_n':'缺失数','mean':'均值','median':'中位数','q1':'下四分位','q3':'上四分位','minimum':'最小值','maximum':'最大值'}
CATEGORIES={'fn':'漏报 FN','tp':'正确检出 TP','fp':'误报 FP','tn':'正常判对 TN'}
def display_rows(rows):
    return [{LABELS.get(k,k):(round(v,4) if isinstance(v,float) else CATEGORIES.get(v,v) if isinstance(v,str) else v)
        for k,v in row.items() if k not in ('source_analysis_id','analysis_id','input_sha256','batch_sha256')} for row in rows]
def assumptions(entries):
    return list(dict.fromkeys(a for e in entries for a in e.get('data',{}).get('assumptions',[])))
def comparison(entries):
    """Only combine disjoint category groups with otherwise identical operations."""
    aggregates=[e['data'] for e in entries if e.get('data',{}).get('groups')]
    if not aggregates:return []
    signatures=[];groups={}
    for d in aggregates:
        op=dict(d.get('operation') or {})
        if op.get('group_by')!='category' or d.get('has_more'):return []
        filters=op.pop('filters',[])
        if any(f.get('field')!='category' or f.get('op') not in ('eq','in') for f in filters):return []
        signatures.append(json.dumps([d.get('batch_sha256'),op],sort_keys=True))
        for g in d['groups']:
            cat=g['group_value']
            if cat in groups:return []
            groups[cat]=g
    if len(set(signatures))!=1 or not {'fn','tp'}<=groups.keys():return []
    fields=list(groups['fn'].get('metrics',{}))
    if set(fields)!=set(groups['tp'].get('metrics',{})):return []
    rows=[]
    for field in fields:
        row={'指标':LABELS.get(field,field)}
        for cat in ('fn','tp'):
            g=groups[cat];m=g['metrics'][field];prefix=cat.upper()
            row[prefix+' 中位数']=m.get('median')
            row[prefix+' 有效 / 总数']=f"{m.get('valid_n',0)} / {g['count']}"
        rows.append(row)
    return display_rows(rows)
