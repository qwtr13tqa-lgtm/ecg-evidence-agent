"""Final-answer projection only; canonical evidence and JSON pointers stay unchanged."""
import json
import math
from collections import defaultdict
from copy import deepcopy

METRICS=('reconstruction_error','shape_error','region_count','rr_cv','score')
ROW_FIELDS=('index','sample_index','category','score','score_margin','reconstruction_error','shape_error','region_count','rr_cv','history_status','top_lead')

def pointers(value,path=''):
    if isinstance(value,dict):
        for k,v in value.items():
            yield from pointers(v,path+'/'+str(k).replace('~','~0').replace('/','~1'))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from pointers(v,path+'/'+str(i))
    else:yield {'path':path,'value':value}

def selected_rows(rows,limit=12):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[str(row.get('category','unknown'))].append(i)
    candidates=[]
    for field in METRICS:
        for category in sorted(groups):
            valid=[i for i in groups[category] if type(rows[i].get(field)) in (float,int) and math.isfinite(rows[i][field])]
            if valid:
                ordered=sorted(valid,key=lambda i:(rows[i][field],i))
                candidates.extend((ordered[0],ordered[-1]))
    candidates.extend(range(len(rows)))
    return list(dict.fromkeys(candidates))[:limit]

def project(evidence):
    views=[];before=after=0
    for eid,e in evidence.items():
        data=e['data'];rows=data.get('rows')
        item={'evidence_id':eid,'scope':e.get('scope'),'scalars':[]}
        if isinstance(rows,list) and all(isinstance(r,dict) for r in rows):
            chosen=selected_rows(rows);before+=len(rows);after+=len(chosen)
            item['row_selection']={'total_rows':len(rows),'shown_rows':len(chosen),'omitted_rows':len(rows)-len(chosen),
                'original_positions':chosen,'policy':'per-category metric extremes, deduplicated, maximum 12; descriptive examples, not proven support or counterexamples'}
            for k,v in data.items():
                if k!='rows':item['scalars'].extend(pointers(v,'/'+str(k).replace('~','~0').replace('/','~1')))
            for i in chosen:
                for field in ROW_FIELDS:
                    if field in rows[i]:item['scalars'].extend(pointers(rows[i][field],f'/rows/{i}/{field}'))
        else:item['scalars']=list(pointers(data))
        views.append(item)
    return views,{'rows_before':before,'rows_retained':after,'rows_omitted':before-after}

def build_final_messages(system,question,evidence,old_messages):
    view,stats=project(evidence)
    policy='''\n最终整理阶段：只允许提交答案，禁止补查。以下scalars是原始证据的路径和值，引用path原样复制；不得将展示位置当作原数组下标。完整证据仍在本地。
正文目标600字，按“差异与有效数/缺失、代表记录及反例、不确定性”组织；支持与反例各最多3条，明细在页面表格中查看。代表候选按各类别指标极值确定，不预先认定其支持某结论。若未保留所需记录、缺少支持或反例，明确未完成部分，不宣称全量核查，也不编造。用户明确要求完整明细时说明正文仅摘录，不能将摘录说成全量。observations保留支撑实际陈述必需的原始标量。'''
    messages=[{'role':'system','content':system+policy},{'role':'user','content':json.dumps({'evidence_projection':view},ensure_ascii=False,separators=(',',':'))},{'role':'user','content':question}]
    size=lambda x:len(json.dumps(x,ensure_ascii=False,separators=(',',':')))
    stats.update(stage='final_evidence_projection',context_before=size(old_messages),context_after=size(messages),
                 original_evidence_preserved=True,original_pointers_preserved=True)
    return messages,stats
