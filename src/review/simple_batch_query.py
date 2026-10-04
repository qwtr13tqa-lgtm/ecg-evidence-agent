"""Exact, conservative local routing. Unknown clauses never silently discarded."""
import re
import time
import hashlib
import json
from src.review.batch import query

CATEGORIES={'误报':'fp','漏报':'fn','判对的正常记录':'tn','正确检出的异常记录':'tp','正确检出的异常样本':'tp','判对的正常样本':'tn','FP':'fp','FN':'fn','TP':'tp','TN':'tn','全部记录':'all','所有记录':'all'}
def parse_simple_query(question):
    if not isinstance(question,str) or len(question)>4000:return None
    q=re.sub(r'\s+','',question).rstrip('。.!！')
    cats='|'.join(sorted(map(re.escape,CATEGORIES),key=len,reverse=True))
    m=re.fullmatch(r'(?:(?:你好|您好)[，,！!。:：;；]*)?(?:请|请帮我|麻烦你|帮我)?(?:找出|列出|查看|查询)(?:所有|全部)?('+cats+r')(?:样本|记录)?[，,；;]?(?:按|按照)(?:模型)?分数(降序|升序|从高到低|从低到高)(?:排列|排序)?',q,re.I)
    if not m:return None
    c=m[1].upper() if m[1].upper() in CATEGORIES else m[1]
    return {'category':CATEGORIES[c],'order':'desc' if m[2] in ('降序','从高到低') else 'asc'}

def run_simple(history,aid,question,report,batch):
    args=parse_simple_query(question)
    if args is None:raise ValueError('SIMPLE_QUERY_NOT_SUPPORTED')
    start=time.perf_counter();selected=query(report,**args)
    rows=[{k:r[k] for k in ('index','label','prediction','score','threshold')} for r in selected]
    # Matching history is navigation only; failure must not discard report query results.
    warnings=[];matched={}
    if history is not None and rows:
        try:
            from src.review.conversation_batch import review
            matched={r['index']:r for r in review(report,history,args['category'])}
        except Exception as exc:warnings.append({'stage':'history_matching','error_type':type(exc).__name__})
    for r in rows:
        m=matched.get(r['index'],{})
        r['state']=m.get('state','history_lookup_failed' if warnings else 'needs_local_analysis')
        r['source_analysis_id']=m.get('source_analysis_id')
    data={'batch_sha256':batch,**args,'record_count':len(rows),'rows':rows,'query_status':'completed','explanation_status':'not_requested','warnings':warnings}
    eid=aid+':local-query:'+hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:16]
    obs=[{'evidence_id':eid,'path':'/record_count','value':len(rows)}]
    for i,r in enumerate(rows):
        for k in ('index','label','prediction','score','threshold'):obs.append({'evidence_id':eid,'path':f'/rows/{i}/{k}','value':r[k]})
    answer=f"本地查询完成：{args['category'].upper()} 共 {len(rows)} 条，按模型分数{'降序' if args['order']=='desc' else '升序'}排列。完整结果见表格。这是程序查询结果，不是LLM回答。"
    return {'analysis_id':aid,'data_kind':'real_ecg','requires_review':True,'status':'completed_draft','error':'',
        'query_status':'completed','explanation_status':'not_requested','answer_source':'deterministic_local_query',
        'local_query':data,'draft':{'answer':answer,'evidence_ids':[eid],'knowledge_ids':[],'observations':obs},
        'validation':{'passed':True,'scope':'deterministic_report_filter_and_sort; no_llm_semantic_claim'},
        'evidence':{eid:{'analysis_id':aid,'evidence_id':eid,'ok':True,'scope':'explicit_batch_report','data':data}},
        'knowledge':{},'trace':[{'stage':'local_query','tool':'query_batch','arguments':args,'ok':True,'evidence_id':eid}],
        'model_calls':0,'tool_calls':0,'local_query_calls':1,'elapsed_seconds':time.perf_counter()-start}
