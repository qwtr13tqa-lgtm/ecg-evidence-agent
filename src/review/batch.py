"""Read-only, report-bound batch queries. No global record access."""
import hashlib
import json
import math


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def validate(report):
    threshold=report['threshold']; rows=report['rows']
    if type(threshold) not in (int,float) or not math.isfinite(threshold):raise ValueError('Invalid threshold')
    if not isinstance(rows,list) or not rows:raise ValueError('Missing rows')
    ids=set()
    for r in rows:
        i=r['index'];s=r['score']
        if type(i) is not int or i<0 or i in ids:raise ValueError('Invalid/duplicate index')
        ids.add(i)
        if type(r['label']) is not int or r['label'] not in (0,1):raise ValueError('Invalid label')
        if type(s) not in (int,float) or not math.isfinite(s):raise ValueError('Invalid score')
        if type(r['prediction']) is not int or r['prediction']!=int(s>=threshold):raise ValueError('Prediction mismatch')
        if not isinstance(r.get('input_sha256'),str) or len(r['input_sha256'])!=64:raise ValueError('Missing input hash')
    return report


def query(report, category='fp', order='desc'):
    validate(report)
    if category not in ('fp','fn','tp','tn','all') or order not in ('asc','desc'):raise ValueError('Invalid query')
    classes={'fp':(0,1),'fn':(1,0),'tp':(1,1),'tn':(0,0)}
    rows=[dict(r,threshold=report['threshold']) for r in report['rows']
          if category=='all' or (r['label'],r['prediction'])==classes[category]]
    return sorted(rows,key=lambda r:((-r['score'] if order=='desc' else r['score']),r['index']))


def signal(report,row,path):
    import numpy as np
    if sha_file(path)!=report['data_sha256']:raise ValueError('数据文件哈希与报告不符，拒绝打开。')
    x=np.load(path,mmap_mode='r',allow_pickle=False)
    if x.ndim!=3 or x.shape[1:]!=(5000,12):raise ValueError('Expected N x 5000 x 12')
    raw=np.ascontiguousarray(x[row['index'],100:4900,:],dtype=np.float32)
    if hashlib.sha256(raw.tobytes()).hexdigest()!=row['input_sha256']:raise ValueError('片段哈希不符')
    if not np.isfinite(raw).all():raise ValueError('Nonfinite waveform')
    return raw

TOOLS=[{'type':'function','function':{'name':'query_batch','description':'查询所选评测批次。fp=误报(label0 prediction1), fn=漏报, tp=检出异常, tn=判对正常。仅支持分类筛选和按原始分数排序；不支持区域、RR或其他条件。',
 'parameters':{'type':'object','properties':{'category':{'type':'string','enum':['fp','fn','tp','tn','all']},'order':{'type':'string','enum':['asc','desc']}},'required':['category','order'],'additionalProperties':False}}}]


def ask(report,question,gateway):
    if not isinstance(question,str) or not 0<len(question.strip())<=2000:raise ValueError('Invalid question')
    reply=gateway.complete([
      {'role':'system','content':'将用户需求映射到所选批次查询工具，只调用一次query_batch。不支持的条件（如区域、RR、健康判断）必须说明不支持，不得丢弃条件后调用工具。不得访问其他批次。'},
      {'role':'user','content':question}],TOOLS)
    calls=reply.get('tool_calls',[])
    if not calls:return {'status':'unsupported','message':'本入口仅支持分类筛选和分数排序，请使用明确的筛选条件。'}
    if reply.get('finish_reason')!='tool_calls' or len(calls)!=1:raise ValueError('Invalid tool protocol')
    c=calls[0]
    if c.get('type')!='function' or c['function']['name']!='query_batch':raise ValueError('Invalid tool')
    args=json.loads(c['function']['arguments'])
    if not isinstance(args,dict) or set(args)!={'category','order'}:raise ValueError('Invalid arguments')
    rows=query(report,**args)
    return {'status':'completed','arguments':args,'rows':rows,'model_calls':1,'query_calls':1}
