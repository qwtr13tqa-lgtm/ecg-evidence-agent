"""Resumable single-process local analysis queue. No LLM access."""
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from src.review.batch import query,validate
from src.review.open_analysis import open_analysis,compatible
from src.evaluation.records import atomic_json

LOCK=threading.Lock()

def now():return datetime.now(timezone.utc).isoformat()

def plan(report,categories,root):
    validate(report)
    if not categories or not set(categories)<=set(('fn','tp','fp','tn')):raise ValueError('请选择有效类别')
    indices=sorted({r['index'] for c in categories for r in query(report,c)})
    identity={'report':report,'categories':sorted(set(categories)),'version':'local-batch-1'}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
    path=Path(root)/'evaluation/local_analysis_jobs'/key/'job.json'
    if path.exists():
        job=json.loads(path.read_text(encoding='utf-8'))
        if job['job_id']!=key or [r['index'] for r in job['rows']]!=indices:raise ValueError('任务记录与报告不一致')
        return path,job
    job={'job_id':key,'categories':sorted(set(categories)),'created_at':now(),'updated_at':now(),
         'rows':[{'index':i,'status':'pending','analysis_id':None,'attempts':0,'error':'','events':[]} for i in indices]}
    path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,job)
    return path,job


def advance(report,categories,data_path,root,history,pipeline_factory,limit=5,retry_failed=False,progress=None):
    if type(limit) is not int or not 1<=limit<=100:raise ValueError('本次处理条数必须为1..100')
    if not LOCK.acquire(blocking=False):raise ValueError('已有本地批量任务执行中，请等待完成')
    try:
        path,job=plan(report,categories,root)
        sources={r['index']:r for r in report['rows']}
        # A completion marker never substitutes for a readable, version-matched analysis.
        for item in job['rows']:
            if item['status'] in ('created','reused'):
                try:
                    result,_,_=history.load_analysis(item['analysis_id'])
                    if not compatible(result,report,sources[item['index']]):raise ValueError('历史版本不再匹配')
                except (ValueError,KeyError,OSError,TypeError):item.update(status='pending',error='历史缺失或不再匹配，需要重新处理')
            elif item['status']=='running':item.update(status='pending',error='上次执行中断，将检查历史后继续')
        atomic_json(path,job);processed=0
        for item in job['rows']:
            if processed>=limit:break
            if item['status'] not in ('pending','failed') or item['status']=='failed' and not retry_failed:continue
            item.update(status='running',attempts=item['attempts']+1,error='');job['updated_at']=now();atomic_json(path,job)
            if progress:progress(job)
            try:
                aid,reused=open_analysis(report,item['index'],data_path,Path(root),history,pipeline_factory)
                item.update(status='reused' if reused else 'created',analysis_id=aid,error='')
            except Exception as exc:
                item.update(status='failed',analysis_id=None,error=type(exc).__name__+': '+str(exc))
            item['events'].append({'at':now(),'status':item['status'],'analysis_id':item['analysis_id'],'error':item['error']})
            job['updated_at']=now();atomic_json(path,job);processed+=1
            if progress:progress(job)
        return path,job
    finally:LOCK.release()
