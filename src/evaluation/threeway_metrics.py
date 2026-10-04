"""Read-only result aggregation. No LLM judge, no pending-to-pass conversion."""
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCHEMES=('summary','rules','agent')
FIELDS=('task_correct','evidence_support','text_complete')


def outcome(record,review):
    out=record['output'];m=record.get('automatic',{}).get('metrics',{})
    if out.get('status')!='completed_draft' or m.get('structure_values_references') is False or m.get('analysis_binding') is False or m.get('task_observations') is False:
        return 'fail'
    if any(review.get(k)=='fail' for k in FIELDS):return 'fail'
    if all(review.get(k)=='pass' for k in FIELDS):return 'pass'
    return 'pending'


def quantile(values,p):
    a=sorted(values)
    if not a:return None
    x=(len(a)-1)*p;lo=int(x);hi=math.ceil(x)
    return a[lo]+(a[hi]-a[lo])*(x-lo)


def load_batch(root,batch):
    root=Path(root)
    manifest=json.loads((root/(batch+'.manifest.json')).read_text(encoding='utf-8'))
    if manifest['config'].get('protocol')!='threeway-1.0':raise ValueError('Not a threeway batch')
    rows=[];warnings=[];seen=set()
    from src.evaluation.records import read_review
    for path in sorted(root.glob('*/result.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        c=r.get('config',{})
        if c.get('batch_id')!=batch:continue
        key=(c['case_id'],c['scheme'],c['repetition'],r['case']['turn_index'])
        if key in seen:raise ValueError('Duplicate comparison key')
        seen.add(key)
        try:review=read_review(path.parent).get('scores',{})
        except ValueError as exc:review={};warnings.append(f'{r["run_id"]}: invalid review: {exc}')
        o=r['output'];auto=r.get('automatic',{});m=auto.get('metrics',{});trace=o.get('trace',[])
        calls=[t for t in trace if t.get('stage')=='tool' and t.get('source')!='bootstrap']
        identities=set();repeats=0
        for t in calls:
            ident=json.dumps([t.get('tool'),t.get('arguments')],sort_keys=True)
            repeats+=ident in identities;identities.add(ident)
        measurements=auto.get('request_measurements',[])
        errors=[str(t.get('error_type','')) for t in trace+measurements]
        latency=r.get('wall_seconds',o.get('elapsed_seconds'))
        rows.append(dict(run_id=r['run_id'],case=c['case_id'],scheme=c['scheme'],repetition=c['repetition'],
            turn=r['case']['turn_index'],kind=r['case']['expectation']['kind'],outcome=outcome(r,review),
            completed=o.get('status')=='completed_draft',structure=m.get('structure_values_references') is True,
            task_fields=m.get('task_observations'),timeout=any('Timeout' in e for e in errors),
            query_calls=len(calls),bootstrap_calls=1 if o.get('evidence') else 0,
            invalid_calls=sum(t.get('ok') is False for t in calls),repeat_calls=repeats,
            model_calls=len(measurements),latency=latency,review=review,record=r,path=str(path)))
    expected=sum(t['turn_count'] for t in manifest['tasks'])
    if len(rows)!=expected:warnings.append(f'Incomplete batch: {len(rows)}/{expected} turns; do not present as a complete comparison.')
    return manifest,rows,warnings


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['scheme']].append(row)
    result=[]
    for scheme in SCHEMES:
        g=groups.get(scheme,[]);n=len(g)
        if not n:continue
        counts={k:sum(r['outcome']==k for r in g) for k in ('pass','fail','pending')}
        times=[r['latency'] for r in g if isinstance(r['latency'],(int,float)) and math.isfinite(r['latency'])]
        fields=[r['task_fields'] for r in g if r['task_fields'] is not None]
        result.append(dict(scheme=scheme,runs=n,**counts,
            confirmed_success_rate=counts['pass']/n, possible_success_rate=(counts['pass']+counts['pending'])/n,
            structure_rate=sum(r['structure'] for r in g)/n,
            task_field_rate=sum(fields)/len(fields) if fields else None,task_field_n=len(fields),
            timeout_rate=sum(r['timeout'] for r in g)/n,
            median_seconds=statistics.median(times) if times else None,p95_seconds=quantile(times,.95),
            mean_query_calls=sum(r['query_calls'] for r in g)/n,
            mean_model_calls=sum(r['model_calls'] for r in g)/n,
            invalid_calls=sum(r['invalid_calls'] for r in g),repeat_calls=sum(r['repeat_calls'] for r in g)))
    return result


def conversation_outcomes(manifest,rows):
    result=[]
    for task in manifest['tasks']:
        g=[r for r in rows if (r['case'],r['scheme'],r['repetition'])==(task['case_id'],task['scheme'],task['repetition'])]
        state='fail' if any(r['outcome']=='fail' for r in g) else ('pass' if len(g)==task['turn_count'] and all(r['outcome']=='pass' for r in g) else 'pending')
        result.append({**task,'outcome':state})
    return result


def save_review(path,reviewer,scores,notes):
    from src.evaluation.records import atomic_json
    path=Path(path)
    if not reviewer.strip():raise ValueError('Reviewer required')
    if set(scores)!=set(FIELDS) or any(v not in (None,'pass','fail') for v in scores.values()):raise ValueError('Invalid review')
    r=json.loads(path.read_text(encoding='utf-8'))
    # Append an audit copy before replacing editable review.json; result.json never changes.
    from datetime import datetime, timezone
    old=path.parent/'review.json'
    if old.exists():
        history=path.parent/'review_history';history.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        (history/(stamp+'.json')).write_bytes(old.read_bytes())
    atomic_json(old,dict(run_id=r['run_id'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reviewer=reviewer,notes=notes,**scores))
