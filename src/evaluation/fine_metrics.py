"""Offline field coverage and auditable, non-overlapping tool classification."""
import hashlib
import json
from pathlib import Path

VERDICTS=('pending','necessary','redundant','failed')
CLAIM_VERDICTS=('pending','supported','unsupported')


def calls_for(record):
    out=record['output'];scheme=record['config']['scheme'];kind=record['case']['expectation']['kind']
    seen=set();items=[]
    for index,event in enumerate(out.get('trace',[])):
        if event.get('stage')!='tool' or event.get('source')=='bootstrap':continue
        name=event.get('tool');args=event.get('arguments');eid=event.get('evidence_id')
        identity=json.dumps([name,args],sort_keys=True,ensure_ascii=False)
        reason='needs_review';verdict='pending'
        if not event.get('ok'):
            reason='execution_failed';verdict='failed'
        elif identity in seen:
            reason='same_arguments_repeated';verdict='redundant'
        elif scheme=='agent' and name=='get_analysis_summary' and args=={}:
            # threeway-1.0 Agent always receives a bootstrap snapshot of this immutable analysis.
            evidence=out.get('evidence',{}).get(eid,{})
            if record['config'].get('protocol')=='threeway-1.0' and 'input' in evidence.get('data',{}):
                reason='bootstrap_summary_repeated';verdict='redundant'
        elif name=='get_model_decision' and kind in ('window','rr','missing'):
            reason='possibly_unrelated_to_requested_fields'
        elif name=='search_knowledge' and out.get('evidence',{}).get(eid,{}).get('data',{}).get('status')=='no_match':
            reason='no_match_not_automatically_redundant'
        elif name in ('inspect_recent_error','inspect_error_window','inspect_rr_intervals','inspect_recent_rr_alignment'):
            reason='check_arguments_and_task_relevance'
        seen.add(identity)
        items.append(dict(trace_index=index,tool=name,arguments=args,evidence_id=eid,
            elapsed_seconds=event.get('elapsed_seconds'),suggestion=verdict,reason=reason))
    return items


def analyze(row,review=None):
    r=row['record'];review=review or {};fields=r.get('automatic',{}).get('field_checks',{})
    calls=calls_for(r);decisions=review.get('calls',{})
    for c in calls:
        manual=decisions.get(str(c['trace_index']),{})
        c['verdict']=manual.get('verdict',c['suggestion'])
        c['note']=manual.get('note','')
        c['origin']='human' if manual else 'automatic' if c['suggestion']!='pending' else 'unreviewed'
    claims=review.get('claims',[])
    support=sum(c['verdict']=='supported' for c in claims)
    unsupported=sum(c['verdict']=='unsupported' for c in claims)
    pending=sum(c['verdict']=='pending' for c in claims)
    counts={v:sum(c['verdict']==v for c in calls) for v in VERDICTS}
    return dict(run_id=row['run_id'],scheme=row['scheme'],case=row['case'],turn=row['turn'],
        field_correct=sum(v is True for v in fields.values()),field_total=len(fields),
        field_coverage=sum(v is True for v in fields.values())/len(fields) if fields else None,
        query_requests=len(calls),**{v+'_calls':n for v,n in counts.items()},
        redundancy_lower=counts['redundant']/len(calls) if calls else None,
        redundancy_upper=(counts['redundant']+counts['pending'])/len(calls) if calls else None,
        supported_claims=support,unsupported_claims=unsupported,pending_claims=pending,
        claim_support_rate=support/(support+unsupported) if support+unsupported else None,
        claims_complete=bool(claims) and pending==0 and review.get('claims_complete') is True,
        model_requests=row['model_calls'],latency=row['latency'],input_tokens=None,output_tokens=None,
        calls=calls,claims=claims)


def aggregate(items):
    result=[]
    for scheme in ('summary','rules','agent'):
        g=[x for x in items if x['scheme']==scheme]
        if not g:continue
        total=sum(x['field_total'] for x in g);correct=sum(x['field_correct'] for x in g)
        n=sum(x['query_requests'] for x in g);redundant=sum(x['redundant_calls'] for x in g);pending=sum(x['pending_calls'] for x in g)
        result.append(dict(scheme=scheme,runs=len(g),field_correct=correct,field_total=total,
            field_coverage=correct/total if total else None,unscored_field_runs=sum(x['field_total']==0 for x in g),
            query_requests=n,redundant_calls=redundant,pending_calls=pending,
            failed_calls=sum(x['failed_calls'] for x in g),necessary_calls=sum(x['necessary_calls'] for x in g),
            redundancy_lower=redundant/n if n else None,redundancy_upper=(redundant+pending)/n if n else None,
            model_requests=sum(x['model_requests'] for x in g)))
    return result


def review_path(path):return Path(path).with_name('fine_review.json')


def read_review(path):
    p=review_path(path)
    if not p.exists():return {}
    value=json.loads(p.read_text(encoding='utf-8'))
    if value.get('result_sha256')!=hashlib.sha256(Path(path).read_bytes()).hexdigest():raise ValueError('Fine review/result hash mismatch')
    record=json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('run_id')!=record['run_id']:raise ValueError('Fine review/run mismatch')
    validate(record,value)
    return value


def validate(record,value):
    if not isinstance(value.get('reviewer'),str) or not value['reviewer'].strip():raise ValueError('Reviewer required')
    indices={str(c['trace_index']) for c in calls_for(record)}
    for k,v in value.get('calls',{}).items():
        if k not in indices or v.get('verdict') not in VERDICTS or not isinstance(v.get('note'),str) or not v['note'].strip():raise ValueError('Call decisions require valid trace index, verdict and reason')
    for c in value.get('claims',[]):
        if not isinstance(c.get('claim'),str) or not c['claim'].strip() or c.get('verdict') not in CLAIM_VERDICTS:raise ValueError('Each claim requires text and verdict')
        if c['verdict']!='pending' and not str(c.get('basis','')).strip():raise ValueError('Adjudicated claims require evidence path or reasoning')
    if type(value.get('claims_complete',False)) is not bool:raise ValueError('Invalid claims_complete')


def save_review(path,reviewer,calls,claims,claims_complete):
    from src.evaluation.records import atomic_json
    from datetime import datetime,timezone
    p=Path(path);r=json.loads(p.read_text(encoding='utf-8'))
    v=dict(version='fine-review-1.0',run_id=r['run_id'],result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        reviewer=reviewer,calls=calls,claims=claims,claims_complete=claims_complete)
    validate(r,v)
    target=review_path(p)
    if target.exists():
        d=p.parent/'fine_review_history';d.mkdir(exist_ok=True)
        (d/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.json')).write_bytes(target.read_bytes())
    atomic_json(target,v)
