"""Lossless evidence-data projection and strict local judge reference mapping."""
from copy import deepcopy


def judge_payload(record):
    out=record['output'];evidence=out.get('evidence',{})
    mapping={'E'+str(i+1):eid for i,eid in enumerate(evidence)}
    reverse={eid:short for short,eid in mapping.items()}
    def replace(value):
        if isinstance(value,dict):return {reverse.get(k,k):replace(v) for k,v in value.items()}
        if isinstance(value,list):return [replace(v) for v in value]
        if isinstance(value,str):
            for eid,short in sorted(reverse.items(),key=lambda x:-len(x[0])):value=value.replace(eid,short)
        return value
    # Evidence data is retained in full; drop only redundant path catalogs and
    # duplicate response/plan dumps from execution trace. Never sample rows.
    ev={eid:{k:v for k,v in e.items() if k!='observation_paths'} for eid,e in evidence.items()}
    trace=[{k:v for k,v in t.items() if k not in ('response','payload','plan','messages','reply')}
           for t in out.get('trace',[])]
    payload={'question':record['case']['question'],'answer':out.get('draft'),
             'evidence':ev,'oracle':record['reference_workflow'],'execution_trace':trace}
    return replace(payload),mapping


def map_review(review,mapping):
    if not isinstance(review,dict) or not isinstance(review.get('claims'),list):raise ValueError('JUDGE_SCHEMA')
    result=deepcopy(review)
    for c in result.get('claims',[]):
        if not isinstance(c,dict):raise ValueError('JUDGE_CLAIM_SCHEMA')
        ids=c.get('evidence_ids')
        if not isinstance(ids,list) or any(not isinstance(e,str) or e not in mapping for e in ids):
            raise ValueError('JUDGE_UNKNOWN_SHORT_REFERENCE')
        c['evidence_ids']=[mapping[e] for e in ids]
    return result


def reliability_metrics(runs,planned):
    def repaired(r):
        return any(t.get('stage') in ('planning_recovery','plan_repair','dependency_repair','validation_repair','timeout_repair','truncation_repair','format_repair','submission_repair')
                   for t in r['output'].get('trace',[]))
    passed=lambda r:bool(r['automatic']['automatic_acceptance'])
    first=sum(passed(r) and not repaired(r) for r in runs)
    recovered=sum(passed(r) and repaired(r) for r in runs)
    return {'first_pass_without_recovery':first,'recovered_pass':recovered,
        'recovery_attempted_runs':sum(repaired(r) for r in runs),
        'first_pass_rate':first/planned if planned else None,
        'eventual_pass_rate':(first+recovered)/planned if planned else None,
        'total_seconds':sum(r['seconds'] for r in runs),
        'total_model_requests':sum(len(r['requests']) for r in runs),
        'scope':'All planned runs denominator; missing runs not successes; first pass excludes any repair/recovery.'}
