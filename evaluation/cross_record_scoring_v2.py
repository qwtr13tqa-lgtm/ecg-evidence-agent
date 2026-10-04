"""Normalize proven equivalent category scopes for grading only, never mutate evidence."""
from copy import deepcopy

CATEGORIES={'fn':(1,0),'tp':(1,1),'fp':(0,1),'tn':(0,0)}
def categories(filters):
    possible=set(CATEGORIES)
    for f in filters:
        field=f.get('field')
        if field not in ('category','label','prediction'):continue
        if f.get('op') not in ('eq','in'):return set()
        vals=f.get('value') if f['op']=='in' else [f.get('value')]
        if not isinstance(vals,list):return set()
        possible={c for c in possible if (c if field=='category' else CATEGORIES[c][0 if field=='label' else 1]) in vals}
    return possible

def normalized_output(output):
    out=deepcopy(output)
    for ev in out.get('evidence',{}).values():
        d=ev.get('data',{});op=d.get('operation',{});fs=op.get('filters',[])
        cat=categories(fs)
        if len(cat)!=1:continue
        category=next(iter(cat))
        if op.get('kind')=='select':
            # Preserve ALL metric filters; only annotate logically equivalent cohort selection.
            op['filters']=fs+[{'field':'category','op':'eq','value':category}]
        if op.get('kind')=='aggregate' and op.get('group_by') is None and not any(f.get('field') not in ('category','label','prediction') for f in fs):
            for g in d.get('groups',[]):
                if g.get('group_value')=='all':g['group_value']=category
    return out

def consistency(review):
    raw=deepcopy(review);effective=review.get('task_complete','uncertain');reasons=[]
    claims=review.get('claims',[])
    critical_bad=[c for c in claims if c.get('task_critical') is True and c.get('verdict')=='contradicted']
    critical_unknown=[c for c in claims if c.get('task_critical') is True and c.get('verdict')=='insufficient']
    if critical_bad:effective='no';reasons.append('CRITICAL_TASK_CONTRADICTION')
    elif critical_unknown and effective=='yes':effective='uncertain';reasons.append('CRITICAL_TASK_UNSUPPORTED')
    elif effective=='yes' and any(c.get('verdict')=='contradicted' for c in claims):effective='uncertain';reasons.append('CONTRADICTION_WITHOUT_CRITICALITY_REVIEW')
    return {'raw_review':raw,'effective_task_complete':effective,'consistency_errors':reasons,'human_verified':False}
