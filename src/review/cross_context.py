"""Small transport projections; full original evidence remains in the audit record."""
from copy import deepcopy
from src.agent.answer_validator import scalar_paths


def model_evidence(evidence):
    e=deepcopy(evidence);data=e['data']
    if e.get('scope')=='explicit_batch_report':
        # Keep original list order and field values, so all surviving JSON pointers
        # resolve to the same original evidence. This is not row sampling.
        data['rows']=[{k:r.get(k) for k in ('index','category','history_status','top_lead')}
                      for r in data.get('rows',[])]
        data.pop('batch_sha256',None)
    if e.get('scope')=='batch_collection':
        # Preserve prefixes so displayed pointers still resolve in the full evidence.
        for key in ('rows','groups'):
            if len(data.get(key,[]))>40:
                data[key]=data[key][:40]
                e['transport_notice']='Only first 40 result rows shown to model; full result retained locally. Query with offset/limit for remaining records.'
        e['transport_paths']=scalar_paths(data,limit=160)
    # Remove repeated display metadata, retaining all statistical values and list positions.
    if e.get('scope')=='explicit_batch_report':
        for group in data.get('groups',[]):
            group.pop('source',None);group.pop('metric',None)
    # JSON pointers are derivable from data; repeating them consumes input without evidence.
    e.pop('observation_paths',None)
    for key in ('tool_version',):e.pop(key,None)
    return e


def recent_context(recent):
    # Prose is optional context, not evidence. Bound it separately from tool data.
    return [{'question':str(t.get('question',''))[:1000],
             'unverified_previous_answer':str(t.get('unverified_previous_answer',''))[:1500]}
            for t in (recent or [])[-3:]]
