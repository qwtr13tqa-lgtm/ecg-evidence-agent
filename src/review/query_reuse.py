"""Conservative reuse: same batch, filters, ordering and fully covered page/fields."""
from src.review.collection_ops import validate_operation

def find_reusable(evidence, operation, aid, batch):
    op=validate_operation(operation)
    for eid,e in evidence.items():
        d=e.get('data',{});old=d.get('operation',{})
        if e.get('analysis_id')!=aid or d.get('batch_sha256')!=batch or e.get('scope')!='batch_collection' or e.get('ok') is not True:continue
        if any(old.get(k)!=op[k] for k in ('kind','filters','group_by','order_by','direction')):continue
        field='columns' if op['kind']=='select' else 'metrics'
        if not set(op[field]).issubset(old.get(field,[])):continue
        items=d.get('rows' if op['kind']=='select' else 'groups')
        if not isinstance(items,list):continue
        start=op['offset']-old.get('offset',0)
        end=start+op['limit']
        if start<0 or start>len(items):continue
        if end>len(items) and d.get('has_more') is not False:continue
        return eid,list(range(start,min(end,len(items))))
    return None

def is_timeout(exc):
    return isinstance(exc,TimeoutError) or type(exc).__name__ in ('APITimeoutError','ReadTimeout','ConnectTimeout','WriteTimeout','PoolTimeout')
