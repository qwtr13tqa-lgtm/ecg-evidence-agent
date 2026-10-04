"""Typed backward references for collection plans; never guess a missing scalar."""
from copy import deepcopy
import re
from src.review.collection_ops import OP_SCHEMA, finite, validate_operation

REF_SCHEMA={'type':'object','additionalProperties':False,'properties':{'from_step':{'type':'integer','minimum':0,'maximum':2},'path':{'type':'string','pattern':r'^/groups/[0-9]+/metrics/[a-z_]+/(mean|median|q1|q3|minimum|maximum)$'}},'required':['from_step','path']}
PATH=re.compile(r'^/groups/(0|[1-9][0-9]*)/metrics/([a-z_]+)/(mean|median|q1|q3|minimum|maximum)$')

def planning_schema():
    s=deepcopy(OP_SCHEMA)
    branches=s['properties']['filters']['items']['anyOf']
    branches[0]['properties']['value']={'anyOf':[{'type':'number'},REF_SCHEMA]}
    return s

def reference_filters(op):
    return [(j,f) for j,f in enumerate(op.get('filters',[])) if isinstance(f,dict) and isinstance(f.get('value'),dict)]

def validate_bound_operation(op,steps,index):
    candidate=deepcopy(op)
    for j,f in reference_filters(op):
        ref=f['value']
        if set(ref)!={'from_step','path'} or type(ref['from_step']) is not int or not 0<=ref['from_step']<index:raise ValueError('DEPENDENCY_NOT_PREVIOUS_STEP')
        if not isinstance(ref['path'],str) or not PATH.fullmatch(ref['path']):raise ValueError('DEPENDENCY_PATH_NOT_AGGREGATE_SCALAR')
        source=steps[ref['from_step']];field=PATH.fullmatch(ref['path'])[2]
        if source.get('group_by') is None and PATH.fullmatch(ref['path'])[1]!='0':raise ValueError('DEPENDENCY_GROUP_INDEX_INVALID')
        if source.get('kind')!='aggregate' or field not in source.get('metrics',[]) or f.get('field')!=field:raise ValueError('DEPENDENCY_METRIC_MISMATCH')
        if f.get('op') not in ('eq','ne','lt','le','gt','ge'):raise ValueError('DEPENDENCY_OPERATOR')
        candidate['filters'][j]['value']=0 # Schema-only placeholder, never executed or returned as evidence.
    validate_operation(candidate)

def resolve_operation(op,results):
    resolved=deepcopy(op);bindings=[]
    for j,f in reference_filters(op):
        ref=f['value'];n=ref['from_step']
        if not 0<=n<len(results):raise ValueError('DEPENDENCY_RESULT_UNAVAILABLE')
        source=results[n];v=source['data']
        try:
            for token in ref['path'][1:].split('/'):v=v[int(token)] if isinstance(v,list) else v[token]
        except (KeyError,IndexError,TypeError,ValueError):raise ValueError('DEPENDENCY_PATH_UNAVAILABLE')
        if not finite(v):raise ValueError('DEPENDENCY_NO_VALID_VALUE')
        resolved['filters'][j]['value']=v
        bindings.append({'target_path':f'/filters/{j}/value','source_step':n,'source_evidence_id':source['evidence_id'],'source_path':ref['path'],'resolved_value':v})
    validate_operation(resolved)
    return resolved,bindings

FIELDS=r'(?:score|分数|阈值|重构|形状|区域数量|region_count|rr_cv|心率|shape_error|reconstruction_error)'
def ambiguity_reason(question):
    # Conservative lexical gate for this supported intent, not a universal intent validator.
    if re.search(r'最值得关注|最重要|漏报最严重',question) and not re.search(r'(?:按|以|根据|依据|优先).*?'+FIELDS,question,re.I):
        return '“值得关注”的排序标准尚未指定。请说明按模型分数、与阈值的距离、区域数量，还是其他指标排序？'
    return None

def dependency_requested(question):
    return bool(re.search(r'中位数|median|均值|平均值|mean|分位数|最大值|最小值',question,re.I) and re.search(r'高于|大于|低于|小于|超过|为界|作为.{0,10}(?:阈值|界限)|greater than|less than',question,re.I))

def guard_plan(question,plan):
    reason=ambiguity_reason(question)
    if reason:return {'mode':'clarify','operations':[],'clarification':reason,'assumptions':[]},'AMBIGUOUS_PRIORITY'
    if plan['mode']=='clarify':return plan,None
    if dependency_requested(question):
        refs=[f for op in plan['operations'] for _,f in reference_filters(op)]
        if not refs:raise ValueError('DEPENDENCY_BINDING_REQUIRED')
        # An additional literal filter on a dynamically bounded metric can silently change the task.
        fields={f['field'] for f in refs}
        for op in plan['operations']:
            for f in op['filters']:
                if f['field'] in fields and f['op'] not in ('is_missing','not_missing') and not isinstance(f['value'],dict):raise ValueError('DEPENDENCY_LITERAL_OVERRIDE')
    return plan,None
