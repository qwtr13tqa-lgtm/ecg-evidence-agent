"""Finite report-bound relational operations. No eval, SQL, inference or network."""
import hashlib
import json
import math
import re
from decimal import Decimal
from copy import deepcopy
from src.review.batch import validate

NUMERIC=('index','label','prediction','threshold','score','score_margin','reconstruction_error','shape_error','region_count','heart_rate_bpm','mean_rr_seconds','rr_cv')
TEXT=('category','top_lead','history_status','measurement_status')
FIELDS=NUMERIC+TEXT
BASE=('index','label','prediction','threshold','score','score_margin','category')
OPS=('eq','ne','lt','le','gt','ge','in','is_missing','not_missing')

def finite(v):return type(v) in (int,float) and math.isfinite(v)

def base_rows(report):
    validate(report)
    if len(report['rows'])>500:raise ValueError('BATCH_LIMIT_500')
    categories={(0,0):'tn',(0,1):'fp',(1,0):'fn',(1,1):'tp'}
    return [{**{f:None for f in FIELDS},**{k:r[k] for k in ('index','label','prediction','score')},
        'threshold':report['threshold'],'score_margin':r['score']-report['threshold'],
        'category':categories[r['label'],r['prediction']],'source_analysis_id':None,'history_status':'not_loaded'} for r in report['rows']]

def enrich(report,history):
    from src.review.batch_statistics import collect
    stats=collect(report,history,['fn','tp','fp','tn'])
    rows=[]
    for original in stats['rows']:
        r=deepcopy(original)
        tops=[e.get('lead') for e in (r.get('lead_evidence') or []) if isinstance(e,dict) and e.get('rank')==1 and isinstance(e.get('lead'),str)]
        r['top_lead']=tops[0] if len(tops)==1 else None
        r['source_analysis_id']=r.get('analysis_id')
        rows.append(r)
    return rows,stats

class CollectionValidationError(ValueError):
    def __init__(self,code,path,field=None,operator=None,value=None):
        super().__init__(code)
        self.code=code
        self.details={'requested_path':path,'field':field,'operator':operator,
                      'received_type':type(value).__name__,'received_value':str(value)[:160]}


def normalize_operation(operation,path='/operation'):
    """Only lossless-in-intent numeric string representation conversion; never invent filters."""
    op=deepcopy(operation);changes=[]
    if not isinstance(op,dict) or not isinstance(op.get('filters'),list):return op,changes
    for i,f in enumerate(op['filters']):
        if not isinstance(f,dict) or f.get('field') not in NUMERIC or f.get('op') in ('is_missing','not_missing'):continue
        values=f.get('value') if f.get('op')=='in' else [f.get('value')]
        if not isinstance(values,list):continue
        converted=list(values)
        for j,v in enumerate(values):
            if not isinstance(v,str) or len(v)>64:continue
            text=v.strip()
            if not re.fullmatch(r'[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?',text):continue
            try:
                number=int(text) if not any(c in text for c in '.eE') else float(text)
                if not finite(number) or (number==0 and Decimal(text)!=0):continue
            except (ValueError,OverflowError):continue
            converted[j]=number
            changes.append({'path':path+f'/filters/{i}/value'+(f'/{j}' if f.get('op')=='in' else ''),'before':v,'after':number,
                            'reason':'numeric_string_representation'})
        if f.get('op')=='in':f['value']=converted
        elif converted:f['value']=converted[0]
    return op,changes


def validate_operation(op):
    if not isinstance(op,dict) or set(op)!={'kind','filters','order_by','direction','limit','offset','columns','group_by','metrics'}:raise ValueError('COLLECTION_SCHEMA')
    if op['kind'] not in ('select','aggregate'):raise ValueError('COLLECTION_KIND')
    if not isinstance(op['filters'],list) or len(op['filters'])>12:raise ValueError('FILTER_LIMIT')
    for i,f in enumerate(op['filters']):
        if not isinstance(f,dict) or set(f)!={'field','op','value'} or f['field'] not in FIELDS or f['op'] not in OPS:raise ValueError('FILTER_SCHEMA')
        field,operator,value=f['field'],f['op'],f['value']
        if operator in ('is_missing','not_missing'):
            if value is not None:raise ValueError('MISSING_OPERATOR_VALUE')
            continue
        if operator in ('lt','le','gt','ge') and field not in NUMERIC:raise ValueError('NUMERIC_OPERATOR_REQUIRED')
        values=value if operator=='in' else [value]
        if not isinstance(values,list) or not values or len(values)>500:raise ValueError('FILTER_VALUE_LIST')
        for v in values:
            if field in NUMERIC and not finite(v):raise CollectionValidationError('FINITE_NUMERIC_REQUIRED',f'/filters/{i}/value',field,operator,v)
            if field in TEXT and (not isinstance(v,str) or not v or len(v)>80):raise ValueError('TEXT_REQUIRED')
            if field=='category' and v not in ('fn','tp','fp','tn'):raise ValueError('CATEGORY_INVALID')
    for key in ('columns','metrics'):
        if not isinstance(op[key],list) or any(not isinstance(x,str) or x not in (FIELDS if key=='columns' else NUMERIC) for x in op[key]) or len(set(op[key]))!=len(op[key]):raise ValueError('FIELD_LIST')
    if len(op['columns'])>16 or len(op['metrics'])>8:raise ValueError('FIELD_LIMIT')
    if op['direction'] not in ('asc','desc') or op['order_by'] not in (*FIELDS,None):raise ValueError('SORT_INVALID')
    if op['group_by'] not in (*TEXT,None):raise ValueError('GROUP_INVALID')
    if type(op['limit']) is not int or not 1<=op['limit']<=500 or type(op['offset']) is not int or not 0<=op['offset']<=500:raise ValueError('PAGE_INVALID')
    if op['kind']=='select' and (op['metrics'] or op['group_by'] is not None):raise ValueError('SELECT_UNUSED_ARGUMENTS')
    if op['kind']=='aggregate' and (op['order_by'] is not None or op['columns']):raise ValueError('AGGREGATE_UNUSED_ARGUMENTS')
    return op

def needs_history(op):
    fields=[f['field'] for f in op['filters']]+op['columns']+op['metrics']+[op['order_by'],op['group_by']]
    return any(f is not None and f not in BASE for f in fields)

def match(row,f):
    v=row.get(f['field']);operator=f['op'];target=f['value']
    missing=v is None or (f['field'] in NUMERIC and not finite(v))
    if operator=='is_missing':return missing
    if operator=='not_missing':return not missing
    if missing:return False  # Missing values never turn into zero or a negative predicate match.
    if operator=='in':return v in target
    return {'eq':lambda:v==target,'ne':lambda:v!=target,'lt':lambda:v<target,'le':lambda:v<=target,'gt':lambda:v>target,'ge':lambda:v>=target}[operator]()

def quantile(values,p):
    if not values:return None
    x=sorted(values);pos=(len(x)-1)*p;i=int(pos);j=min(i+1,len(x)-1)
    return x[i]+(x[j]-x[i])*(pos-i)

def execute(rows,operation):
    op=validate_operation(operation)
    selected=[r for r in rows if all(match(r,f) for f in op['filters'])]
    missing_filters={f['field']:sum(r.get(f['field']) is None for r in rows) for f in op['filters']}
    data={'operation':deepcopy(op),'input_n':len(rows),'matched_n':len(selected),'filter_missing_n':missing_filters,
          'offset':op['offset'],'limit':op['limit'],'missing_policy':'Excluded from comparisons; kept as missing group; never zero',
          'scope':'Descriptive only; selection is not causal evidence'}
    if op['kind']=='select':
        ordered=sorted(selected,key=lambda r:r['index']);field=op['order_by']
        if field:
            available=[r for r in ordered if r.get(field) is not None]
            missing=[r for r in ordered if r.get(field) is None]
            ordered=sorted(available,key=lambda r:r[field],reverse=op['direction']=='desc')+missing
        columns=list(dict.fromkeys(['index','category']+op['columns']))
        data['rows']=[{**{k:r.get(k) for k in columns},'source_analysis_id':r.get('source_analysis_id'),'history_status':r.get('history_status','not_loaded')} for r in ordered[op['offset']:op['offset']+op['limit']]]
        data['returned_n']=len(data['rows']);data['has_more']=op['offset']+data['returned_n']<len(ordered)
    else:
        key=op['group_by'];groups={}
        for r in selected:groups.setdefault(r.get(key) if key else 'all',[]).append(r)
        if not key and not groups:groups={'all':[]}
        result=[]
        for value,group in sorted(groups.items(),key=lambda pair:(pair[0] is None,str(pair[0]))):
            item={'group_value':value,'count':len(group),'fraction':len(group)/len(selected) if selected else None,'metrics':{}}
            for field in op['metrics']:
                vals=[r[field] for r in group if finite(r.get(field))]
                item['metrics'][field]={'valid_n':len(vals),'missing_n':len(group)-len(vals),
                    'mean':math.fsum(vals)/len(vals) if vals else None,'median':quantile(vals,.5),'q1':quantile(vals,.25),'q3':quantile(vals,.75),
                    'minimum':min(vals) if vals else None,'maximum':max(vals) if vals else None}
            result.append(item)
        data['group_total_n']=len(result);data['groups']=result[op['offset']:op['offset']+op['limit']]
        data['returned_n']=len(data['groups']);data['has_more']=op['offset']+data['returned_n']<len(result)
        data['quantile_method']='linear interpolation over finite values'
    return data

def evidence(aid,batch,data):
    from src.agent.answer_validator import scalar_paths
    data={**data,'batch_sha256':batch}
    eid=aid+':collection:'+hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    return {'analysis_id':aid,'evidence_id':eid,'ok':True,'scope':'batch_collection','data':data,'observation_paths':scalar_paths(data,limit=400)}

# Same schema for task planning and iterative collection queries.
def filter_branch(fields,operators,value_schema):
    return {'type':'object','additionalProperties':False,'properties':{
        'field':{'type':'string','enum':list(fields)},'op':{'type':'string','enum':list(operators)},'value':value_schema},
        'required':['field','op','value']}

FILTER_SCHEMA={'anyOf':[
    filter_branch(NUMERIC,('eq','ne','lt','le','gt','ge'),{'type':'number'}),
    filter_branch(NUMERIC,('in',),{'type':'array','minItems':1,'maxItems':500,'items':{'type':'number'}}),
    filter_branch(TEXT,('eq','ne'),{'type':'string','minLength':1,'maxLength':80}),
    filter_branch(TEXT,('in',),{'type':'array','minItems':1,'maxItems':500,'items':{'type':'string','minLength':1,'maxLength':80}}),
    filter_branch(FIELDS,('is_missing','not_missing'),{'type':'null'})]}
OP_SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'kind':{'type':'string','enum':['select','aggregate']},
 'filters':{'type':'array','maxItems':12,'items':FILTER_SCHEMA},
 'order_by':{'anyOf':[{'type':'string','enum':list(FIELDS)},{'type':'null'}]},'direction':{'type':'string','enum':['asc','desc']},
 'limit':{'type':'integer','minimum':1,'maximum':500},'offset':{'type':'integer','minimum':0,'maximum':500},
 'columns':{'type':'array','items':{'type':'string','enum':list(FIELDS)},'maxItems':16},
 'group_by':{'anyOf':[{'type':'string','enum':list(TEXT)},{'type':'null'}]},
 'metrics':{'type':'array','items':{'type':'string','enum':list(NUMERIC)},'maxItems':8}},
 'required':['kind','filters','order_by','direction','limit','offset','columns','group_by','metrics']}
TOOL={'type':'function','function':{'name':'query_batch_collection','description':'对当前批次统一记录表筛选、稳定排序、分页或分组统计。filters为AND；in可选多类别。缺失不当零。aggregate按group_by分组输出各metrics的有效数/缺失/均值/分位数和范围。select不读取波形；有has_more请勿声称返回全量。参数不支持的条件必须澄清。','parameters':OP_SCHEMA}}
