"""Static grammar validation only; result-dependent types/ranges remain runtime checks."""
import json
import math
import re
ARITY={'if':3,'ge':2,'eq':2,'sub':2,'mul':2,'len':1,'get':2,'argmax':2,'filter_eq':3}
TOOLS={'get_analysis_summary':(set(),set()),'inspect_recent_error':({'duration_seconds','lead'},{'duration_seconds'}),'inspect_error_window':({'start_sample','end_sample','lead'},{'start_sample','end_sample'}),'inspect_rr_intervals':({'offset','limit'},set())}

class PlanValidationError(ValueError):
    code='PLAN_STATIC_VALIDATION_FAILED'
    def __init__(self,errors):
        super().__init__(self.code);self.details={'plan_errors':errors}

def validate_plan(plan):
    errors=[]
    def err(path,code): errors.append({'path':path,'code':code})
    try:
        if len(json.dumps(plan,allow_nan=False))>16000:err('','PLAN_TOO_LARGE')
    except (ValueError,TypeError,RecursionError):err('','INVALID_JSON')
    if not isinstance(plan,dict) or set(plan)!={'steps'}:err('','PLAN_SCHEMA');return errors
    steps=plan['steps']
    if not isinstance(steps,list) or not 1<=len(steps)<=12:err('/steps','STEP_COUNT');return errors
    known={'summary'}
    def expr(x,path,depth=0):
        if depth>16:err(path,'EXPRESSION_DEPTH');return
        if isinstance(x,list):
            for i,v in enumerate(x):expr(v,path+'/'+str(i),depth+1)
        elif isinstance(x,dict):
            if 'tool' in x:err(path,'NESTED_TOOL_NOT_ALLOWED')
            if 'ref' in x:
                if set(x)!={'ref','path'}:err(path,'REFERENCE_SCHEMA')
                if not isinstance(x['ref'],str) or x['ref'] not in known:err(path+'/ref','UNKNOWN_OR_FORWARD_REFERENCE')
                p=x.get('path')
                if not isinstance(p,str) or (p!='' and not p.startswith('/')) or (isinstance(p,str) and re.search(r'~(?![01])',p)):err(path+'/path','INVALID_POINTER')
            elif 'op' in x:
                if set(x)!={'op','args'}:err(path,'OP_SCHEMA')
                op=x['op'];args=x.get('args')
                if not isinstance(op,str) or op not in ARITY:err(path+'/op','UNKNOWN_OP')
                elif not isinstance(args,list) or len(args)!=ARITY[op]:err(path+'/args','OP_ARITY')
                if isinstance(args,list):
                    for i,v in enumerate(args):expr(v,path+'/args/'+str(i),depth+1)
            else:
                for k,v in x.items():expr(v,path+'/'+k,depth+1)
        elif x is not None and type(x) not in (str,bool,int,float):err(path,'INVALID_LITERAL')
        elif type(x) is float and not math.isfinite(x):err(path,'NONFINITE')
    for i,s in enumerate(steps):
        p='/steps/'+str(i)
        if not isinstance(s,dict):err(p,'STEP_SCHEMA');continue
        sid=s.get('id')
        validid=isinstance(sid,str) and re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,31}',sid) and sid not in known
        if not validid:err(p+'/id','INVALID_OR_DUPLICATE_ID')
        if 'tool' in s:
            if not {'id','tool','args'}<=set(s) or set(s)-{'id','tool','args','when'}:err(p,'TOOL_STEP_SCHEMA')
            name=s['tool'];args=s.get('args')
            if not isinstance(name,str) or name not in TOOLS:err(p+'/tool','UNKNOWN_TOOL')
            elif not isinstance(args,dict):err(p+'/args','ARGUMENT_OBJECT_REQUIRED')
            else:
                allowed,required=TOOLS[name]
                for k in sorted(set(args)-allowed):err(p+'/args/'+k,'UNEXPECTED_ARGUMENT')
                for k in sorted(required-set(args)):err(p+'/args/'+k,'MISSING_ARGUMENT')
            if isinstance(args,dict):
                for k,v in args.items():expr(v,p+'/args/'+k)
            if 'when' in s:
                expr(s['when'],p+'/when')
                if not isinstance(s['when'],dict) and type(s['when']) is not bool:err(p+'/when','BOOLEAN_CONDITION_REQUIRED')
        else:
            if set(s)!={'id','value'}:err(p,'VALUE_STEP_SCHEMA')
            if 'value' in s:expr(s['value'],p+'/value')
        if validid:known.add(sid)
    return errors

def require_valid_plan(plan):
    errors=validate_plan(plan)
    if errors:raise PlanValidationError(errors)
