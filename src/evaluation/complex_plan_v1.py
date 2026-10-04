"""Restricted dataflow plan. No generated Python, eval, I/O or arbitrary analysis access."""
import json
import math
import re
import time
from copy import deepcopy
from src.evaluation.plan_validation_v12 import require_valid_plan

ALLOWED = {'get_analysis_summary', 'inspect_recent_error', 'inspect_error_window', 'inspect_rr_intervals'}
PROTOCOL_VERSION = 'complex-plan-protocol-1.2'
DSL = '''You are in PLAN SUBMISSION phase, not ECG tool execution.
The ONLY callable function now is submit_plan. ECG tool schemas below are documentation
for steps INSIDE plan_json, not functions you may call now. Do not call ECG tools or
submit_answer in this phase. Submit the complete data-dependent plan in one call.
Call submit_plan with plan_json containing JSON {"steps":[...]}, at most 12 steps.
Whole-plan static validation happens before any planned query. Every ref MUST have path;
use path:"" for the whole variable. No extra keys such as args2. get takes exactly
[object,key]; a ref with a scalar path already returns that scalar and needs no get.
Tool calls MUST be top-level steps, never inside value, if or another expression.
A tool step may add "when": EXPR. It must evaluate to boolean. False skips the tool
without evaluating its args, stores null in that variable, and creates no evidence.
Use lazy if to guard downstream reads of skipped results. when is not allowed on value steps.
Generic syntax examples (not a task solution):
{"id":"count","value":{"op":"len","args":[{"ref":"rows","path":""}]}}
{"id":"item","value":{"op":"get","args":[{"ref":"object","path":""},"key"]}}
{"id":"check","tool":"get_analysis_summary","args":{},"when":{"ref":"enabled","path":""}}
Examples assume rows/object/enabled are earlier variables; do not invent their values.
Each step has unique id (letters/digits/underscore, not summary) and either
{"id":"x","tool":"inspect_error_window","args":{"lead":"V1",...}} or
{"id":"x","value":EXPR}. Tool results are their data objects, not envelopes.
Variable summary is the current get_analysis_summary data. References only to earlier steps:
{"ref":"x","path":"/leads/0/maximum"}; empty path returns whole variable.
An expression may be a scalar, list, plain object, or {"op":NAME,"args":[...] }.
Supported ops: if(condition,then,else) lazy; ge(a,b); eq(a,b); sub(a,b); mul(a,b);
len(list); get(object,key); argmax(rows,numeric_key) returns first max row;
filter_eq(rows,key,value). No loops, code, extra operations or analysis_id arguments.
Use if+len before selecting empty lists. Missing fields are errors, never guess.
RR response has intervals, total_intervals, next_offset. This pilot supports <=100 RR only.
Window response: start_sample,end_sample,leads:[{lead,mean,maximum,peak_sample,...}].
RR row: rr_index,left_peak_sample,right_peak_sample,rr_seconds,retained.
Summary input: num_samples,sampling_rate. Summary evidence: temporal_regions with start,end,score.
Tool arguments follow the attached tool schemas. Scalar observations in the final answer
must cite actual tool evidence and paths; computed planning variables are not evidence.
'''


def response_metadata(reply):
    """Bounded protocol diagnostics, no raw prompts, content or argument values."""
    if not isinstance(reply, dict):
        return {'reply_type': type(reply).__name__}
    calls = reply.get('tool_calls', [])
    def label(x):
        return x[:100] if isinstance(x, str) else type(x).__name__
    return {'reply_type': 'dict', 'finish_reason': label(reply.get('finish_reason')),
        'content_type': type(reply.get('content')).__name__,
        'content_length': len(reply['content']) if isinstance(reply.get('content'), str) else None,
        'tool_calls_type': type(calls).__name__,
        'tool_call_count': len(calls) if isinstance(calls, list) else None,
        'tools': [{'type': label(c.get('type')), 'name': label(c.get('function', {}).get('name')),
            'arguments_type': type(c.get('function', {}).get('arguments')).__name__,
            'arguments_length': len(c['function']['arguments']) if isinstance(c.get('function', {}).get('arguments'), str) else None}
            for c in calls[:12] if isinstance(c, dict) and isinstance(c.get('function'), dict)] if isinstance(calls, list) else []}


def parse_plan_reply(reply):
    from src.agent.tool_protocol import strict_json
    from src.agent.answer_parser import parse_answer
    if not isinstance(reply, dict): raise ValueError('PLAN_REPLY_NOT_OBJECT')
    reason = reply.get('finish_reason')
    if reason == 'length': raise ValueError('PLAN_OUTPUT_TRUNCATED')
    if reason not in ('stop', 'tool_calls'): raise ValueError('PLAN_FINISH_REASON_INVALID')
    calls = reply.get('tool_calls', [])
    if not isinstance(calls, list): raise ValueError('PLAN_CALLS_NOT_LIST')
    if calls:
        if len(calls) != 1: raise ValueError('PLAN_SUBMIT_MUST_BE_ALONE')
        call = calls[0]
        if not isinstance(call, dict) or call.get('type') != 'function' or not isinstance(call.get('function'), dict):
            raise ValueError('PLAN_CALL_SCHEMA_INVALID')
        if call['function'].get('name') != 'submit_plan': raise ValueError('PLAN_WRONG_TOOL')
        args = call['function'].get('arguments')
        if not isinstance(args, str): raise ValueError('PLAN_ARGUMENTS_NOT_STRING')
        wrapper = strict_json(args)
        if not isinstance(wrapper, dict) or set(wrapper) != {'plan_json'} or not isinstance(wrapper['plan_json'], str):
            raise ValueError('PLAN_WRAPPER_INVALID')
        plan = strict_json(wrapper['plan_json'])
    else:
        if reason == 'tool_calls': raise ValueError('PLAN_CALLS_MISSING')
        plan = parse_answer(reply.get('content'))
    if not isinstance(plan, dict) or set(plan) != {'steps'}: raise ValueError('PLAN_SCHEMA')
    return plan


def pointer(value, path):
    if path == '': return value
    if not isinstance(path, str) or not path.startswith('/') or re.search(r'~(?![01])',path):
        raise ValueError('INVALID_POINTER')
    for k in path[1:].split('/'):
        k = k.replace('~1','/').replace('~0','~')
        if isinstance(value,list):
            if not re.fullmatch(r'0|[1-9][0-9]*', k): raise ValueError('INVALID_INDEX')
            value=value[int(k)]
        elif isinstance(value,dict): value=value[k]
        else: raise ValueError('POINTER_THROUGH_SCALAR')
    return value


def expression(x, env, depth=0):
    if depth > 16: raise ValueError('EXPRESSION_DEPTH')
    ev=lambda y: expression(y,env,depth+1)
    if isinstance(x,list): return [ev(y) for y in x]
    if not isinstance(x,dict):
        if x is None or type(x) in (str,bool,int,float): return x
        raise ValueError('INVALID_LITERAL')
    if 'ref' in x:
        if set(x)!={'ref','path'}: raise ValueError('REFERENCE_SCHEMA')
        return deepcopy(pointer(env[x['ref']],x['path']))
    if 'op' not in x: return {k:ev(v) for k,v in x.items()}
    if set(x)!={'op','args'} or not isinstance(x['args'],list): raise ValueError('OP_SCHEMA')
    op=x['op']; args=x['args']
    arity={'if':3,'ge':2,'eq':2,'sub':2,'mul':2,'len':1,'get':2,'argmax':2,'filter_eq':3}
    if op not in arity or len(args)!=arity[op]: raise ValueError('UNKNOWN_OP_OR_ARITY')
    if op=='if':
        condition=ev(args[0])
        if type(condition) is not bool: raise ValueError('CONDITION_TYPE')
        return ev(args[1] if condition else args[2])
    a=[ev(y) for y in args]
    if op=='eq': return type(a[0]) is type(a[1]) and a[0]==a[1]
    if op in ('ge','sub','mul'):
        if any(type(y) not in (int,float) or not math.isfinite(y) for y in a): raise ValueError('NUMBER_REQUIRED')
        return a[0]>=a[1] if op=='ge' else a[0]-a[1] if op=='sub' else a[0]*a[1]
    if op=='get':
        if not isinstance(a[0],dict) or not isinstance(a[1],str): raise ValueError('OBJECT_KEY_REQUIRED')
        return a[0][a[1]]
    if not isinstance(a[0],list) or len(a[0])>100: raise ValueError('BOUNDED_LIST_REQUIRED')
    if op=='len': return len(a[0])
    if not isinstance(a[1],str): raise ValueError('KEY_REQUIRED')
    if op=='filter_eq': return [r for r in a[0] if type(r[a[1]]) is type(a[2]) and r[a[1]]==a[2]]
    if not a[0] or any(type(r[a[1]]) not in (int,float) or not math.isfinite(r[a[1]]) for r in a[0]):
        raise ValueError('NONEMPTY_NUMERIC_ROWS_REQUIRED')
    return max(a[0],key=lambda r:r[a[1]])


def execute_plan(plan, executor, summary, evidence, trace, max_queries=6):
    if len(json.dumps(plan,allow_nan=False))>16000 or not isinstance(plan,dict) or set(plan)!={'steps'}:
        raise ValueError('PLAN_SCHEMA')
    if not isinstance(plan['steps'],list) or not 1<=len(plan['steps'])<=12: raise ValueError('PLAN_STEPS')
    require_valid_plan(plan)
    env={'summary':deepcopy(summary)}; seen=set(); queries=0
    for step in plan['steps']:
        if not isinstance(step,dict) or not isinstance(step.get('id'),str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,31}',step['id']) or step['id'] in env:
            raise ValueError('STEP_ID')
        if set(step)=={'id','value'}:
            env[step['id']]=expression(step['value'],env)
            trace.append({'stage':'plan_value','id':step['id'],'value':env[step['id']]})
            continue
        if 'when' in step:
            condition=expression(step['when'],env)
            if type(condition) is not bool: raise ValueError('WHEN_MUST_BE_BOOLEAN')
            if not condition:
                env[step['id']]=None
                trace.append({'stage':'plan_skip','id':step['id'],'tool':step['tool'],'reason':'when_false'})
                continue
        args=expression(step['args'],env)
        if not isinstance(args,dict) or 'analysis_id' in args: raise ValueError('ARGS_OR_SCOPE')
        identity=json.dumps([step['tool'],args],sort_keys=True,allow_nan=False)
        if identity in seen: raise ValueError('REPEATED_TOOL_CALL')
        if queries>=max_queries: raise ValueError('TOOL_BUDGET_EXHAUSTED')
        seen.add(identity);queries+=1;tick=time.perf_counter()
        response=executor.execute(step['tool'],args)
        trace.append({'stage':'tool','tool':step['tool'],'arguments':args,'ok':response.get('ok',False),
            'evidence_id':response.get('evidence_id'),'error':response.get('error'),'elapsed_seconds':time.perf_counter()-tick})
        if not response.get('ok'): raise ValueError('TOOL_EXECUTION_FAILED')
        if step['tool']=='inspect_rr_intervals' and (args.get('offset',0)!=0 or response['data'].get('next_offset') is not None):
            raise ValueError('RR_PAGINATION_OUT_OF_PILOT_SCOPE')
        response['data'].pop('provenance',None)
        evidence[response['evidence_id']]=deepcopy(response)
        env[step['id']]=deepcopy(response['data'])
    return env


def run_plan(store,aid,question,gateway):
    from src.tools.executor import ECGToolExecutor
    from src.agent.tool_protocol import TOOLS,SYSTEM,strict_json
    from src.agent.answer_parser import parse_answer
    from src.agent.answer_validator import validate_answer,scalar_paths
    executor=ECGToolExecutor(store,aid); trace=[]; evidence={}
    out={'analysis_id':aid,'status':'failed','error':'','draft':{},'validation':{},'evidence':evidence,'knowledge':{},'trace':trace,'model_calls':0,'plan_protocol_version':PROTOCOL_VERSION}
    phase='bootstrap'
    try:
        boot=executor.execute('get_analysis_summary',{})
        if not boot.get('ok'): raise ValueError('BOOTSTRAP_FAILED')
        boot['data'].pop('provenance',None);evidence[boot['evidence_id']]=boot
        trace.append({'stage':'tool','tool':'get_analysis_summary','arguments':{},'source':'bootstrap','ok':True,'evidence_id':boot['evidence_id']})
        descriptions=[t for t in TOOLS if t['function']['name'] in ALLOWED]
        phase='planning_gateway';out['model_calls']+=1
        reply=gateway.complete([{'role':'system','content':DSL+'\nTool schemas: '+json.dumps(descriptions,ensure_ascii=False)},
            {'role':'user','content':json.dumps({'summary':boot['data']},ensure_ascii=False)}, {'role':'user','content':question}], [{'type':'function','function':{'name':'submit_plan','description':'Submit bounded query plan once','parameters':{'type':'object','properties':{'plan_json':{'type':'string'}},'required':['plan_json'],'additionalProperties':False}}}])
        phase='planning_protocol'
        trace.append({'stage':'plan_protocol','version':PROTOCOL_VERSION,**response_metadata(reply)})
        plan=parse_plan_reply(reply)
        out['plan']=plan
        phase='plan_validation';require_valid_plan(plan)
        trace.append({'stage':'plan_validation','passed':True,'scope':'grammar_and_reference_order_only'})
        phase='plan_execution';execute_plan(plan,executor,boot['data'],evidence,trace)
        for item in evidence.values(): item['observation_paths']=scalar_paths(item['data'])
        phase='answer_gateway';out['model_calls']+=1
        reply=gateway.complete([{'role':'system','content':SYSTEM+'本轮仅根据已执行计划的工具证据提交答案，不能继续查询。'},
            {'role':'user','content':json.dumps({'analysis_id':aid,'evidence':evidence},ensure_ascii=False)},
            {'role':'user','content':question}], [t for t in TOOLS if t['function']['name']=='submit_answer'])
        phase='answer_protocol';calls=reply.get('tool_calls',[])
        if reply.get('finish_reason') not in ('stop','tool_calls'): raise ValueError('MODEL_OUTPUT_TRUNCATED')
        if calls:
            if len(calls)!=1 or calls[0].get('type')!='function' or calls[0]['function']['name']!='submit_answer': raise ValueError('ANSWER_PROTOCOL_INVALID')
            answer=strict_json(calls[0]['function']['arguments'])
        else: answer=parse_answer(reply.get('content'))
        phase='validation';validation=validate_answer(answer,evidence,{})
        out.update(status='completed_draft',draft=answer,validation=validation)
    except Exception as exc:
        out['error']='GATEWAY_REQUEST_FAILED' if phase.endswith('gateway') else getattr(exc,'code',str(exc)[:160])
        trace.append({'stage':'failure','phase':phase,'error_type':type(exc).__name__,'error':out['error'],**getattr(exc,'details',{})})
    out['supplemental_queries']=sum(t.get('stage')=='tool' and t.get('source')!='bootstrap' for t in trace)
    return out
