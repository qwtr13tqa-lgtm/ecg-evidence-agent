"""Predeclared baselines. No rubric or reference answer is accepted here."""
import copy
import hashlib
import json
import time
from src.analysis.interpretation import build_interpretation
from src.tools.executor import ECGToolExecutor
from src.agent.tool_protocol import SYSTEM, TOOLS, strict_json
from src.agent.answer_parser import parse_answer
from src.agent.answer_validator import validate_answer, scalar_paths

SCHEMES = ('summary', 'summary_rag', 'fixed_tools', 'agent')


def run_baseline(store, retriever, gateway, analysis_id, question, scheme):
    if scheme not in SCHEMES[:-1]:
        raise ValueError('Unknown baseline')
    started = time.perf_counter()
    result = store.get_result(analysis_id)
    policy = build_interpretation(result)
    executor = ECGToolExecutor(store, analysis_id)
    evidence, knowledge, trace = {}, {}, []
    output = dict(analysis_id=analysis_id, data_kind='real_ecg', status='failed', error='',
                  draft={}, validation={}, limitations=policy['limitations'], requires_review=True,
                  evidence=evidence, knowledge=knowledge, trace=trace, model_calls=0, tool_calls=0)
    def collect(name, args):
        tick=time.perf_counter()
        if name == 'search_knowledge':
            try:
                docs=[hit.to_dict() for hit in retriever.search(**args)]
                suffix=hashlib.sha256(json.dumps([name,args],sort_keys=True).encode()).hexdigest()[:16]
                response={'analysis_id':analysis_id,'ok':True,'evidence_id':analysis_id+':knowledge:'+suffix,
                          'data':{'documents':docs,'status':'candidates_returned' if docs else 'no_match'}}
                knowledge.update({d['id']:d for d in docs})
            except Exception:
                response={'analysis_id':analysis_id,'ok':False,'error':'RETRIEVAL_FAILED'}
        else:
            response=executor.execute(name,args)
        if response.get('ok'):
            response['data'].pop('provenance',None)
            response['observation_paths']=scalar_paths(response['data'])
            evidence[response['evidence_id']]=response
        trace.append(dict(stage='tool',tool=name,arguments=args,ok=response.get('ok',False),
                          error=response.get('error'),evidence_id=response.get('evidence_id'),
                          elapsed_seconds=time.perf_counter()-tick))
        output['tool_calls']+=1
        return response
    summary=collect('get_analysis_summary',{})
    if not summary.get('ok'):
        output['error']='ANALYSIS_UNAVAILABLE'
        output['elapsed_seconds']=time.perf_counter()-started
        return output
    # Fixed schedule does not inspect the question or evaluation expectation.
    # Explicitly limited scope: one RR page and V1's final 0.6 seconds.
    if scheme == 'fixed_tools':
        collect('inspect_rr_intervals',{'offset':0,'limit':50})
        collect('inspect_recent_error',{'duration_seconds':0.6,'lead':'V1'})
    if scheme in ('summary_rag','fixed_tools'):
        collect('search_knowledge',{'query':question,'top_k':3})
    payload={'analysis_id':analysis_id,'data_kind':'real_ecg','interpretation':policy,
             'evidence':evidence,'knowledge':knowledge,'question':question}
    text=json.dumps(payload,ensure_ascii=False,allow_nan=False)
    if len(text)>90000:
        output['error']='CONTEXT_BUDGET_EXHAUSTED'
        output['elapsed_seconds']=time.perf_counter()-started
        return output
    messages=[{'role':'system','content':SYSTEM+'\n本次为预先收集证据的固定基线。只能提交回答，不能补查。证据不足时明确说明；不得声称执行了未提供的查询。'},
              {'role':'user','content':text}]
    submit=[t for t in TOOLS if t['function']['name']=='submit_answer']
    phase='gateway'; tick=time.perf_counter(); output['model_calls']=1
    try:
        reply=gateway.complete(copy.deepcopy(messages),copy.deepcopy(submit))
        phase='response_protocol'
        calls=reply.get('tool_calls',[])
        if reply.get('finish_reason') not in ('stop','tool_calls') or not isinstance(calls,list):
            raise ValueError('Invalid response')
        if calls:
            if len(calls)!=1 or calls[0].get('type')!='function' or calls[0]['function']['name']!='submit_answer':
                raise ValueError('Baseline cannot query tools')
            phase='answer_json'; answer=strict_json(calls[0]['function']['arguments'])
        else:
            phase='answer_json'; answer=parse_answer(reply.get('content'))
        phase='answer_validation'
        validation=validate_answer(answer,evidence,knowledge)
        output.update(status='completed_draft',draft=answer,validation=validation)
        trace.append(dict(stage='model',call=1,elapsed_seconds=time.perf_counter()-tick,tool_call_count=len(calls)))
        if calls:
            output['tool_calls']+=1
            trace.append(dict(stage='submission',tool='submit_answer',ok=True))
    except Exception as exc:
        output['error']={'gateway':'GATEWAY_REQUEST_FAILED','response_protocol':'MODEL_PROTOCOL_INVALID',
                         'answer_json':'ANSWER_JSON_INVALID','answer_validation':'ANSWER_VALIDATION_FAILED'}[phase]
        trace.append(dict(stage='model',call=1,elapsed_seconds=time.perf_counter()-tick,
                          error_type=type(exc).__name__,failure_phase=phase))
    output['elapsed_seconds']=time.perf_counter()-started
    return output
