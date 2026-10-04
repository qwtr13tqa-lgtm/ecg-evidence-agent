"""Single-call baselines with the same validator and history resolver as Agent."""
import json
import time
from src.agent.answer_validator import validate_answer, scalar_paths
from src.agent.answer_parser import parse_answer
from src.agent.tool_protocol import SYSTEM, TOOLS, strict_json
from src.agent.window_memory import prepare_reference, check_response, check_submission
from src.tools.executor import ECGToolExecutor
from src.evaluation.threeway_routes import route


def baseline_evidence(store,aid,scheme,question,gateway):
    if scheme not in ('summary','rules'):raise ValueError('Unknown scheme')
    executor=ECGToolExecutor(store,aid)
    pool={};trace=[]
    resolution=prepare_reference(gateway,aid,question)
    calls=[];reason='summary_only'
    if scheme=='rules':calls,reason=route(question,resolution)
    resolution={**resolution,'route_note':reason}
    # Summary never fetches new window values; history is context only.
    if scheme=='summary' and resolution.get('status')=='resolved':
        resolution={**resolution,'status':'context_only','message':'Summary baseline cannot requery this window.'}
    for name,args in [('get_analysis_summary',{})]+calls:
        tick=time.perf_counter();item=executor.execute(name,args)
        trace.append(dict(stage='tool',tool=name,arguments=args,ok=item.get('ok',False),
            evidence_id=item.get('evidence_id'),error=item.get('error'),
            source='bootstrap' if name=='get_analysis_summary' else 'rule_route',
            elapsed_seconds=time.perf_counter()-tick))
        if item.get('ok'):
            item['data'].pop('provenance',None)
            item['observation_paths']=scalar_paths(item['data']);pool[item['evidence_id']]=item
            if resolution.get('status')=='resolved' and name=='inspect_error_window':
                check_response(resolution,item);resolution['current_evidence_id']=item['evidence_id']
    trace.append(dict(stage='routing',reason=reason))
    return pool,trace,resolution


def run_fixed(store,aid,question,scheme,gateway):
    started=time.perf_counter()
    evidence,trace,resolution=baseline_evidence(store,aid,scheme,question,gateway)
    output={'analysis_id':aid,'status':'failed','error':'','draft':{},'validation':{},
        'evidence':evidence,'knowledge':{},'trace':trace,'model_calls':1,'tool_calls':len(trace),'requires_review':True}
    # Three messages keep the shared ConversationGateway insertion position compatible with Agent.
    messages=[{'role':'system','content':
        SYSTEM+'你是ECG研究辅助问答系统。仅依据给定数据回答，不编造测量或诊断。分数不是概率，峰测量未验证。'
        '本轮不能补查。信息不足时说明缺少什么。调用submit_answer提交中文答案及原样复制的标量observations。'
        '至少引用一条已提供证据，缺少专业数据时可引用输入元信息并说明无法判断。历史回答不是本轮证据。'},
        {'role':'user','content':json.dumps({'analysis_id':aid,'evidence':evidence,'reference_resolution':resolution},ensure_ascii=False,allow_nan=False)},
        {'role':'user','content':question}]
    tick=time.perf_counter();phase='gateway'
    try:
        reply=gateway.complete(messages,[t for t in TOOLS if t['function']['name']=='submit_answer'])
        phase='protocol';calls=reply.get('tool_calls',[])
        if reply.get('finish_reason') not in ('stop','tool_calls') or not isinstance(calls,list):raise ValueError('Invalid reply')
        if calls:
            if len(calls)!=1 or calls[0].get('type')!='function' or calls[0]['function']['name']!='submit_answer':raise ValueError('Unexpected tool')
            phase='answer_json';answer=strict_json(calls[0]['function']['arguments'])
        else:phase='answer_json';answer=parse_answer(reply.get('content'))
        phase='validation';validation=validate_answer(answer,evidence,{})
        check_submission(resolution,answer)
        output.update(status='completed_draft',draft=answer,validation=validation)
        output['trace'].append({'stage':'model','call':1,'elapsed_seconds':time.perf_counter()-tick})
        if calls:output['trace'].append({'stage':'submission','tool':'submit_answer','ok':True});output['tool_calls']+=1
    except Exception as exc:
        output['error']={'gateway':'GATEWAY_REQUEST_FAILED','protocol':'MODEL_PROTOCOL_INVALID','answer_json':'ANSWER_JSON_INVALID','validation':'ANSWER_VALIDATION_FAILED'}[phase]
        output['trace'].append({'stage':'model','error_type':type(exc).__name__,'failure_phase':phase,'elapsed_seconds':time.perf_counter()-tick})
    output['elapsed_seconds']=time.perf_counter()-started
    output['reference_resolution']=resolution
    return output

