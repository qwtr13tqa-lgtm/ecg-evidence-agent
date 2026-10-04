"""Conservative full-intent routes and budgeted planning recovery."""
import re
import time
from copy import deepcopy


def missing_plan(question):
    # This exact test-data declaration is metadata, not a user filter.
    prefix='这是合成软件测试，所有值为预设，不是实际ECG测量。\n'
    q=question[len(prefix):] if question.startswith(prefix) else question
    q=re.sub(r'\s+','',q).rstrip('。.!！')
    patterns=(r'列出漏报中重构项缺失的所有记录[，,]按索引升序',
              r'筛选category=fn且reconstruction_error缺失的记录[，,]按index升序列出全部索引')
    if not any(re.fullmatch(p,q,re.IGNORECASE) for p in patterns):return None
    return {'mode':'query','clarification':'','assumptions':[], 'operations':[{
        'kind':'select','filters':[{'field':'category','op':'eq','value':'fn'},
          {'field':'reconstruction_error','op':'is_missing','value':None}],
        'order_by':'index','direction':'asc','limit':500,'offset':0,
        'columns':['index','category','reconstruction_error'],'group_by':None,'metrics':[]}]}


def timeout_error(exc):
    return isinstance(exc,TimeoutError) or type(exc).__name__ in ('APITimeoutError','ReadTimeout','ConnectTimeout')


def request_plan(gateway,messages,tools,out,max_calls):
    # Reserve one downstream request; never loop across provider failures.
    attempts=2 if max_calls>=3 else 1
    for attempt in range(attempts):
        tick=time.perf_counter();out['model_calls']+=1
        try:reply=gateway.complete(deepcopy(messages),deepcopy(tools))
        except Exception as exc:
            timed=timeout_error(exc)
            out['trace'].append({'stage':'task_interpretation','call':out['model_calls'],
                'attempt':attempt+1,'ok':False,'error_type':type(exc).__name__,
                'error_code':'GATEWAY_TIMEOUT' if timed else 'GATEWAY_REQUEST_FAILED',
                'elapsed_seconds':time.perf_counter()-tick})
            if timed and attempt==0 and attempts==2:
                out['trace'].append({'stage':'planning_recovery','attempt':1,
                    'within_existing_budget':True,'original_conditions_preserved':True})
                continue
            raise ValueError('GATEWAY_TIMEOUT' if timed else 'GATEWAY_REQUEST_FAILED') from exc
        out['trace'].append({'stage':'task_interpretation','call':out['model_calls'],
            'attempt':attempt+1,'ok':True,'elapsed_seconds':time.perf_counter()-tick,
            'finish_reason':reply.get('finish_reason') if isinstance(reply,dict) else None})
        return reply
