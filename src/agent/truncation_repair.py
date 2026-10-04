"""Bounded re-submission: discard partial reply, retain valid conversation and evidence."""
from copy import deepcopy

def diagnostics(reply):
    if not isinstance(reply,dict):return {}
    calls=reply.get('tool_calls')
    d={'finish_reason':reply.get('finish_reason'),'tool_calls_type':type(calls).__name__,
       'tool_call_count':len(calls) if isinstance(calls,list) else None}
    for source,keys in [('usage',('prompt_tokens','completion_tokens','total_tokens')),
                        ('request_metadata',('max_tokens','message_count','message_bytes'))]:
        values=reply.get(source)
        if isinstance(values,dict):
            d[source]={k:values[k] for k in keys if type(values.get(k)) is int and values[k]>=0}
    return d

def repair_messages(messages,plain_only=False):
    result=deepcopy(messages)
    result[0]['content']+='\n精简重提交阶段：禁止继续查询任何工具；仅使用已有本轮证据。正文尽量控制在300字内，保留问题必需字段和局限，不添加无关RR或诊断。observations保留必要的原始标量值、路径与证据ID，仍需通过原有校验。缺少证据必须明确说明，不得编造。'
    result.append({'role':'user','content':
        '上一条响应因长度限制未完成，未被接纳。请重新生成完整、简短答案，不要续写。'+
        ('直接输出符合原答案schema的JSON对象，不调用工具。' if plain_only else '仅调用submit_answer提交完整短答案；也可直接输出符合原答案schema的JSON对象。')})
    return result
