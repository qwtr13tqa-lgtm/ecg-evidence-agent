"""Deterministic user-facing failure feedback. No model calls or state mutation."""
NETWORK={
 'APITimeoutError':('回答服务未能在等待时间内返回结果。','稍后可手动重试；反复出现时请维护者检查网关响应时间。'),
 'TimeoutError':('回答服务未能在等待时间内返回结果。','稍后可手动重试；反复出现时请维护者检查网关响应时间。'),
 'AuthenticationError':('回答服务的连接凭据未被接受。','请维护者检查服务凭据；反复提交相同问题不能解决认证问题。'),
 'APIConnectionError':('暂时无法连接回答服务。','请维护者检查服务连接，恢复后再手动重试。'),
 'RateLimitError':('回答服务暂时限制了请求，或可用额度不足。','请稍后重试；持续出现时请维护者检查服务配额。'),
 'InternalServerError':('回答服务返回了服务端错误。','请稍后手动重试；持续出现时请维护者检查模型通道。')}

def describe(output,can_retry_explanation=False):
    code=str(output.get('error') or 'UNKNOWN_FAILURE');trace=output.get('trace') or []
    gateway=output.get('gateway_failure') or {}
    candidates=[gateway.get('error_type')]+[t.get('error_type') for t in reversed(trace) if t.get('stage') in ('model','task_interpretation','plan_repair_result')]
    kind=next((x for x in candidates if x in NETWORK),None)
    local=[];roots=[];records=[];valid_evidence_n=0;aid=output.get('analysis_id')
    for key,e in (output.get('evidence') or {}).items():
        if not isinstance(e,dict) or e.get('ok') is not True or e.get('analysis_id')!=aid or e.get('evidence_id')!=key:continue
        valid_evidence_n+=1
        d=e.get('data') or {}
        if e.get('scope')=='batch_collection' and d.get('query_status')=='completed':local.append(d)
        if e.get('scope')=='explicit_batch_report':roots.append(d)
        if e.get('scope')=='batch_record_query':records.append(d.get('sample_index'))
    if local:
        agg=sum(d.get('operation',{}).get('kind')=='aggregate' for d in local)
        sel=sum(d.get('operation',{}).get('kind')=='select' for d in local)
        progress=f'已保留 {len(local)} 项集合计算结果（{agg} 项分组统计、{sel} 项记录查询），下方表格可以继续查看。'
    elif records:progress=f'已保留 {len(set(records))} 条记录的补查证据，但本轮完整回答尚未完成。'
    elif roots:progress='已有批次概览可查看，但尚不能据此认定本轮要求的查询或解释已经完成。'
    elif valid_evidence_n:progress='已有本地分析证据，可在下方调用记录中查看；完整回答尚未通过校验。'
    else:progress='本轮尚无已通过校验的完整回答，也没有可确认已完成的查询结果。'
    reason='本轮处理未能完成；现有诊断不足以确定具体原因。'
    action='请保留本轮记录并查看下方诊断，交由维护者定位；不需要先修改数据或阈值。'
    title='这次暂时没能完成回答'
    if code in ('GATEWAY_REQUEST_FAILED','GATEWAY_TIMEOUT'):
        reason,action=NETWORK.get(kind,('回答服务调用失败；现有记录未说明具体原因。','可稍后手动重试；如果持续失败，请把本轮诊断交给维护者。'))
        if local or records:title='查询结果已保留，回答暂未完成'
    elif code.startswith(('PLAN_','COLLECTION_','FINITE_','FILTER_','FIELD_','CATEGORY_','SORT_','GROUP_','PAGE_','SELECT_','AGGREGATE_','NUMERIC_','TEXT_','MISSING_OPERATOR')):
        reason='系统没有把问题转换成可执行的查询条件，已停止执行，避免使用错误条件。'
        if code.startswith('PLAN_CLARIFICATION'):
            reason='系统生成的计划在“执行查询”和“需要澄清”之间存在字段冲突，未能继续执行。'
        action='请查看本轮实际计划；若你的条件已清楚，这通常需要维护者修正解析逻辑，不必反复换同义问法。'
    elif code.startswith(('CITATION_','OBSERVATION_','ANSWER_','BATCH_EVIDENCE','SUBMIT_')):
        reason='生成的回答未通过证据或格式校验，系统没有将它作为可信结果发布。'
        action='可先查看已保存的证据；需要完整解释时可手动重试，重复失败请提交本轮诊断。'
    elif code in ('TASK_TIME_BUDGET_EXHAUSTED','CONTEXT_BUDGET_EXHAUSTED','TOOL_BUDGET_EXHAUSTED','MODEL_BUDGET_EXHAUSTED','MODEL_BUDGET_EXHAUSTED_AFTER_PLAN'):
        reason='本轮达到处理容量或查询次数上限，尚未完成全部要求。'
        action='可将后续任务缩小到一个比较维度或明确的一组记录；维护者应检查重复查询和上下文大小。'
    elif code=='FINALIZATION_QUERY_REJECTED':
        reason='最终解释阶段仍请求补查，系统已阻止执行，完整回答未完成。'
        action='已有证据保留；请查看调用记录并核对未完成范围。'
    elif code=='MODEL_OUTPUT_TRUNCATED':
        reason='回答服务返回的内容不完整，未能形成可校验的完整回答。'
        action='可要求先给出简短结论和关键证据；不要把截断内容当作完成结果。'
    elif code in ('NEEDS_LOCAL_ANALYSIS','ANALYSIS_UNAVAILABLE','HISTORY_CHANGED_OR_VERSION_MISMATCH'):
        reason='所需分析缺失、无法读取，或与当前批次版本不匹配。'
        action='请先在批量本地分析中补齐匹配记录，再继续复核；缺失值不能按零处理。'
    elif code in ('WAIT_ABANDONED','REQUEST_INTERRUPTED'):
        reason='上次等待已结束或服务进程中断，没有保存完整回答。'
        action='如仍需要结果，请手动重新提交；系统不会自动重复发送请求。'
    elif code=='EXTERNAL_TRANSFER_NOT_ENABLED':
        reason='当前未开启向回答服务发送问题和结构化证据的授权。'
        action='如希望使用模型解释，请在页面确认外发授权后再提交；本地结果仍可单独查看。'
    if can_retry_explanation and roots and code in ('GATEWAY_TIMEOUT','TASK_TIME_BUDGET_EXHAUSTED','GATEWAY_REQUEST_FAILED','MODEL_OUTPUT_TRUNCATED','CONTEXT_BUDGET_EXHAUSTED','MODEL_BUDGET_EXHAUSTED'):
        if kind not in ('AuthenticationError','APIConnectionError','RateLimitError'):
            action='可点击“仅重试上一轮解释（不补查）”，使用已保存证据再次生成回答，无需重跑统计。此操作会产生新的模型请求，仍可能失败。'
    return {'title':title,'reason':reason,'progress':progress,'next_step':action,'original_status':output.get('status'),
            'technical_code':code,'gateway_error_type':kind,'automatic_retry':any(t.get('stage')=='timeout_repair' for t in trace)}

def render(st,output,can_retry_explanation=False):
    view=describe(output,can_retry_explanation)
    st.warning(view['title'])
    st.write('**原因：** '+view['reason'])
    st.write('**已完成的部分：** '+view['progress'])
    st.write('**下一步：** '+view['next_step'])
    st.caption('这是本地执行状态说明，不是模型结论；任务仍未完成。'+('已在预算内重试一次最终解释，未重新查询。' if view['automatic_retry'] else '未自动重试超时请求。'))
    with st.expander('技术诊断（供维护者排查）'):
        st.json({'status':view['original_status'],'error':view['technical_code'],'gateway_error_type':view['gateway_error_type'],
                 'model_calls':output.get('model_calls'),'tool_calls':output.get('tool_calls'),
                 'recent_failures':[t for t in output.get('trace',[]) if t.get('error') or t.get('error_code') or t.get('error_type')][-5:]})
