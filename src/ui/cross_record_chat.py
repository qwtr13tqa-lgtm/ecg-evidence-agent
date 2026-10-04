# CROSS_OUTPUT_FIX_V1
# CROSS_BUDGET_FIX_V1
import hashlib
import json
import uuid
from datetime import datetime,timezone
from src.review.cross_record_agent import run
from src.review.simple_batch_query import parse_simple_query
from src.evaluation.records import atomic_json
from src.ui.cross_navigation import render_sources


def render(st,root,history,open_record,report,batch):
    st.subheader('向本批样本提问')
    st.caption('范围为当前报告的FN、TP、FP、TN及其匹配历史；可跨记录调用已有工具。初步统计自动作为后台证据，不必先手动生成表格。')
    def use_example(text):st.session_state['cross_record_question']=text
    for col,label,text in zip(st.columns(2),['筛选漏报','比较 FN / TP'],['找出所有漏报中异常分数最低的5条记录。','比较FN与TP的重构项、形状项和区域数量，列出支持记录和反例。']):
        col.button(label,on_click=use_example,args=(text,),key='example_'+label)
    question=st.text_area('向所选批次提问',value='漏报样本有什么共同特点？与正确检出的异常样本比较，列出支持记录和反例，说明为什么没有被检出及哪些原因仍不能确定。',key='cross_record_question')
    consent=st.checkbox('允许将所选批次的结构化指标、问题和最近三轮对话发送到已配置网关',key='cross_record_consent')
    key='cross_record_dialogue_'+batch;turns=st.session_state.setdefault(key,[])
    local_request=parse_simple_query(question) is not None
    if local_request:st.caption('已识别本地筛选排序：不发送网关，直接显示完整列表。')
    start_new=st.button('开始跨记录复核',type='primary',disabled=(not consent and not local_request) or not question.strip(),key='cross_record_run')
    retry=st.button('仅重试上一轮解释（不补查）',disabled=not consent or not turns or (bool(turns[-1]['output'].get('local_query')) or not any(e.get('scope')=='explicit_batch_report' for e in turns[-1]['output'].get('evidence',{}).values())),key='cross_record_retry')
    if start_new or retry:
        gateway=None
        try:
            from src.agent.gateway import ToolGateway
            if retry or not local_request:gateway=ToolGateway(timeout=90,max_tokens=6000)
            rid=str(uuid.uuid4());aid='batch-review-'+rid
            recent=[{'question':t['question'],'unverified_previous_answer':t['output'].get('draft',{}).get('answer','')} for t in turns[-3:]]
            parent=turns[-1] if retry else None
            executed_question=parent['question'] if parent else question
            if parent:
                aid=parent['output']['analysis_id']
                output=run(None,gateway,aid,executed_question,None,batch,max_model_calls=1,snapshot=parent['output'])
            else:output=run(history,gateway,aid,executed_question,report,batch,recent=recent)
            record={'version':'cross-record-review-1','run_id':rid,'batch_sha256':batch,'question':executed_question,
                'attempt_kind':'explanation_retry' if retry else ('local_query' if local_request else 'agent_run'),'parent_run_id':parent['run_id'] if parent else None,
                'created_at':datetime.now(timezone.utc).isoformat(),'output':output,
                'config':{'model':gateway.model if gateway else None,'timeout_seconds':90,'max_tokens':6000,'max_model_calls':(1 if retry else 6) if gateway else 0,'max_tool_calls':12 if gateway else 0,'execution_mode':'agent' if gateway else 'deterministic_local_query',
                    'endpoint_sha256':hashlib.sha256(gateway.base_url.encode()).hexdigest() if gateway else None,
                    'agent_sha256':hashlib.sha256((root/'src/review/cross_record_agent.py').read_bytes()).hexdigest()}}
            path=root/'evaluation/cross_record_runs'/rid;path.mkdir(parents=True,exist_ok=False);atomic_json(path/'result.json',record)
            turns.append(record)
        except Exception as exc:
            st.warning('本轮执行或保存未完成，结果尚未确认保存。请保留页面，交由维护者检查；不要将它当作成功记录。')
            with st.expander('执行或保存诊断'):st.json({'error_type':type(exc).__name__})
        finally:
            if gateway is not None:gateway.close()
    if turns:
        latest=turns[-1];out=latest['output']
        overview=next((e['data'] for e in out.get('evidence',{}).values() if e.get('scope')=='explicit_batch_report'),None)
        if overview and overview.get('groups') and not any(e.get('scope')=='batch_collection' for e in out.get('evidence',{}).values()):
            st.write('**本地统计 · 已保存证据快照（不是LLM回答）**')
            st.caption('下表覆盖当前批次各类别；valid_n为该字段有效数，missing_n为缺失数。解释失败不影响查看；仅重试解释不会刷新这些数据。')
            st.dataframe([{k:g.get(k) for k in ('category','field','total_n','valid_n','missing_n','median','q1','q3')} for g in overview.get('groups',[])],use_container_width=True,hide_index=True)
            with st.expander('导联分布（本地统计）'):st.dataframe(overview.get('top_leads',[]),use_container_width=True)
        st.write('**本轮问题：** '+latest['question'])
        if out['status']=='completed_draft':st.markdown(out['draft']['answer'])
        else:
            computed=sum(e.get('scope')=='batch_collection' and e.get('data',{}).get('query_status')!='needs_clarification' for e in out.get('evidence',{}).values())
            st.warning(f'已保留 {computed} 项本地集合计算结果 · 完整回答未完成' if computed else '本轮回答未完成')
            from src.ui.failure_feedback import describe
            feedback=describe(out,can_retry_explanation=True)
            st.caption(feedback['reason'])
            with st.expander('下一步怎么做'):st.write(feedback['next_step'])
        render_sources(st,out,open_record,latest['run_id'])
        st.caption('统计差异提供复核线索，不证明漏报原因。执行细节及下载见“执行记录”。')


def render_execution(st,batch):
    turns=st.session_state.get('cross_record_dialogue_'+batch,[])
    if not turns:
        st.info('尚无本次会话的执行记录。运行一次查询后可在这里检查调用与证据。');return
    i=st.selectbox('选择执行记录',range(len(turns)),index=len(turns)-1,
        format_func=lambda i:f"第 {i+1} 次 · {turns[i]['question'][:48]}",key='execution_'+batch)
    latest=turns[i];out=latest['output']
    st.write(latest['question'])
    cols=st.columns(4)
    for col,label,value in zip(cols,['任务状态','模型请求','补充工具请求','耗时 / 秒'],
            [out.get('status'),out.get('model_calls',0),out.get('tool_calls',0),round(out.get('elapsed_seconds',0),2)]):col.metric(label,value)
    if out.get('status')!='completed_draft':
        from src.ui.failure_feedback import render as render_failure
        render_failure(st,out)
    with st.expander('调用轨迹与失败定位',expanded=True):st.json(out.get('trace',[]))
    with st.expander('完整参数、证据与校验'):st.json(latest)
    st.download_button('下载执行记录',json.dumps(latest,ensure_ascii=False,indent=2),file_name=latest['run_id']+'.json',key='execution_download')
    st.caption('原始文件保存在 evaluation/cross_record_runs；刷新后会话列表可能清空，已保存文件保留。')
