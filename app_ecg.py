"""Unified local ECG UI. Run: python -m streamlit run app_ecg.py"""
import os
os.environ['LANGSMITH_TRACING']='false'
os.environ['LANGCHAIN_TRACING_V2']='false'
import hashlib
import sqlite3
import time
import uuid
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from src.analysis.pipeline import ECGAnalysisPipeline
from src.analysis.history import HistoryRepository
from src.analysis.rr_facts import build_rr_facts, render_rr_facts
from src.reporting.readable_view import render_answer_details
from src.analysis.model_decision import get_model_decision, render_decision
from src.tools.window_alignment import inspect_recent_rr_alignment, inspect_window_rr_alignment
from src.agent.failure_help import failure_help, DiagnosticGateway
from src.agent.conversation import new_conversation, begin_turn, finish_turn, ConversationGateway
from src.reporting.conversation_export import build_conversation_export, conversation_markdown
from src.reporting.analysis_export import build_export,json_bytes,markdown_bytes,request_matches

# ECG_WORKSPACE_UI_V1
from src.ui.ecg_workspace import initialize as initialize_workspace, render_help, render_history
from src.ui.region_diagnostics import render_regions

ROOT=Path(__file__).resolve().parent
LEADS=('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')
st.set_page_config(page_title='ECG 信号分析与证据解释',layout='wide')
page_heading=st.empty()
# Scope notice is displayed beside the active analysis summary.

@st.cache_resource
def pipeline_resource():return ECGAnalysisPipeline()

@st.cache_resource
def history_resource():
    return HistoryRepository(ROOT/'local_history/history.sqlite3',recover=True,exclusive=True)

try:history=history_resource()
except Exception as exc:
    st.error('历史数据库不可用：'+str(exc));st.stop()


def open_record(analysis_id,conversation_id=None):
    for key in ('bundle','answer','question_input','conversation'):
        st.session_state.pop(key,None)
    st.query_params['analysis']=analysis_id
    if conversation_id is None:
        choices=history.conversations(analysis_id)
        conversation_id=choices[0]['conversation_id'] if choices else history.new_conversation(analysis_id)
    st.query_params['conversation']=conversation_id

initialize_workspace(history)
render_help(st,history)
# CROSS_RETURN_ENTRY_V1
from src.ui.cross_navigation import render_back
render_back(st)
st.sidebar.header('分析与历史')
mode=st.sidebar.radio('入口',['新建分析','历史记录','批量复核'],key='workspace_entry')
if mode!='批量复核':page_heading.markdown('## ECG 分析工作区')
st.sidebar.caption('已保存记录可跨刷新和服务重启恢复；新建分析不会删除历史。')
if mode=='新建分析':
    try:
        data=np.load(ROOT/'data/Processed_PTBXL/test.npy',mmap_mode='r',allow_pickle=False)
        sample=st.sidebar.number_input('新分析的样本索引',min_value=0,max_value=len(data)-1,value=0,step=1)
    except Exception as exc:
        data=None;st.sidebar.warning('无法加载新样本：'+type(exc).__name__+'；仍可打开历史。')
    if data is not None and st.sidebar.button('本地分析并保存'):
        result_new=None;pipeline=None
        try:
            tick=time.perf_counter()
            with st.spinner('运行模型并保存分析记录…'):
                pipeline=pipeline_resource()
                # The UI now queries disk snapshots, so terminal inference cache entries can be released.
                for record in pipeline.store.list_records():
                    if record.status!='running':
                        try:pipeline.store.discard(record.analysis_id)
                        except KeyError:pass
                raw=np.asarray(data[sample,100:4900,:],dtype=np.float32)
                result_new=pipeline.analyze(raw,source_id='test.npy',sample_index=int(sample),crop_start_sample=100)
                history.save_analysis(result_new,raw,time.perf_counter()-tick)
            open_record(result_new.analysis_id)
        except Exception as exc:
            st.error('分析或保存失败，未切换当前记录：'+type(exc).__name__)
        finally:
            if result_new is not None:
                try:pipeline.store.discard(result_new.analysis_id)
                except (KeyError,ValueError):pass
elif mode=='批量复核':
    from src.ui.batch_integrated import render as render_batch
    render_batch(st,ROOT,history,pipeline_resource,open_record)
    st.stop()
else:
    # HISTORY_SIDEBAR_V1
    with st.sidebar:
        render_history(st,history,open_record)
# BATCH_INTEGRATION_V1
if st.session_state.get('batch_open_notice'):
    st.success(st.session_state.pop('batch_open_notice'))
def return_to_batch():
    st.session_state['workspace_entry']='批量复核'
st.sidebar.button('返回批量复核',on_click=return_to_batch)

requested=st.query_params.get('analysis')
if not requested:
    st.write('新建一次本地分析，或从侧栏“历史记录”打开已保存分析。');st.stop()
try:
    result,stored_signal,active=history.load_analysis(requested)
    cid=st.query_params.get('conversation')
    if not cid:
        open_record(requested);cid=st.query_params['conversation']
    session=history.conversation(requested,cid)
except (KeyError,ValueError,sqlite3.Error) as exc:
    st.error('无法恢复所选记录：'+type(exc).__name__+'。请从历史记录重新选择。');st.stop()
from src.ui.step_guide import note as guide_note
guide_note(st,'analysis_open')
st.session_state['active']=active
st.session_state['conversation']=session
# ECG_COMPACT_LAYOUT_V1
aid=active['analysis_id'];context=result.to_llm_context();facts=build_rr_facts(result)
decision=get_model_decision(result)
record_name=f"样本 {active['sample']}" if active['sample'] is not None else '上传心电片段'
st.subheader(record_name)
st.caption('研究原型 · 模型结果不等同于临床诊断。峰检测尚未验证。')
summary=st.columns(4)
prediction=decision.get('prediction')
summary[0].metric('模型预测 · 开发阈值',{'model_normal':'正常','model_anomaly':'异常','model_abnormal':'异常'}.get(prediction,'暂不可判定'))
score=context['model'].get('anomaly_score');threshold=decision.get('threshold')
summary[1].metric('模型分数',f'{score:.4f}' if score is not None else '未提供')
summary[1].caption('阈值 '+(f'{threshold:.4f}' if threshold is not None else '未配置')+' · 非概率')
rhythm=context.get('signal_features',{}).get('rhythm') or {};hr=rhythm.get('heart_rate_bpm')
summary[2].metric('估计心率 / bpm',f'{hr:.1f}' if hr is not None else '未提供')
regions=context['evidence'].get('temporal_regions') or []
summary[3].metric('候选时间区域',str(len(regions))+' 个')
summary[3].caption('在下方信号分析中定位')
with st.expander('记录详情与判定依据'):
    st.write('分析创建时间：'+str(active.get('created_at','未提供')))
    st.code(aid)
    st.caption(f"本地分析耗时 {active['elapsed']:.2f} 秒；创建时间不等于心电采集时间。")
    st.write(render_decision(decision))
    with st.expander('完整分数、公式与阈值来源'):st.json(decision)
# PRODUCT_NAVIGATION_V1
from src.ui.product_help import is_product_help, render_page_help
nav_labels=['查看信号','向本记录提问','查看分析依据','导出记录']
nav=st.session_state.get('product_page','查看信号')
nav_columns=st.columns(4)
for column,label in zip(nav_columns,nav_labels):
    if column.button(label,type='primary' if nav==label else 'secondary',use_container_width=True,key='main_nav_'+label):
        st.session_state['product_page']=label
        st.rerun()
# Preserve selected lead when only one page is rendered.
selected_lead=st.session_state.get('region_locator_lead_'+aid,'II')

if nav=='查看信号':
    render_page_help(st,ROOT,'signal')
    # A single locator owns lead/window controls; avoid duplicate plot panels.
    if regions:
        from src.ui.region_locator import render_locator
        selected_lead=render_locator(st,result,stored_signal) or 'II'
    else:
        st.info('当前提取规则未返回候选窗口，可查看整段波形与误差。筛选原因见“分析详情”。')
        selected_lead=st.selectbox('查看导联',LEADS,index=1,key='workspace_wave_lead_'+aid)
        from src.ui.reconstruction_view import render_reconstruction
        render_reconstruction(st,result,stored_signal,selected_lead)
    with st.expander('原始幅值波形与候选峰'):
        index=LEADS.index(selected_lead);signal=stored_signal[:,index];fs=result.input.sampling_rate
        fig=go.Figure(go.Scatter(x=np.arange(len(signal))/fs,y=signal,mode='lines',name=selected_lead))
        if result.rhythm is not None and result.rhythm.lead_index==index and result.rhythm.r_peaks:
            peaks=np.asarray(result.rhythm.r_peaks,dtype=int);peaks=peaks[(peaks>=0)&(peaks<len(signal))]
            fig.add_trace(go.Scatter(x=peaks/fs,y=signal[peaks],mode='markers',name='候选峰（未验证）'))
        unit='mV' if result.provenance.get('upload',{}).get('units')=='mV' else '单位未核实'
        fig.update_layout(xaxis_title='当前片段时间 / 秒',yaxis_title='输入幅值 / '+unit,height=300)
        st.plotly_chart(fig,key='workspace_raw_signal')
if nav=='查看分析依据':
    render_page_help(st,ROOT,'details')
    st.caption('按需展开计算细节；波形定位统一在“信号分析”中操作。')
    with st.expander('候选区域筛选过程与配置'):
        render_regions(st,result,context)
    with st.expander('导联证据排名'):
        st.dataframe(context['evidence'].get('lead_evidence') or [],hide_index=True)
        st.caption('排名是当前样本内的相对证据分布，不是已确认病变位置。')
    with st.expander('节律与完整RR明细'):
        st.write(render_rr_facts(facts))
        if facts['status']=='verified':st.dataframe(facts['rr_intervals'])
    with st.expander('模型窗口与RR时间对齐'):
        st.caption('时间重叠仅用于对照，不能据此确定或排除疾病。')
        try:
            alignment_lead=selected_lead
            if regions:
                region_index=st.session_state.get('region_locator_window_'+aid,0)
                region=regions[min(max(int(region_index),0),len(regions)-1)]
                alignment=inspect_window_rr_alignment(result,int(region['start']),int(region['end']),alignment_lead)
            else:
                alignment=inspect_recent_rr_alignment(result,.6,alignment_lead)
                st.caption('无候选区域时显示末尾0.6秒，这是人为指定窗口。')
            w=alignment['window']
            st.write(f"{alignment_lead} · [{w['start_seconds']:.3f}, {w['end_seconds']:.3f})秒 · [{w['start_sample']}, {w['end_sample']})点")
            st.dataframe(w['leads'])
            if alignment['rr_status']=='verified':
                st.caption(f"RR提取导联：{alignment['rhythm_lead']}；时间重叠RR共 {alignment['overlapping_rr_count']} 个。")
                if alignment['rr_intervals']:st.dataframe(alignment['rr_intervals'])
                else:st.write('没有已检测RR间隔与该窗口重叠。')
            else:st.warning('RR核算不可用：'+str(alignment['rr_errors']))
        except (ValueError,TypeError,KeyError) as exc:st.warning('对齐暂不可用：'+type(exc).__name__)
    with st.expander('完整模型输出与来源（调试）'):st.json(context)

if nav=='向本记录提问':
    render_page_help(st,ROOT,'chat')
    session=st.session_state['conversation']
    st.caption('同一样本可连续追问。最近三轮成功问答用于理解上下文；数值证据每轮重新查询。每个对话最多20轮，提交后自动保存。')
    conversations=history.conversations(aid)
    options=[c['conversation_id'] for c in conversations]
    titles={c['conversation_id']:f"{c['created_at'][:19]} UTC · {c['conversation_id'][:8]}" for c in conversations}
    chosen=st.selectbox('当前分析的对话',options,index=options.index(cid),format_func=lambda value:titles[value])
    if chosen!=cid:
        open_record(aid,chosen);st.rerun()
    if st.button('新建对话（保留已有对话）'):
        open_record(aid,history.new_conversation(aid));st.rerun()
    if session['pending'] is not None:
        st.warning('有一轮已提交，尚未记录完成结果。可刷新查看状态；刷新不会自动重发。')
        if st.button('刷新当前对话'):st.rerun()
        if st.button('结束本地等待（不保证取消网关请求）'):
            history.abandon_turn(aid,cid,session['pending']['request_id']);st.rerun()
    for i, turn in enumerate(session['turns'],1):
        with st.chat_message('user'):st.write(turn['question'])
        with st.chat_message('assistant'):
            out=turn['output']
            if out.get('status')=='completed_draft':
                st.write(out['draft']['answer'])
                from src.ui.answer_waveform import render_answer_waveforms
                render_answer_waveforms(st,out,result,stored_signal,
                    key='answer_wave_'+aid+'_'+cid+'_'+str(i))
                st.caption('本地使用说明，未调用大模型。' if any(t.get('stage')=='local_help' for t in out.get('trace',[])) else ('本地筛选排序结果，未调用大模型。' if any(t.get('stage')=='local_query' for t in out.get('trace',[])) else ('LLM解析查询条件，本地程序计算结果；请核对实际条件。' if any(t.get('stage')=='collection_plan' and t.get('plan',{}).get('mode') in ('query','clarify') for t in out.get('trace',[])) else '原始模型回答；结构通过不代表文字语义正确。')))
            elif out.get('status')=='needs_clarification':
                st.info(out.get('draft',{}).get('answer','请明确查询范围。'))
            else:
                from src.ui.failure_feedback import render as render_failure
                render_failure(st,out,can_retry_explanation=False)
            from src.review.conversation_batch import render_results
            render_results(st,out,open_record,cid+'_'+str(i))
            with st.expander(f'第 {i} 轮证据、调用流程与校验'):render_answer_details(st,out)
    # UNIFIED_SCOPE_V1
    from src.review.conversation_batch import context_ui
    batch_context=context_ui(st,ROOT,cid)
    from src.review.simple_batch_query import parse_simple_query,run_simple
    question=st.text_area('提问或询问系统用法',key='question_input',
        placeholder='例如：心率依据哪些 RR？随后可追问：其中最长的间隔是哪一个？',max_chars=4000)
    consent=st.checkbox('允许将当前所选范围（单条或批次）的结构化证据、问题及最近对话发送到已配置网关（默认 HTTP）',key='external_consent')
    if len(session['turns'])>=20:st.info('本会话已达20轮，请新建对话；当前对话已保存。')
    local_batch_request=batch_context is not None and parse_simple_query(question) is not None
    if st.button('发送并查询证据',disabled=(not consent and not is_product_help(question) and not local_batch_request) or not question.strip() or len(session['turns'])>=20 or session['pending'] is not None):
        try:rid=history.begin_turn(aid,cid,question)
        except (ValueError,sqlite3.IntegrityError):
            st.warning('该对话已有请求或状态已变化，请刷新。');st.stop()
        cid=session['conversation_id']
        st.session_state.pop('bundle',None)
        gateway=None;diagnostic=None;config={}
        if is_product_help(question):
            try:
                from src.ui.product_help import answer as product_answer
                out=product_answer(ROOT,aid,question)
            except Exception as exc:
                out={'analysis_id':aid,'status':'failed','error':'PRODUCT_HELP_UNAVAILABLE','draft':{},'validation':{},'evidence':{},'trace':[]}
        elif local_batch_request:
            try:
                out=run_simple(history,aid,question,*batch_context)
                config={'mode':'deterministic_batch_query','model':None,'max_model_calls':0}
            except Exception as exc:
                out={'analysis_id':aid,'status':'failed','error':'LOCAL_BATCH_QUERY_FAILED','draft':{},'validation':{},'evidence':{},'knowledge':{},'trace':[{'stage':'local_query','error_type':type(exc).__name__}]}
        else:
            try:
                from src.agent.gateway import ToolGateway
                from src.agent.evidence_agent import ECGEvidenceAgent
                from src.knowledge.retriever import BM25Retriever
                with st.spinner('正在解析批次任务并计算，复杂复核在预算内继续…' if batch_context is not None else 'Agent 查询中；每轮最多4次请求，每次180秒，无自动重试…'):
                    retriever=BM25Retriever.from_jsonl(ROOT/'data/knowledge/ecg_knowledge.jsonl')
                    key=os.getenv('ECG_API_KEY','')
                    if not key or any(c.isspace() for c in key) or key.lower().startswith('bearer '):
                        raise ValueError('INVALID_LOCAL_KEY_CONFIGURATION')
                    gateway=ToolGateway()
                    config={'model':gateway.model,'timeout_seconds':gateway.timeout,'max_tokens':gateway.max_tokens,
                            'max_model_calls':6 if batch_context is not None else 4,'max_tool_calls':12 if batch_context is not None else 6,'sdk_retries':0,
                            'endpoint_sha256':hashlib.sha256(gateway.base_url.encode()).hexdigest()}
                    diagnostic=DiagnosticGateway(gateway)
                    wrapped=ConversationGateway(diagnostic,session,aid)
                    if batch_context is None:
                        out=ECGEvidenceAgent(history,retriever,wrapped).run(aid,question,allow_external=True)
                    else:
                        from src.review.conversation_batch import run as run_batch_conversation
                        out=run_batch_conversation(history,diagnostic,aid,question,*batch_context)
            except Exception as exc:
                out={'analysis_id':aid,'status':'failed','error':'LOCAL_SETUP_OR_EXECUTION_FAILED',
                     'draft':{},'validation':{},'evidence':{},'knowledge':{},'trace':[{'stage':'setup','error_type':type(exc).__name__}]}
            finally:
                if gateway:
                    try:gateway.close()
                    except Exception:pass
        if out.get('status')!='completed_draft':
            if diagnostic and diagnostic.last_error:out['gateway_failure']=diagnostic.last_error
            if batch_context is None:out['local_support']={'source':'deterministic_local_computation_not_llm',
                'analysis_id':aid,'model_decision':decision,'rr_facts_text':render_rr_facts(facts)}
        # Save to the originally bound conversation even if another view was opened meanwhile.
        try:
            if not history.finish_turn(aid,cid,rid,out,config):
                st.warning('该请求已被结束等待或中断；迟到结果没有覆盖原记录。')
        except Exception as exc:
            st.warning('回答尚未成功保存。本轮仍为待完成，请先保留当前页面和记录，联系维护者检查保存过程。')
            with st.expander('保存失败诊断'):st.json({'error_type':type(exc).__name__})
            st.stop()
        if out.get('status')=='completed_draft' and not is_product_help(question):
            guide_note(st,'answer_saved')
        st.rerun()
if nav=='导出记录':
    render_page_help(st,ROOT,'export')
    st.write('分析及对话已自动保存。这里导出当前对话供分享；其它对话可切换后分别导出。')
    if st.button('准备导出完整会话'):
        try:st.session_state['bundle']=build_conversation_export(result,history.conversation(aid,cid))
        except Exception as exc:
            st.session_state.pop('bundle',None);st.error('导出校验失败：'+type(exc).__name__)
    bundle=st.session_state.get('bundle')
    if bundle and bundle['analysis_id']==aid and bundle.get('conversation_id')==st.session_state['conversation']['conversation_id']:
        st.caption(f"已保存 {len(bundle['turns'])} 轮。文字语义未自动验证。")
        st.download_button('下载完整会话 JSON',json_bytes(bundle),file_name='ecg_chat_'+aid+'.json',mime='application/json')
        st.download_button('下载可读会话 Markdown',conversation_markdown(bundle),file_name='ecg_chat_'+aid+'.md',mime='text/markdown')
        st.caption('不包含完整原始 ECG、重构或误差数组。')
