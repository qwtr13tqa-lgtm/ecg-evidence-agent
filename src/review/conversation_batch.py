"""Bounded batch review within the single conversation UI.
LLM selects category; local code checks coverage and renders evidence facts.
No model inference, no changes to records or source-analysis permissions.
"""
import hashlib,json,time
from src.review.batch import query,validate
from src.review.open_analysis import compatible

TOOLS=[{'type':'function','function':{'name':'review_batch_history',
 'description':'对所选报告分类记录逐条检查兼容历史，读取匹配摘要；缺少则提示需要本地分析。不启动分析。支持误报fp、漏报fn、tp、tn及all。不支持其他条件或医学诊断。',
 'parameters':{'type':'object','properties':{'category':{'type':'string','enum':['fp','fn','tp','tn','all']}},'required':['category'],'additionalProperties':False}}}]


def review(report,history,category):
    selected=query(report,category,'desc'); results=[];offset=0;records=[]
    while True:
        page=history.list_analyses(limit=100,offset=offset);records.extend(page)
        if len(page)<100:break
        offset+=100
    for row in selected:
        found=None;invalid=False
        for record in records:
            if record['sample_index']!=row['index']:continue
            try:
                result,_,_=history.load_analysis(record['analysis_id'])
                if compatible(result,report,row):found=result;break
            except (ValueError,KeyError,OSError):invalid=True
        item={'index':row['index'],'label':row['label'],'prediction':row['prediction'],
              'score':row['score'],'threshold':report['threshold'],
              'state':'matched' if found else ('unreadable_history' if invalid else 'needs_local_analysis')}
        if found:
            context=found.to_llm_context()
            item.update(source_analysis_id=found.analysis_id,evidence={
                'model':context.get('model'), 'evidence':context.get('evidence'),
                'signal_features':context.get('signal_features')})
        results.append(item)
    if [x['index'] for x in results]!=[r['index'] for r in selected]:raise ValueError('Coverage mismatch')
    return results


def legacy_run(history,gateway,aid,question,report,report_hash):
    started=time.perf_counter();validate(report)
    base={'analysis_id':aid,'data_kind':'real_ecg','requires_review':True,'knowledge':{},'evidence':{},'draft':{},'validation':{},'trace':[]}
    reply=gateway.complete([{'role':'system','content':'当前范围是用户明确选择的评测批次，不是当前打开样本。调用review_batch_history选择要逐条复核的类别。只支持按分类查询兼容历史及读取已有摘要；不支持的条件请不调用工具，不能丢弃条件。'},
        {'role':'user','content':question}],TOOLS)
    calls=reply.get('tool_calls',[])
    if not calls:
        return dict(base,status='failed',error='BATCH_SCOPE_UNSUPPORTED',draft={'answer':'本批次对话目前支持：按误报/漏报等类别逐条检查匹配历史、读取摘要和提供入口。请明确类别；不会自动运行缺失分析。'})
    if reply.get('finish_reason')!='tool_calls' or len(calls)!=1:raise ValueError('Invalid batch protocol')
    c=calls[0]
    if c.get('type')!='function' or c['function']['name']!='review_batch_history':raise ValueError('Invalid tool')
    args=json.loads(c['function']['arguments'])
    if not isinstance(args,dict) or set(args)!={'category'}:raise ValueError('Invalid args')
    rows=review(report,history,args['category'])
    eid=aid+':batch:'+report_hash+':'+args['category']
    data={'batch_sha256':report_hash,'category':args['category'],'record_count':len(rows),'rows':rows}
    lines=[f"所选批次 {report_hash[:12]}：按 {args['category']} 检查 {len(rows)} 条记录。以下为本地工具生成的核查结果。"]
    for r in rows:
        label={'matched':'已有匹配分析，已读取摘要','needs_local_analysis':'未找到可验证匹配的分析，需要本地分析','unreadable_history':'存在无法读取的历史，需检查后重试'}[r['state']]
        lines.append(f"样本 {r['index']}：{label}；标签 {r['label']}，预测 {r['prediction']}，分数 {r['score']:.8f}，报告阈值 {r['threshold']:.8f}。")
    lines.append('未执行新的深度模型分析。版本匹配按输入、权重、适配器、阈值与分数检查，不以configured代替。')
    base.update(status='completed_draft',error='',draft={'answer':'\n\n'.join(lines),'evidence_ids':[eid],'knowledge_ids':[],'observations':[]},
      validation={'passed':True,'scope':'deterministic_batch_coverage_and_history_matching; question_intent_not_verified'},
      evidence={eid:{'ok':True,'analysis_id':aid,'evidence_id':eid,'data':data,'scope':'explicit_batch_report'}},
      trace=[{'stage':'tool','tool':'review_batch_history','arguments':args,'ok':True,'evidence_id':eid,'batch_sha256':report_hash,
              'task_state':rows}],model_calls=1,tool_calls=1,elapsed_seconds=time.perf_counter()-started)
    return base


def context_ui(st,root,cid):
    files=sorted((root/'evaluation/website_path_reports').glob('*.json'))
    scope=st.radio('本轮查询范围',['当前分析','所选批次'],horizontal=True,key='scope_'+cid)
    if scope=='当前分析':return None
    if not files:st.warning('没有批次报告');st.stop()
    remembered=st.session_state.get('_batch_report_keep');names=[str(p) for p in files]
    path=st.selectbox('对话关联批次',files,index=names.index(remembered) if remembered in names else 0,format_func=lambda p:p.name,key='scope_report_'+cid)
    content=path.read_bytes();report=validate(json.loads(content));digest=hashlib.sha256(content).hexdigest()
    st.caption('本轮可跨所选批次查询匹配记录，自动读取分组统计，按需查询窗口或RR；不自动重跑缺失分析。')
    return report,digest


def legacy_render_results(st,out,open_record,key):
    for e in (out.get('evidence') or {}).values():
        if e.get('scope')!='explicit_batch_report':continue
        data=e['data']
        st.caption('实际执行类别：'+data['category']+'；批次：'+data['batch_sha256'][:12])
        for row in data['rows']:
            if row['state']=='matched':
                with st.expander('样本 '+str(row['index'])+' 已读取证据'):st.json(row['evidence'])
                if st.button('打开样本 '+str(row['index'])+' 的匹配分析',key='batch_link_'+key+e['evidence_id']+str(row['index'])):
                    open_record(row['source_analysis_id']);st.rerun()
            else:
                st.caption('样本 '+str(row['index'])+'：请通过侧栏“返回批量复核”选择记录并显式运行本地分析。')

# CROSS_RECORD_SCOPE_V1
def run(history,gateway,aid,question,report,report_hash):
    from src.review.cross_record_agent import run as cross_run
    return cross_run(history,gateway,aid,question,report,report_hash)

# COMPACT_BATCH_SOURCES_V1
def render_results(st,out,open_record,key):
    from src.ui.cross_navigation import render_sources
    render_sources(st,out,open_record,key)
