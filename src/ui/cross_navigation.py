"""Compact source chooser and return navigation via pre-render callbacks."""
def open_source(st,open_record,aid):
    origin={'query':dict(st.query_params),'entry':st.session_state.get('workspace_entry','历史记录')}
    open_record(aid)
    st.session_state['_cross_return_origin']=origin
    st.session_state['workspace_entry']='历史记录'


def return_origin(st):
    origin=st.session_state.pop('_cross_return_origin',None)
    if not origin:return
    for key in ('bundle','answer','question_input','conversation'):st.session_state.pop(key,None)
    st.query_params.clear();st.query_params.update(origin['query'])
    st.session_state['workspace_entry']=origin['entry']


def render_back(st):
    if st.session_state.get('_cross_return_origin'):
        st.button('← 返回刚才的批次 / 对话',key='cross_return_button',on_click=return_origin,args=(st,))


def render_sources(st,out,open_record,key):
    from src.ui.collection_results import render as render_collection
    render_collection(st,out,key)
    local=out.get('local_query') or next((e.get('data') for e in out.get('evidence',{}).values() if e.get('scope')=='explicit_batch_report' and e.get('data',{}).get('explanation_status')=='not_requested'),None)
    if local:
        st.write('**本地筛选结果 · 查询已完成（不是LLM回答）**')
        st.caption(f"实际条件：{local['category']}，分数 {local['order']}；共 {local['record_count']} 条。相同分数按样本编号升序。")
        from src.ui.batch_display import display_rows
        st.dataframe(display_rows(local['rows']),use_container_width=True,hide_index=True)
        if not local['rows']:st.info('没有符合条件的记录。')
        if local.get('warnings'):st.warning('筛选排序已完成；部分历史入口匹配失败，请查看诊断。')
        if any(not r.get('source_analysis_id') for r in local['rows']):st.caption('无匹配历史的记录仍保留在列表中；可到批量本地分析入口生成匹配分析。')
    sources={};queried=set()
    for e in (out.get('evidence') or {}).values():
        d=e.get('data') or {}
        if e.get('scope') in ('explicit_batch_report','batch_collection'):
            for r in d.get('rows',[]):
                aid=r.get('source_analysis_id') or r.get('analysis_id')
                if aid:sources[r['index']]={'analysis_id':aid,'data':r}
        elif e.get('scope')=='batch_record_query' and d.get('source_analysis_id'):
            index=d['sample_index'];queried.add(index)
            sources[index]={'analysis_id':d['source_analysis_id'],'data':d}
    if not sources:return
    with st.expander(f'来源记录（实际补查 {len(queried)} 条；匹配历史 {len(sources)} 条）',expanded=False):
        st.caption('批次概览不等于已逐条复核波形。一次选择一条记录查看，避免占满对话。')
        choices=sorted(sources,key=lambda i:(i not in queried,i))
        selected=st.selectbox('选择来源样本',choices,format_func=lambda i:f'样本 {i}'+(' · 已调用工具补查' if i in queried else ' · 仅批次概览'),key='source_select_'+key)
        st.button('打开所选样本',key='source_open_'+key,on_click=open_source,
                  args=(st,open_record,sources[selected]['analysis_id']))
        st.caption('完整来源对象可在执行记录的证据中查看。')
