import json
from src.review.local_batch import plan,advance

LABELS={'pending':'待处理','running':'执行中','created':'新分析已保存','reused':'已复用匹配历史','failed':'失败'}

def render(st,root,history,pipeline_factory,open_record,report,batch):
    st.subheader('批量本地分析 · 先准备研究样本')
    st.caption('建议选择漏报FN和正确检出TP，为后续共性比较准备证据。不调用LLM，不做默认末尾窗口补查。')
    categories=st.multiselect('准备哪些类别',['fn','tp','fp','tn'],default=['fn','tp'],
        format_func=lambda x:{'fn':'漏报FN','tp':'正确检出TP','fp':'误报FP','tn':'正常判对TN'}[x],key='local_batch_categories')
    if not categories:return
    try:path,job=plan(report,categories,root)
    except Exception as exc:st.error('任务无法加载：'+str(exc));return
    data_path=st.text_input('批量分析使用的数据文件',value=str(root/'data/Processed_PTBXL/test.npy'),key='local_batch_data')
    limit=st.number_input('本次处理上限（非总样本数）',min_value=1,max_value=100,value=5,step=1,key='local_batch_limit')
    retry=st.checkbox('本次也重试失败记录',key='local_batch_retry')
    st.caption('每次按上限串行处理，避免同时占满CPU/GPU。每条完成立即保存；再次点击继续未完成项。已完成项重新核验后跳过。')
    bar=st.progress(0.0);note=st.empty()
    def update(j):
        n=len(j['rows']);done=sum(r['status'] in ('created','reused') for r in j['rows'])
        bar.progress(done/n if n else 0.0)
        running=[r['index'] for r in j['rows'] if r['status']=='running']
        note.caption(f'准备进度 {done}/{n} 条；执行中：{running or "无"}')
    if st.button('开始 / 继续批量本地分析',type='primary',key='local_batch_start',disabled=not job['rows']):
        try:path,job=advance(report,categories,data_path,root,history,pipeline_factory,int(limit),retry,update)
        except Exception as exc:st.error('批量执行未完成：'+str(exc))
    update(job)
    status_filter=st.selectbox('任务状态筛选',['all','created','reused','failed','pending','running'],format_func=lambda x:'全部' if x=='all' else LABELS[x],key='local_status_filter')
    st.dataframe([{'样本':r['index'],'状态':LABELS[r['status']],'尝试次数':r['attempts'],
        '分析ID':(r['analysis_id'] or '')[:8],'失败原因':r['error']} for r in job['rows'] if status_filter=='all' or r['status']==status_filter],hide_index=True,use_container_width=True)
    ready=[r for r in job['rows'] if r['status'] in ('created','reused')]
    if ready:
        selected=st.selectbox('打开已完成的分析',range(len(ready)),format_func=lambda i:'样本 '+str(ready[i]['index']),key='local_batch_open_index')
        # BATCH_STATS_NAV_FIX_V1
        from src.ui.cross_navigation import open_source as navigate
        st.button('打开匹配分析',key='local_batch_open',on_click=navigate,args=(st,open_record,ready[selected]['analysis_id']))
    st.download_button('下载批量任务记录',json.dumps(job,ensure_ascii=False,indent=2),file_name='local_analysis_job.json',key='local_batch_download')
    with st.expander('完整任务记录与保存位置'):
        st.json(job)
        st.caption('任务清单：'+str(path.relative_to(root))+'；实际分析保存在原历史库。记录中的“已完成”是上次检查结果，继续执行会重新核验。')
