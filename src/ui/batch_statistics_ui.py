import json
import pandas as pd
from src.review.batch_statistics import collect


def navigate(st,open_record,aid):
    # Runs before widgets are instantiated in the subsequent Streamlit script pass.
    # CROSS_RETURN_CALLBACK_V1
    from src.ui.cross_navigation import open_source
    open_source(st,open_record,aid)


def render(st,root,history,open_record,report,batch):
    st.subheader('初步批量统计 · FN / TP及其他类别')
    categories=st.multiselect('统计类别',['fn','tp','fp','tn'],default=['fn','tp'],key='batch_stats_categories')
    st.caption('读取同一历史库的匹配分析，不运行模型、不调用LLM。分数来自报告；其他指标来自匹配历史。不同字段分别报告有效与缺失数量。')
    key='batch_stats_'+batch+'_'+','.join(sorted(categories))
    if st.button('生成 / 刷新批量统计',disabled=not categories,key='batch_stats_refresh'):
        try:st.session_state[key]=collect(report,history,categories)
        except Exception as exc:st.error('统计失败：'+str(exc));st.session_state.pop(key,None)
    result=st.session_state.get(key)
    if not result:return
    st.caption('统计快照时间：'+result['created_at']+'。新增/删除分析后请刷新。')
    st.dataframe(result['groups'],hide_index=True,use_container_width=True)
    st.caption('median=中位数，q1/q3=25%/75%分位数；valid_n=有效数，missing_n=缺失数。区域数0是有效值，缺失不是0。')
    rows=[]
    for r in result['rows']:
        rows.append({**r,'lead_evidence':json.dumps(r['lead_evidence'],ensure_ascii=False),
            'temporal_regions':json.dumps(r['temporal_regions'],ensure_ascii=False),
            'missing_reasons':json.dumps(r['missing_reasons'],ensure_ascii=False)})
    with st.expander('逐样本指标、来源和缺失原因'):
        st.dataframe(rows,hide_index=True,use_container_width=True)
    with st.expander('排名第一导联的分布（非病变定位）'):
        if result['top_leads']:st.dataframe(result['top_leads'],hide_index=True,use_container_width=True)
        else:st.info('没有可统计的唯一第一导联。')
    matched=[r for r in result['rows'] if r['analysis_id']]
    if matched:
        i=st.selectbox('定位到统计样本',range(len(matched)),format_func=lambda i:f"{matched[i]['category']} · 样本 {matched[i]['index']}",key='batch_stats_open_choice')
        st.button('打开统计来源分析',key='batch_stats_open',on_click=navigate,args=(st,open_record,matched[i]['analysis_id']))
    st.download_button('下载逐样本CSV',pd.DataFrame(rows).to_csv(index=False).encode('utf-8-sig'),file_name='batch_samples.csv',key='stats_csv')
    st.download_button('下载分组统计CSV',pd.DataFrame(result['groups']).to_csv(index=False).encode('utf-8-sig'),file_name='batch_groups.csv',key='stats_groups')
    st.download_button('下载完整统计JSON',json.dumps(result,ensure_ascii=False,indent=2),file_name='batch_statistics.json',key='stats_json')
    st.info('这些差异是复核线索，不能直接解释漏报原因。FN与TP数量、缺失率及版本匹配情况都需同时检查。')
