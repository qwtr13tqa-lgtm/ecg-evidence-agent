"""Locate saved candidate windows on stored ECG plots."""
import numpy as np
import plotly.graph_objects as go
from src.ui.reconstruction_view import figures, LEADS


def bounds(region,n,fs):
    start,end=region.get('start'),region.get('end')
    if type(start) is not int or type(end) is not int or not 0<=start<end<=n:
        raise ValueError('候选窗口坐标与保存的波形不匹配。')
    padding=max(.2,(end-start)/fs*.5)
    return [max(0,start/fs-padding),min(n/fs,end/fs+padding)]


def render_locator(st,result,raw):
    st.caption('选择区域与导联，波形自动定位；勾选完整片段可恢复总览。')
    evidence=result.to_llm_context()['evidence']
    regions=evidence.get('temporal_regions') or []
    aid=result.analysis_id
    if not regions:
        st.info('当前没有候选区域可定位；仍可在“波形与节律”查看完整重构曲线与误差热图。')
        return
    fs=result.input.sampling_rate
    n=len(raw)
    try:
        for region in regions:bounds(region,n,fs)
    except ValueError as exc:st.error(str(exc));return
    cols=st.columns([3,2,2])
    selected=cols[0].selectbox('定位窗口',range(len(regions)),
        format_func=lambda i:f"区域{i+1} · {regions[i]['start']/fs:.3f}–{regions[i]['end']/fs:.3f}秒 · [{regions[i]['start']},{regions[i]['end']})",
        key='region_locator_window_'+aid)
    ranked=[r.get('lead') for r in evidence.get('lead_evidence',[]) if r.get('lead') in LEADS]
    lead=cols[1].selectbox('查看导联',LEADS,index=LEADS.index(ranked[0]) if ranked else 1,
        key='region_locator_lead_'+aid)
    full=cols[2].checkbox('显示完整片段',key='region_locator_full_'+aid)
    chosen=regions[selected]
    st.caption('默认查看排名最高导联；候选区域来自跨导联聚合，不表示该导联在此窗口一定贡献最大。黄色为当前窗口，灰色为其他候选区域。')
    overview=go.Figure()
    for i,region in enumerate(regions):
        overview.add_trace(go.Bar(x=[(region['end']-region['start'])/fs],base=[region['start']/fs],
            y=['候选位置'],orientation='h',marker_color='#f59e0b' if i==selected else '#cbd5e1',
            name=f'区域{i+1}',customdata=[[region['start'],region['end']]],
            hovertemplate='片段采样点 [%{customdata[0]}, %{customdata[1]})<extra>%{fullData.name}</extra>'))
    overview.update_layout(barmode='overlay',height=150,showlegend=False,
        xaxis=dict(range=[0,n/fs],title='当前分析片段时间 / 秒'),margin=dict(t=10,b=45,l=20,r=20))
    st.plotly_chart(overview,key='region_locator_overview_'+aid)
    try:
        fig,norm,recon,score=figures(raw,result.model.reconstruction,result.model.error_map,lead,fs,[])
    except ValueError as exc:st.info(str(exc));return
    for row in (1,2,3):
        for i,region in enumerate(regions):
            fig.add_vrect(x0=region['start']/fs,x1=region['end']/fs,
                fillcolor='#f59e0b' if i==selected else '#94a3b8',opacity=.2 if i==selected else .08,
                line_width=0,row=row,col=1)
    fig.update_xaxes(range=[0,n/fs] if full else bounds(chosen,n,fs))
    fig.update_layout(title_text=f"区域{selected+1} · {lead}",height=570)
    st.plotly_chart(fig,key='region_locator_detail_'+aid)
    start,end=chosen['start'],chosen['end'];index=LEADS.index(lead)
    values=score[start:end,index]
    metrics=st.columns(4)
    metrics[0].metric('窗口长度',f'{(end-start)/fs:.3f} 秒')
    metrics[1].metric('窗口模型分数均值',f'{float(np.mean(values)):.4f}')
    metrics[2].metric('窗口模型分数最大值',f'{float(np.max(values)):.4f}')
    metrics[3].metric('首个最大值采样点',str(start+int(np.argmax(values))))
    st.caption('坐标相对当前分析片段，右端点不包含；最大值位置不是已确认事件。绝对重构差与模型组合分数含义不同，模型分数可为负。')

    with st.expander('12导联误差热图'):
        from src.ui.reconstruction_view import heatmap
        kind=st.radio('热图指标',['绝对重构差','模型组合分数'],horizontal=True,key='compact_heatmap_'+aid)
        values=np.abs(norm-recon) if kind=='绝对重构差' else score
        st.plotly_chart(heatmap(values,fs,kind),key='compact_heatmap_plot_'+aid)
        st.caption('颜色越深数值越大；色标按当前记录范围设置，不用于跨记录比较。')
    return lead
