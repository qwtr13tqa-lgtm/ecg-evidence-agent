"""Stored input/reconstruction visualization; no inference or external calls."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
LEADS=('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')

def prepare(raw,reconstruction,error_map):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=12 or len(raw)==0 or not np.isfinite(raw).all():
        raise ValueError('输入信号无效。')
    if reconstruction is None or error_map is None:raise ValueError('历史记录未保存重构或误差图，无法显示。')
    recon=np.asarray(reconstruction);score=np.asarray(error_map)
    if recon.shape!=raw.shape or score.shape!=raw.shape or not np.isfinite(recon).all() or not np.isfinite(score).all():
        raise ValueError('重构或误差图与输入形状不一致，或含无效数值。')
    normalized=raw.copy()
    for i in range(12):
        x=raw[:,i];lo=x.min();hi=x.max()
        normalized[:,i]=2*(x-lo)/(hi-lo)-1 if hi-lo>1e-6 else 0
    return normalized,recon,score

def figures(raw,reconstruction,error_map,lead,fs,regions=()):
    if lead not in LEADS or not isinstance(fs,(int,float)) or fs<=0:raise ValueError('导联或采样率无效。')
    norm,recon,score=prepare(raw,reconstruction,error_map)
    i=LEADS.index(lead);samples=np.arange(len(norm));t=samples/fs
    absolute=np.abs(norm[:,i]-recon[:,i])
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.06,
        subplot_titles=('模型输入与重构叠加','归一化幅值绝对差 |输入 − 重构|','模型组合分数（可为负）'),row_heights=[.5,.25,.25])
    for y,name,color in [(norm[:,i],'模型输入（归一化）','#2563eb'),(recon[:,i],'重构（策略平均）','#f97316')]:
        fig.add_trace(go.Scatter(x=t,y=y,customdata=samples,name=name,line=dict(color=color,width=1.3),
            hovertemplate='时间 %{x:.3f}s<br>片段采样点 %{customdata}<br>幅值 %{y:.5f}<extra>%{fullData.name}</extra>'),row=1,col=1)
    fig.add_trace(go.Scatter(x=t,y=absolute,customdata=samples,name='绝对重构差',fill='tozeroy',line=dict(color='#dc2626',width=1),
        hovertemplate='时间 %{x:.3f}s<br>片段采样点 %{customdata}<br>绝对差 %{y:.5f}<extra></extra>'),row=2,col=1)
    fig.add_trace(go.Scatter(x=t,y=score[:,i],customdata=samples,name='模型组合分数',line=dict(color='#7c3aed',width=1),
        hovertemplate='时间 %{x:.3f}s<br>片段采样点 %{customdata}<br>模型分数 %{y:.6f}<extra></extra>'),row=3,col=1)
    for region in regions:
        start,end=region.get('start'),region.get('end')
        if type(start) is int and type(end) is int and 0<=start<end<=len(norm):
            fig.add_vrect(x0=start/fs,x1=end/fs,fillcolor='#f59e0b',opacity=.12,line_width=0,row=1,col=1)
    fig.update_xaxes(title_text='相对当前分析片段时间 / 秒',row=3,col=1)
    fig.update_yaxes(title_text='归一化幅值',row=1,col=1)
    fig.update_layout(height=720,hovermode='x unified',legend=dict(orientation='h',y=1.12),margin=dict(t=100))
    return fig,norm,recon,score

def heatmap(values,fs,label):
    # Keep all saved samples; color scale uses this recording's full range.
    fig=go.Figure(go.Heatmap(z=values.T,x=np.arange(len(values))/fs,y=LEADS,
        colorscale='YlOrRd',colorbar=dict(title=label),
        hovertemplate='导联 %{y}<br>片段时间 %{x:.3f}s<br>数值 %{z:.6f}<extra></extra>'))
    fig.update_layout(height=430,xaxis_title='相对当前分析片段时间 / 秒',yaxis_title='导联')
    return fig

def render_reconstruction(st,result,raw,lead):
    st.subheader('输入、重构与误差定位')
    st.caption('蓝线为模型输入的归一化波形，橙线为已保存重构；沿用原始波形所选导联。上方原始波形保留其原有单位。')
    context=result.to_llm_context()
    try:
        fig,norm,recon,score=figures(raw,result.model.reconstruction,result.model.error_map,
            lead,result.input.sampling_rate,context['evidence'].get('temporal_regions') or [])
    except ValueError as exc:st.info(str(exc));return
    st.plotly_chart(fig,key='ecg_reconstruction_overlay')
    st.caption('浅黄色区间是现有候选时间区域（跨导联聚合），不代表当前导联的已确认事件。拖动框选可放大，双击恢复。')
    choice=st.radio('12导联热图指标',['绝对重构差','模型组合分数'],horizontal=True,key='ecg_error_heatmap_metric')
    values=np.abs(norm-recon) if choice=='绝对重构差' else score
    st.plotly_chart(heatmap(values,result.input.sampling_rate,choice),key='ecg_reconstruction_heatmap')
    st.caption('颜色越深表示所选指标越大。色标按当前记录取值范围显示，不用于跨记录比较；平坦信号不会被强制制造出高分区域。')
    with st.expander('两种误差为什么不同？'):
        st.write('绝对重构差比较归一化输入与策略平均重构，便于直观看波形哪里不一致。模型组合分数直接读取保存的error_map，包含加权重构项、sigma项和形状项，并对策略求平均；可为负，不等同于绝对差或异常概率。整体分类阈值不能直接画作逐点异常阈值。')
        st.write('这些图只展示本次已保存输入的模型空间差异；对于上传数据，尚未补齐的训练前置滤波不会因增加可视化而自动完成。')
