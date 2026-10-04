"""Event-driven guide. Never submits a question or starts inference."""
SINGLE=[('打开一条记录','侧栏选择“新建分析”并点击分析，或从历史记录打开一条。','analysis_open'),
 ('查看信号','进入“查看信号”，切换导联，查看输入与重构曲线。看完后点击“已完成本步”。','confirm'),
 ('发送窗口问题','进入“向本记录提问”，查询范围选“当前分析”。输入：查看V2最后1.2秒，给出起止采样点、均值和最大值。确认网关授权后自行发送。','answer_saved'),
 ('核对波形','在成功回答下点击“查看波形”，核对V2及[4200,4800)。若该回答没有窗口证据，可返回上一步重新提问。','wave_open'),
 ('完成','已走完记录→看图→提问→波形复核。仍需自行检查回答是否正确。','done')]
BATCH=[('打开批量复核','侧栏入口选择“批量复核”，选择评测报告。','batch_open'),
 ('筛选记录','选择固定筛选、误报、分数降序，检查结果表。完成后点击“已完成本步”。','confirm'),
 ('进入单条分析','选择记录，点击“打开单条分析并继续问答”。若无匹配历史，会进行本地分析；失败时仍停在本步。','analysis_open'),
 ('继续复核','进入“向本记录提问”，查询范围选“当前分析”，自行提交一个问题。成功保存后完成。','answer_saved'),
 ('完成','可从侧栏返回批量复核。原报告与记录保留。','done')]
_SLOT=None


def advance(state,event):
    if not state or not state.get('active'):return False
    steps=BATCH if state['route']=='batch' else SINGLE
    if steps[state['step']][2]!=event or event=='done':return False
    state['step']=min(state['step']+1,len(steps)-1)
    return True


def draw(st):
    if _SLOT is None:return
    s=st.session_state.get('ecg_step_guide')
    if not s or not s['active']:_SLOT.empty();return
    steps=BATCH if s['route']=='batch' else SINGLE
    title,body,_=steps[s['step']]
    with _SLOT.container():
        st.markdown(f"**使用指引 · 第{s['step']+1}/{len(steps)}步：{title}**")
        st.progress((s['step']+1)/len(steps))
        st.info(body)
        st.caption('可以退出或跳过。指引不会自动分析，也不会替你向网关发送问题。')


def note(st,event):
    if advance(st.session_state.get('ecg_step_guide'),event):draw(st)


def render(st,history):
    global _SLOT
    from src.ui.ecg_workspace import guide_seen,mark_guide_seen
    if not guide_seen(history):
        mark_guide_seen(history)
        st.session_state['ecg_step_guide']={'active':True,'route':'single','step':0}
    with st.sidebar:
        if st.button('开始 / 重新打开步骤指引',key='step_guide_launch'):
            st.session_state['ecg_step_guide']={'active':True,'route':'single','step':0}
        s=st.session_state.get('ecg_step_guide')
        if s and s['active']:
            route=st.selectbox('指引路线',['single','batch'],index=0 if s['route']=='single' else 1,
                format_func=lambda x:'单条ECG复核' if x=='single' else '批量错分复核',key='step_guide_route_'+s['route'])
            if route!=s['route']:s.update(route=route,step=0);st.rerun()
            cols=st.columns(2)
            if cols[0].button('上一步',disabled=s['step']==0):s['step']-=1;st.rerun()
            if cols[1].button('跳过本步',disabled=s['step']==4):s['step']=min(4,s['step']+1);st.rerun()
            steps=BATCH if s['route']=='batch' else SINGLE
            if steps[s['step']][2]=='confirm' and st.button('已完成本步'):note(st,'confirm');st.rerun()
            if st.button('结束指引' if s['step']==4 else '退出指引'):
                s['active']=False;st.rerun()
        _SLOT=st.empty()
        draw(st)
