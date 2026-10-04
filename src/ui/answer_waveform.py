"""Resolve chart targets only from cited, successful current-analysis tools."""
def targets(output, analysis_id, num_samples):
    if (output.get('analysis_id') != analysis_id or output.get('status') != 'completed_draft'
            or output.get('validation', {}).get('passed') is not True):
        return []
    cited = (output.get('draft') or {}).get('evidence_ids', [])
    evidence = output.get('evidence') or {}
    valid = {t.get('evidence_id') for t in output.get('trace', [])
             if t.get('stage') == 'tool' and t.get('ok') is True
             and t.get('tool') in ('inspect_error_window', 'inspect_recent_error')}
    found = {}
    for eid in cited:
        e = evidence.get(eid, {})
        if eid not in valid or e.get('analysis_id') != analysis_id or e.get('ok') is not True:
            continue
        d = e.get('data') or {}
        a, b = d.get('start_sample'), d.get('end_sample')
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= num_samples:
            continue
        for row in d.get('leads', []):
            lead = row.get('lead')
            if lead not in ('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'):
                continue
            found.setdefault((lead,a,b), dict(analysis_id=analysis_id, lead=lead,
                start_sample=a, end_sample=b, evidence_id=eid))
    return list(found.values())


def render_answer_waveforms(st, output, result, raw, key):
    windows = targets(output, result.analysis_id, len(raw))
    if not windows:
        return
    # Render beside the originating answer; no tab switching or extra inference.
    selected = st.selectbox('本轮证据窗口', range(len(windows)),
        format_func=lambda i: f"{windows[i]['lead']} · [{windows[i]['start_sample']}, {windows[i]['end_sample']})",
        key=key+'_target') if len(windows)>1 else 0
    w = windows[selected]
    state_key = key+'_visible'
    guide_clicked=st.button('查看波形', key=key+'_show')
    if guide_clicked:
        st.session_state[state_key] = (w['lead'],w['start_sample'],w['end_sample'])
    if st.session_state.get(state_key) != (w['lead'],w['start_sample'],w['end_sample']):
        return
    from src.ui.reconstruction_view import figures
    try:
        fs = result.input.sampling_rate
        fig, *_ = figures(raw,result.model.reconstruction,result.model.error_map,w['lead'],fs,[])
        a,b = w['start_sample'],w['end_sample']
        for row in (1,2,3):
            fig.add_vrect(x0=a/fs,x1=b/fs,fillcolor='#f59e0b',opacity=.2,line_width=0,row=row,col=1)
        pad=max(.2,(b-a)/fs*.25)
        fig.update_xaxes(range=[max(0,a/fs-pad),min(len(raw)/fs,b/fs+pad)])
        fig.update_layout(height=570,title_text=f"{w['lead']} · [{a},{b}) · 本轮查询窗口")
        st.plotly_chart(fig,key=key+'_chart')
        from src.ui.step_guide import note
        if guide_clicked:note(st,'wave_open')
        st.caption(f"证据：{w['evidence_id']}。黄色为查询窗口；背景保留少量上下文。模型分数不是概率。")
    except (ValueError,TypeError,AttributeError) as exc:
        st.warning('保存的波形暂不能显示：'+str(exc))
