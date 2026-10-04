"""Read-only presentation. No model calls, no edits to original output."""
TOOLS={'get_model_decision':'查询异常阈值判定','inspect_recent_rr_alignment':'末尾窗口与RR对齐','inspect_window_rr_alignment':'指定窗口与RR对齐','get_analysis_summary':'读取分析摘要','inspect_rr_intervals':'查询 RR 明细',
       'inspect_recent_error':'查询末尾误差窗口','inspect_error_window':'查询指定误差窗口',
       'search_knowledge':'检索知识','submit_answer':'提交回答'}

def fmt(value):
    if value is None:return '未提供'
    if type(value) is bool:return '是' if value else '否'
    if isinstance(value,float):return f'{value:.6g}'
    return str(value)

def evidence_rows(data):
    rows=[]
    def add(label,key,unit=''):
        if key in data:rows.append({'项目':label,'数值':fmt(data[key]),'单位':unit})
    if 'prediction' in data:
        add('异常检测分数','score');add('阈值','threshold');add('比较规则','comparator')
        add('判定状态','status');add('模型预测','prediction');add('原因','reason')
    elif 'window' in data:
        rows.extend(evidence_rows(data['window']))
        add('RR提取导联','rhythm_lead');add('重叠RR数','overlapping_rr_count');add('RR核算状态','rr_status')
    elif 'intervals' in data:
        add('原始 RR 数','total_intervals');add('保留数','retained_count');add('排除数','excluded_count')
        facts=data.get('calculation_facts') or {}
        for k,label,unit in [('candidate_peak_count','候选峰数',''),('mean_rr_seconds','平均 RR','秒'),('heart_rate_bpm','估计心率','bpm')]:
            if k in facts:rows.append({'项目':label,'数值':fmt(facts[k]),'单位':unit})
    elif 'leads' in data:
        add('窗口起点','start_seconds','秒');add('窗口终点','end_seconds','秒')
        for lead in data['leads']:
            for k,label in [('mean','均值'),('maximum','最大值'),('minimum','最小值'),('peak_sample','首个最大值采样点')]:
                if k in lead:rows.append({'项目':str(lead.get('lead',''))+' '+label,'数值':fmt(lead[k]),'单位':'采样点' if k=='peak_sample' else '模型原始值'})
    elif 'input' in data:
        model=data.get('model') or {};rhythm=(data.get('signal_features') or {}).get('rhythm') or {}
        for label,value,unit in [('片段时长',data['input'].get('duration_seconds'),'秒'),('模型原始分数',model.get('anomaly_score'),''),('估计心率',rhythm.get('heart_rate_bpm'),'bpm')]:
            rows.append({'项目':label,'数值':fmt(value),'单位':unit})
    elif 'documents' in data:rows.append({'项目':'返回知识条目数','数值':str(len(data['documents'])),'单位':''})
    return rows

def partition(output):
    draft=output.get('draft') or {}
    def split(pool,ids):
        return ([(i,pool[i]) for i in ids if i in pool],[(i,v) for i,v in pool.items() if i not in ids])
    e,u=split(output.get('evidence') or {},draft.get('evidence_ids') or [])
    k,v=split(output.get('knowledge') or {},draft.get('knowledge_ids') or [])
    return e,u,k,v

def trace_rows(output):
    rows=[]
    for i,t in enumerate(output.get('trace') or [],1):
        stage=t.get('stage'); name=TOOLS.get(t.get('tool'),t.get('tool','工具'))
        label='模型请求' if stage=='model' else name
        args=t.get('arguments') or {}
        detail='；'.join(f'{k}={fmt(v)}' for k,v in args.items())
        if t.get('error_type') or t.get('error'):status='失败'
        elif stage=='model':status='已返回'
        elif t.get('ok') is True:status='成功'
        elif t.get('ok') is False:status='失败'
        else:status='未记录'
        rows.append({'顺序':i,'步骤':label,'参数':detail or '—','状态':status,
                     '耗时/秒':fmt(t.get('elapsed_seconds')),'错误类型':t.get('error_type') or t.get('error') or '—'})
    return rows

def render_answer_details(st,output):
    cited,other,knowledge,unused=partition(output)
    def card(eid,item):
        data=item.get('data') or {}
        title='RR 计算证据' if 'intervals' in data else '误差窗口证据' if 'leads' in data else '分析摘要' if 'input' in data else '检索结果'
        if 'prediction' in data:title='模型阈值判定'
        if 'window' in data:title='模型窗口与RR时间对齐'
        st.markdown('**'+title+'**');st.caption('证据 ID：'+eid)
        rows=evidence_rows(data)
        if rows:st.dataframe(rows)
        if 'intervals' in data:
            st.caption('当前返回 '+str(len(data['intervals']))+' 条 RR；原始间隔总数 '+fmt(data.get('total_intervals'))+'。坐标相对裁剪输入。')
            st.dataframe([{'序号':r.get('rr_index'),'左峰/采样点':r.get('left_peak_sample'),'右峰/采样点':r.get('right_peak_sample'),
                           'RR/秒':r.get('rr_seconds'),'保留':fmt(r.get('retained')),'排除原因':r.get('exclusion_reason') or '无'} for r in data['intervals']])
        if 'window' in data:
            st.dataframe(data.get('rr_intervals',[]))
            st.caption('只核查时间重叠；不构成医学交叉验证。')
        if 'leads' in data:st.caption('区间为左闭右开；模型误差数值不是概率。最大值采样点不是确认的疾病事件。')
    def knowledge_card(kid,doc):
        st.markdown('**知识条目：'+kid+'**')
        st.text(doc.get('title',''));st.text(doc.get('text',''))
        st.caption('来源：'+str(doc.get('source','未提供')))
        st.caption('定位：'+str(doc.get('locator','未提供')))
    st.subheader('回答引用的证据')
    if not cited:st.caption('本次没有提交证据引用。')
    for eid,item in cited:card(eid,item)
    st.subheader('回答引用的知识')
    if not knowledge:st.caption('本次未引用外部知识；这不表示回答已获知识验证。')
    for kid,doc in knowledge:knowledge_card(kid,doc)
    if other or unused:
        with st.expander('已查询但未引用的资料'):
            for eid,item in other:card(eid,item)
            for kid,doc in unused:knowledge_card(kid,doc)
    st.subheader('调用流程')
    rows=trace_rows(output)
    if rows:st.dataframe(rows)
    else:st.caption('没有调用记录。')
    st.subheader('校验状态')
    val=output.get('validation') or {}
    if output.get('status')=='completed_draft' and val.get('passed') is True:
        st.success('结构、复制数值和引用检查通过（程序记录）。')
    else:st.warning('没有已通过校验的完整草稿，请查看调用状态。')
    st.caption('自由文本语义：未自动验证。医学正确性：未验证。引用存在不代表支持完整解释。')
    for item in output.get('limitations') or []:st.write('• '+str(item.get('message','')))
    with st.expander('开发调试信息（完整 JSON）'):st.json(output)
