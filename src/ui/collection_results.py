"""Compact views of saved collection results, including partial failures."""
from .batch_display import display_rows, comparison, assumptions, CATEGORIES, LABELS

def render(st,out,key):
    entries=[e for e in out.get('evidence',{}).values() if e.get('scope')=='batch_collection']
    if not entries:return
    st.subheader('统计与记录证据')
    st.caption('本地计算快照；请核对筛选条件。缺失值不按零计算，各字段有效数可能不同。')
    combined=comparison(entries)
    if combined:
        st.write('**FN / TP 对比 · 中位数与覆盖情况**')
        st.dataframe(combined,use_container_width=True,hide_index=True)
    grouped=[e for e in entries if 'groups' in e.get('data',{})]
    if grouped:
        with st.expander('完整分组统计',expanded=not bool(combined)):
            for i,e in enumerate(grouped):
                d=e['data'];flat=[]
                st.caption(f"统计 {i+1} · 匹配 {d.get('matched_n','—')} 条")
                for g in d['groups']:
                    for field,v in g.get('metrics',{}).items():
                        flat.append({'类别':CATEGORIES.get(g['group_value'],g['group_value']),'指标':LABELS.get(field,field),'组样本数':g['count'],**v})
                    if not g.get('metrics'):flat.append({'类别':g['group_value'],'数量':g['count'],'比例':g.get('fraction')})
                st.dataframe(display_rows(flat),use_container_width=True,hide_index=True)
    selected=[e for e in entries if 'rows' in e.get('data',{})]
    if selected:
        def title(i):
            d=selected[i]['data'];op=d.get('operation',{})
            cats=[str(f.get('value')) for f in op.get('filters',[]) if f.get('field')=='category']
            return f"{CATEGORIES.get(cats[0],cats[0]) if len(cats)==1 else '记录查询'} · 返回 {len(d['rows'])} 条 · 操作 {i+1}"
        i=st.selectbox('查看记录明细',range(len(selected)),format_func=title,key='collection_rows_'+key)
        d=selected[i]['data'];op=d.get('operation',{})
        st.caption(f"排序：{LABELS.get(op.get('order_by'),op.get('order_by') or '默认')} · {'降序' if op.get('direction')=='desc' else '升序'}；匹配 {d.get('matched_n','—')} 条，显示 {len(d['rows'])} 条")
        if d['rows']:st.dataframe(display_rows(d['rows']),use_container_width=True,hide_index=True)
        else:st.info('没有符合条件的记录。')
    for e in entries:
        d=e['data']
        if d.get('query_status')=='needs_clarification':st.info(d.get('message','请补充查询条件。'))
        if d.get('has_more'):st.warning('部分查询仅返回一页；完整匹配数量见查询条件。')
        if d.get('navigation_warning'):st.warning('部分历史入口未匹配，详见执行记录。')
    with st.expander('实际筛选条件与共同说明'):
        for a in assumptions(entries):st.write('• '+a)
        for i,e in enumerate(entries):
            st.write(f'操作 {i+1}')
            st.json(e['data'].get('operation',{}))
