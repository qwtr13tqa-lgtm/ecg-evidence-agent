"""Local-only history browsing and first-use help; no inference or model calls."""
import math


def initialize(history):
    with history.connect() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS ecg_ui_analysis_meta(
            analysis_id TEXT PRIMARY KEY REFERENCES analyses(analysis_id),
            title TEXT NOT NULL DEFAULT '', archived INTEGER NOT NULL DEFAULT 0)''')
        db.execute('''CREATE TABLE IF NOT EXISTS ecg_ui_preferences(
            key TEXT PRIMARY KEY, value TEXT NOT NULL)''')


def guide_seen(history):
    with history.connect() as db:
        row = db.execute("SELECT value FROM ecg_ui_preferences WHERE key='guide_v1'").fetchone()
    return bool(row and row[0] == 'seen')


def mark_guide_seen(history):
    with history.connect() as db:
        db.execute("INSERT OR REPLACE INTO ecg_ui_preferences VALUES('guide_v1','seen')")


def update_record(history, aid, *, title=None, archived=None):
    if title is not None and (not isinstance(title, str) or len(title.strip()) > 100):
        raise ValueError('Title must be at most 100 characters')
    if archived is not None and type(archived) is not bool:
        raise ValueError('Invalid archive state')
    with history.connect() as db:
        if not db.execute('SELECT 1 FROM analyses WHERE analysis_id=?', (aid,)).fetchone():
            raise KeyError('Analysis not found')
        db.execute('INSERT OR IGNORE INTO ecg_ui_analysis_meta(analysis_id) VALUES(?)', (aid,))
        if title is not None:
            db.execute('UPDATE ecg_ui_analysis_meta SET title=? WHERE analysis_id=?', (title.strip(), aid))
        if archived is not None:
            db.execute('UPDATE ecg_ui_analysis_meta SET archived=? WHERE analysis_id=?', (int(archived), aid))


def list_page(history, *, query='', sample=None, visibility='active', page=1, size=10):
    if type(page) is not int or page < 1 or type(size) is not int or not 1 <= size <= 50:
        raise ValueError('Invalid page')
    if visibility not in ('active', 'archived', 'all'):
        raise ValueError('Invalid visibility')
    if sample is not None and (type(sample) is not int or sample < 0):
        raise ValueError('Invalid sample index')
    where, params = ['1=1'], []
    if visibility != 'all':
        where.append('COALESCE(m.archived,0)=?')
        params.append(int(visibility == 'archived'))
    if sample is not None:
        where.append('a.sample_index=?'); params.append(sample)
    if query.strip():
        where.append("(instr(lower(a.analysis_id),lower(?))>0 OR instr(lower(COALESCE(m.title,'')),lower(?))>0)")
        params += [query.strip(), query.strip()]
    base = ' FROM analyses a LEFT JOIN ecg_ui_analysis_meta m ON a.analysis_id=m.analysis_id WHERE ' + ' AND '.join(where)
    with history.connect() as db:
        total = db.execute('SELECT count(*)' + base, params).fetchone()[0]
        pages = max(1, math.ceil(total / size)); page = min(page, pages)
        rows = db.execute('''SELECT a.analysis_id,a.created_at,a.sample_index,a.elapsed,
            COALESCE(m.title,'') AS title,COALESCE(m.archived,0) AS archived,
            (SELECT count(*) FROM conversations c JOIN turns t USING(conversation_id)
             WHERE c.analysis_id=a.analysis_id) AS turn_count''' + base +
            ' ORDER BY a.created_at DESC,a.analysis_id DESC LIMIT ? OFFSET ?',
            params + [size, (page-1)*size]).fetchall()
    return {'total': total, 'page': page, 'pages': pages, 'rows': [dict(r) for r in rows]}


def render_help(st, history):
    from src.ui.step_guide import render
    render(st,history)


def render_history(st, history, open_record):
    st.subheader('历史分析记录')
    st.caption('按样本或名称查找记录。这里显示分析创建时间，不代表心电采集时间。归档可以恢复，不删除原始数据。')
    with st.form('ecg_history_filters'):
        cols = st.columns(3)
        query = cols[0].text_input('搜索记录名称 / 分析ID', max_chars=100, placeholder='记录名称或ID片段，如117d9482', help='留空不限名称；与样本索引同时填写时需同时匹配。')
        sample_text = cols[1].text_input('数据样本索引（可选）', max_chars=12, placeholder='例如10；上传记录请留空', help='这是数据中的索引，不是患者编号。留空不限制索引。')
        visibility_label = cols[2].selectbox('记录范围', ['未归档', '已归档', '全部'])
        if st.form_submit_button('筛选'):
            if sample_text.strip() and (not sample_text.strip().isascii() or not sample_text.strip().isdigit()):
                st.error('样本编号须为非负整数。')
            else:
                st.session_state['ecg_history_filter'] = {
                    'query': query, 'sample': int(sample_text) if sample_text.strip() else None,
                    'visibility': {'未归档':'active','已归档':'archived','全部':'all'}[visibility_label]}
                st.session_state['ecg_history_page'] = 1
    filters = st.session_state.get('ecg_history_filter', {})
    data = list_page(history, page=st.session_state.get('ecg_history_page', 1), **filters)
    st.session_state['ecg_history_page'] = data['page']
    prev, label, nxt = st.columns([1,3,1])
    if prev.button('上一页', disabled=data['page'] <= 1):
        st.session_state['ecg_history_page'] -= 1; st.rerun()
    label.write(f"共 {data['total']} 条 · 第 {data['page']} / {data['pages']} 页 · 每页10条")
    if nxt.button('下一页', disabled=data['page'] >= data['pages']):
        st.session_state['ecg_history_page'] += 1; st.rerun()
    if not data['rows']:
        st.info('没有符合筛选条件的记录。'); return
    for row in data['rows']:
        aid = row['analysis_id']
        with st.container(border=True):
            title, opened, archive = st.columns([6,1,1])
            name = row['title'] or f"样本 {row['sample_index']}"
            title.write(name + (' · 已归档' if row['archived'] else ''))
            title.caption(f"样本 {row['sample_index']} · {row['created_at'][:19].replace('T',' ')} UTC · {row['turn_count']}轮对话 · {aid[:8]}")
            if opened.button('打开', key='ecg_open_' + aid):
                open_record(aid)
                st.session_state['ecg_history_opened'] = True
                st.rerun()
            if archive.button('恢复' if row['archived'] else '归档', key='ecg_archive_' + aid):
                update_record(history, aid, archived=not bool(row['archived'])); st.rerun()
            # HISTORY_DELETE_V1
            from src.ui.history_delete import render_delete
            render_delete(st,history,aid,name)
            with st.expander('重命名 / 查看完整ID'):
                st.code(aid)
                with st.form('ecg_rename_' + aid):
                    value = st.text_input('记录名称', value=row['title'], max_chars=100)
                    if st.form_submit_button('保存名称'):
                        update_record(history, aid, title=value); st.rerun()
