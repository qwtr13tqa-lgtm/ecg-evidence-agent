"""Transactional deletion of one analysis and its dependent history."""

def delete_analysis(history,aid):
    if not isinstance(aid,str) or not aid:raise ValueError('无效分析ID')
    with history.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute('SELECT 1 FROM analyses WHERE analysis_id=?',(aid,)).fetchone() is None:
            return False
        pending=db.execute("SELECT 1 FROM turns t JOIN conversations c ON c.conversation_id=t.conversation_id WHERE c.analysis_id=? AND t.state='pending' LIMIT 1",(aid,)).fetchone()
        if pending:raise ValueError('该记录有待完成请求，请先结束本地等待后再删除。')
        db.execute('DELETE FROM turns WHERE conversation_id IN (SELECT conversation_id FROM conversations WHERE analysis_id=?)',(aid,))
        db.execute('DELETE FROM conversations WHERE analysis_id=?',(aid,))
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ecg_ui_analysis_meta'").fetchone():
            db.execute('DELETE FROM ecg_ui_analysis_meta WHERE analysis_id=?',(aid,))
        db.execute('DELETE FROM analyses WHERE analysis_id=?',(aid,))
    return True


def render_delete(st,history,aid,name):
    with st.expander('删除这条分析记录'):
        st.write(f'将删除：{name}（{aid}）')
        st.caption('删除该分析保存的波形、模型结果和所有关联对话。原始数据文件、评测报告及其他分析不受影响。此操作不能在界面撤销；其他对话中引用它的历史链接会失效。')
        with st.form('delete_analysis_'+aid):
            confirm=st.checkbox('确认删除该分析及关联对话',key='delete_confirm_'+aid)
            submitted=st.form_submit_button('永久删除')
        if submitted:
            if not confirm:st.warning('请先勾选确认。');return
            try:deleted=delete_analysis(history,aid)
            except Exception as exc:
                st.error(str(exc) if isinstance(exc,ValueError) else '删除失败，事务已回滚：'+type(exc).__name__);return
            if st.query_params.get('analysis')==aid:
                for k in ('analysis','conversation'):
                    if k in st.query_params:del st.query_params[k]
                for k in ('active','conversation','bundle','answer','question_input'):
                    st.session_state.pop(k,None)
            for k in list(st.session_state):
                if aid in str(k):st.session_state.pop(k,None)
            st.rerun()
