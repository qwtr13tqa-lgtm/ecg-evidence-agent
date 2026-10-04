"""Run: python -m streamlit run app_batch_review.py --server.port 8506"""
import json,time,hashlib
from pathlib import Path
import streamlit as st
from src.review.batch import validate,query,signal,ask
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='ECG 批量错分复核',layout='wide')
st.title('ECG 批量错分复核')
st.caption('开发评测记录复核；标签不等于临床确诊。固定查询与自然语言查询共用同一工具。')
files=sorted((ROOT/'evaluation/website_path_reports').glob('*.json'))
if not files:st.info('请将评测报告放到 evaluation/website_path_reports。');st.stop()
p=st.selectbox('评测报告',files,format_func=lambda x:x.name)
try:
    content=p.read_bytes();report=validate(json.loads(content))
except Exception as exc:st.error('报告无效：'+str(exc));st.stop()
batch=hashlib.sha256(content).hexdigest()
counts={k:len(query(report,k)) for k in ('tn','fp','fn','tp')}
st.write({'记录数':len(report['rows']),'正常判对':counts['tn'],'误报':counts['fp'],'漏报':counts['fn'],'异常检出':counts['tp']})
st.caption(f"本批阈值 {report['threshold']}；规则：原始分数 ≥ 阈值预测异常。")
with st.expander('来源与版本'):
    st.json({k:v for k,v in report.items() if k not in ('rows','metrics')})
mode=st.radio('查询方式',['固定筛选','自然语言查询'],horizontal=True)
rows=[];tag='fixed'
if mode=='固定筛选':
    category=st.selectbox('筛选类型',['fp','fn','tp','tn','all'],format_func=lambda x:{'fp':'误报','fn':'漏报','tp':'异常检出','tn':'正常判对','all':'全部'}[x])
    order=st.selectbox('分数排序',['desc','asc'],format_func=lambda x:'从高到低' if x=='desc' else '从低到高')
    rows=query(report,category,order)
else:
    st.caption('当前是单轮工具参数解析与受限执行，不是多步自主规划。只发送问题和工具定义，不发送ECG或整份报告。')
    question=st.text_input('问题',placeholder='找出标签正常但预测异常的记录，按分数从高到低排序')
    consent=st.checkbox('允许将问题发送到已配置网关')
    if st.button('查询',disabled=not consent or not question.strip()):
        gateway=None;t=time.perf_counter()
        try:
            from src.agent.gateway import ToolGateway
            gateway=ToolGateway(timeout=60,max_tokens=300)
            answer=ask(report,question,gateway)
            answer.update(batch_sha256=batch,question=question,elapsed_seconds=time.perf_counter()-t)
            st.session_state['batch_answer_'+batch]=answer
        except Exception as exc:
            st.session_state.pop('batch_answer_'+batch,None)
            st.error('查询失败：'+type(exc).__name__+'；可改用固定筛选，无自动重试。')
        finally:
            if gateway is not None:gateway.close()
    answer=st.session_state.get('batch_answer_'+batch)
    if answer:
        st.write('已执行的问题：'+answer['question'])
        if answer['status']=='completed':
            rows=answer['rows'];st.write('实际执行参数：',answer['arguments'])
            st.caption(f"模型请求1次；查询1次；耗时{answer['elapsed_seconds']:.2f}秒。请核对执行条件是否符合原问题。")
            st.download_button('下载本次查询记录',json.dumps(answer,ensure_ascii=False,indent=2),file_name='batch_query.json')
        else:st.info(answer['message'])
    tag='natural'
st.write(f'结果：{len(rows)} 条')
if rows:
    st.dataframe([{k:r[k] for k in ('index','label','prediction','score','threshold')} for r in rows],hide_index=True)
    chosen=st.selectbox('选择记录',range(len(rows)),format_func=lambda i:f"样本 {rows[i]['index']} · 分数 {rows[i]['score']:.6f}",key=batch+tag)
    row=rows[chosen]
    path=st.text_input('本地数据文件（须与报告哈希匹配）',value=str(ROOT/'data/Processed_PTBXL/test.npy'))
    lead=st.selectbox('导联',['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],index=1)
    if st.button('查看记录波形'):
        try:
            raw=signal(report,row,path)
            import numpy as np
            import plotly.graph_objects as go
            leads=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
            fig=go.Figure(go.Scatter(x=np.arange(4800)/500,y=raw[:,leads.index(lead)],name=lead))
            fig.update_layout(title=f"样本 {row['index']} · {lead}",xaxis_title='裁剪片段时间 / 秒',yaxis_title='保存的输入幅值（单位未核实）')
            st.plotly_chart(fig)
            st.caption('已核验数据文件及片段哈希。这里只展示报告对应输入波形；没有重新推理，也未生成重构或候选区域。')
        except Exception as exc:st.error(str(exc))
