"""Offline fine-grained extension for the existing threeway dashboard."""
import json
import pandas as pd
from src.evaluation.fine_metrics import analyze,aggregate,read_review,save_review,VERDICTS


def render(st,rows):
    st.divider();st.subheader('细粒度评分与冗余查询')
    st.caption('目标字段覆盖率使用已有自动检查；不评价自由文本同义表达。QTc缺失题无数值字段，标为不适用，不算0分或100分。')
    items=[];saved={}
    for row in rows:
        try:saved[row['run_id']]=read_review(row['path'])
        except ValueError as exc:st.warning(str(exc));saved[row['run_id']]={}
        items.append(analyze(row,saved[row['run_id']]))
    if not items:st.info('当前筛选无记录');return
    summary=pd.DataFrame(aggregate(items));a,b=st.columns(2)
    with a:
        st.markdown('**目标字段覆盖率**')
        st.bar_chart(summary.set_index('scheme')[['field_coverage']])
        st.dataframe(summary[['scheme','field_correct','field_total','unscored_field_runs']],hide_index=True)
    with b:
        st.markdown('**查询调用分类（不含共同初始摘要和提交答案）**')
        st.bar_chart(summary.set_index('scheme')[['necessary_calls','redundant_calls','failed_calls','pending_calls']])
    st.caption('分类互斥：必要、冗余、失败、待核查。已确认冗余率为下界，加上待核查为上界；无查询时不适用。查询失败与冗余不重复计数。工具请求记录不等于实际计算次数；缓存命中和token旧记录未提供，不能推断。')
    st.dataframe(summary,hide_index=True,use_container_width=True)
    detail=pd.DataFrame([{k:v for k,v in x.items() if k not in ('calls','claims')} for x in items])
    st.dataframe(detail,hide_index=True,use_container_width=True)
    st.download_button('下载细粒度指标 CSV',detail.to_csv(index=False).encode('utf-8-sig'),'fine_metrics.csv','text/csv',key='fine_csv')
    st.download_button('下载调用分类及评分 JSON',json.dumps(items,ensure_ascii=False,indent=2),'fine_metrics.json','application/json',key='fine_json')
    with st.expander('复核调用与文字陈述（独立保存，不改变原有pass/fail）'):
        rid=st.selectbox('选择运行',list(saved),format_func=lambda rid:next(f"{r['case']} / {r['scheme']} / 第{r['turn']}轮" for r in rows if r['run_id']==rid),key='fine_run')
        row=next(r for r in rows if r['run_id']==rid);item=next(x for x in items if x['run_id']==rid);old=saved[rid]
        st.write(row['record']['case']['question']);st.write(row['record']['output'].get('draft',{}).get('answer',''))
        st.json(row['record']['automatic'].get('field_checks',{}))
        with st.expander('完整本轮证据'):st.json(row['record']['output'].get('evidence',{}))
        st.dataframe(pd.DataFrame(item['calls']),use_container_width=True)
        st.caption('默认只自动确认重复摘要/相同入参重复请求与执行失败。模型判定是否无关、空检索是否冗余需结合任务人工判断，不能把no_match一律算错。')
        with st.form('fine_form_'+rid):
            reviewer=st.text_input('复核人',value=old.get('reviewer',''))
            decisions={}
            for c in item['calls']:
                st.write(f"轨迹 {c['trace_index']} · {c['tool']} · {c['arguments']}")
                value=st.selectbox('分类',VERDICTS,index=VERDICTS.index(c['verdict']),key=f"fine_v_{rid}_{c['trace_index']}")
                note=st.text_input('判断理由',value=c['note'],key=f"fine_n_{rid}_{c['trace_index']}")
                if note.strip() or value!=c['verdict']:decisions[str(c['trace_index'])]={'verdict':value,'note':note}
            st.caption('文字支持率需逐条列出可核查陈述并复核。JSON格式如下；basis填写证据ID/路径或缺失判断依据。只评部分陈述时不要勾选完整覆盖。')
            st.code('[{"claim":"平均RR约0.986秒","verdict":"supported","basis":"证据ID /calculation_facts/mean_rr_seconds"}]',language='json')
            claims_text=st.text_area('文字陈述评分 JSON',value=json.dumps(old.get('claims',[]),ensure_ascii=False,indent=2),height=160)
            complete=st.checkbox('已覆盖答案中全部可核查陈述',value=old.get('claims_complete',False))
            st.caption('文字支持率只以已裁定陈述为分母，同时展示未评数量和完整覆盖标记。缺少标注时不显示100%。')
            if st.form_submit_button('保存细粒度复核'):
                try:
                    claims=json.loads(claims_text)
                    if not isinstance(claims,list):raise ValueError('Claims must be a JSON list')
                    save_review(row['path'],reviewer,decisions,claims,complete);st.rerun()
                except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
