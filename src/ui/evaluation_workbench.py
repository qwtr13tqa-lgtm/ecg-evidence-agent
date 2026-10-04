"""Four-page local evaluation workbench. No gateway or inference calls."""
import json
from pathlib import Path
import pandas as pd
from src.evaluation.workbench import METRICS,NAMES,snapshot,issues,paired,timing
from src.evaluation.threeway_metrics import summarize,conversation_outcomes,save_review as save_task
from src.evaluation.records import read_review as task_review
from src.evaluation.fine_metrics import aggregate,read_review as fine_review,save_review as save_fine,analyze,VERDICTS


def guide(st,keys):
    with st.expander('指标解释 · 异常时如何定位与优化'):
        key=st.selectbox('指标',keys,format_func=lambda k:METRICS[k][0],key='help_'+keys[0])
        name,formula,meaning,cause,action=METRICS[key]
        st.markdown('**口径**：'+formula);st.markdown('**如何判断**：'+meaning)
        st.markdown('**排查与优化方向**：'+cause);st.markdown('**入口**：'+action)


def pretty_rate(value):return '不适用' if value is None else f'{value:.1%}'


def status_chart(st,summaries):
    rows=[{'方案':NAMES[s['scheme']],'状态':name,'数量':s[key]} for s in summaries for key,name in [('pass','已通过'),('fail','失败'),('pending','待复核')]]
    st.vega_lite_chart(pd.DataFrame(rows),{
        'mark':'bar','height':190,
        'encoding':{'x':{'field':'方案','type':'nominal','axis':{'labelAngle':0}},
        'y':{'field':'数量','type':'quantitative','scale':{'zero':True},'axis':{'tickMinStep':1}},
        'color':{'field':'状态','type':'nominal','scale':{'domain':['已通过','失败','待复核'],'range':['#2a9d8f','#d95d55','#e5b33e']}},
        'tooltip':[{'field':'方案'},{'field':'状态'},{'field':'数量'}]}},use_container_width=True)


def table(st,data):
    if data:st.dataframe(pd.DataFrame(data),hide_index=True,use_container_width=True)
    else:st.info('当前没有记录。')


def main():
    import streamlit as st
    root=Path(__file__).resolve().parents[2]
    st.set_page_config(page_title='ECG 评测与优化工作台',layout='wide')
    st.title('ECG 评测与优化工作台')
    st.caption('先看完成质量，再定位失败和冗余，最后做同题前后对照。所有操作在本地，不发起模型请求。')
    # CONDITIONAL_BATCH_EVAL_V1
    page=st.sidebar.radio('工作步骤',['实验总览','问题定位','逐题复核','优化前后对照','批量条件补查'],key='workbench_page')
    if page=='批量条件补查':
        from src.ui.conditional_batch_ui import render_evaluation
        render_evaluation(st,root)
        return
    runs=root/'evaluation/threeway_runs'
    manifests=sorted(runs.glob('*.manifest.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not manifests:st.info('尚无评测记录。先运行 evaluation.run_threeway。');return
    batches=[p.name.removesuffix('.manifest.json') for p in manifests]
    batch=st.sidebar.selectbox('当前实验批次',batches)
    try:manifest,rows,items,warnings=snapshot(runs,batch)
    except Exception as exc:st.error('无法加载批次：'+str(exc));return
    st.sidebar.caption(batch)
    schemes=st.sidebar.multiselect('方案',list(NAMES),default=sorted({r['scheme'] for r in rows}),format_func=lambda k:NAMES[k])
    kinds=st.sidebar.multiselect('题型',sorted({r['kind'] for r in rows}),default=sorted({r['kind'] for r in rows}))
    selected=[r for r in rows if r['scheme'] in schemes and r['kind'] in kinds]
    selected_ids={r['run_id'] for r in selected};fine=[x for x in items if x['run_id'] in selected_ids]
    index={r['run_id']:r for r in rows};findex={x['run_id']:x for x in items}
    for w in warnings:st.warning(w)
    st.caption(f'当前批次 {len(rows)} 条已落盘轮次 · 当前筛选 {len(selected)} 条 · 方案：'+ ' / '.join(NAMES[s] for s in schemes))
    if not selected:st.info('请调整筛选，或等待运行结果落盘。');return
    if page=='实验总览':
        st.subheader('1. 本批完成到哪一步')
        counts={s:sum(r['outcome']==s for r in selected) for s in ('pass','fail','pending')}
        c=st.columns(4)
        for col,(label,value) in zip(c,[('已落盘轮次',len(selected)),('已确认通过',counts['pass']),('失败',counts['fail']),('待复核',counts['pending'])]):col.metric(label,value)
        if counts['pending']:st.info('仍有待复核记录：当前确认成功率只是下界。请到“逐题复核”保存评分。待复核使用黄色，不表示失败。')
        status_chart(st,summarize(selected))
        st.subheader('2. 质量与成本对照')
        fine_summary={x['scheme']:x for x in aggregate(fine)};view=[]
        for s in summarize(selected):
            f=fine_summary[s['scheme']]
            view.append({'方案':NAMES[s['scheme']],'任务成功率下界':pretty_rate(s['confirmed_success_rate']),
                '任务成功率可能上界':pretty_rate(s['possible_success_rate']),
                '目标字段覆盖':f"{f['field_correct']}/{f['field_total']}" if f['field_total'] else '不适用',
                '无字段标准轮次':f['unscored_field_runs'],'数值与引用通过率':pretty_rate(s['structure_rate']),
                '补充查询合计':f['query_requests'],'确认冗余 / 待核查':f"{f['redundant_calls']} / {f['pending_calls']}",
                '模型请求合计':f['model_requests']})
        table(st,view)
        st.caption('字段覆盖与整题成功是不同口径。无字段标准的缺失题仍参与任务成功率；文字证据支持未完整复核时不生成全文得分。')
        claims_view=[]
        for scheme in schemes:
            group=[x for x in fine if x['scheme']==scheme]
            if not group:continue
            complete=all(x['claims_complete'] for x in group)
            supported=sum(x['supported_claims'] for x in group);unsupported=sum(x['unsupported_claims'] for x in group)
            claims_view.append({'方案':NAMES[scheme],'全文复核完成':str(sum(x['claims_complete'] for x in group))+'/'+str(len(group)),
                '文字证据支持率':pretty_rate(supported/(supported+unsupported)) if complete and supported+unsupported else '未完整复核',
                '已标注不支持陈述':unsupported})
        table(st,claims_view)
        st.subheader('3. 耗时与可靠性')
        table(st,[{'方案':NAMES[s['scheme']],'耗时中位数（秒）':round(s['median_seconds'],2) if s['median_seconds'] is not None else None,
            'P95（秒）':str(round(s['p95_seconds'],2)) if s['runs']>=20 and s['p95_seconds'] is not None else '样本不足20条，暂不展示',
            '样本量':s['runs'],'超时率':pretty_rate(s['timeout_rate'])} for s in summarize(selected)])
        st.caption('中位数与P95独立展示，不堆叠相加。耗时含失败等待，不含共享深度模型推理。token/缓存命中记录未提供，不推算。')
        with st.expander('整段会话结果（按当前选中案例展示全部轮次）'):
            selected_tasks={(r['case'],r['scheme'],r['repetition']) for r in selected}
            table(st,[x for x in conversation_outcomes(manifest,rows) if (x['case_id'],x['scheme'],x['repetition']) in selected_tasks])
        guide(st,['success','coverage','structure','redundancy','queries','requests','latency','timeout','claims','conversation'])
        st.download_button('下载本页指标 CSV',pd.DataFrame(view).to_csv(index=False).encode('utf-8-sig'),file_name=batch+'_overview.csv',mime='text/csv')
        with st.expander('实验配置与版本（高级）'):st.json(manifest['config'])
    elif page=='问题定位':
        st.subheader('按问题找到具体运行')
        findings=[x for row in selected for x in issues(row,findex[row['run_id']])]
        if not findings:st.success('未发现自动可识别的问题；这不替代文字证据复核。');return
        categories=sorted({x['category'] for x in findings})
        category=st.selectbox('问题类别',categories)
        subset=[x for x in findings if x['category']==category]
        st.caption('同一运行可以有多个问题标签，不应把各标签数量相加当作失败总数。')
        table(st,[{'案例':x['case'],'方案':NAMES[x['scheme']],'说明':x['detail'],'运行ID':x['run_id']} for x in subset])
        rid=st.selectbox('定位到运行',list(dict.fromkeys(x['run_id'] for x in subset)))
        row=index[rid];st.write(row['record']['case']['question'])
        st.json({'error':row['record']['output'].get('error'),'field_checks':row['record'].get('automatic',{}).get('field_checks'),
                 'reference_resolution':row['record']['output'].get('reference_resolution')})
        def open_review():
            st.session_state['workbench_target']=rid
            st.session_state['workbench_page']='逐题复核'
        st.button('打开此运行的逐题复核',type='primary',on_click=open_review)
        with st.expander('调用轨迹与异常详情'):st.json(row['record']['output'].get('trace',[]))
        guide(st,['coverage','structure','redundancy','requests','latency','timeout','conversation'])
    elif page=='逐题复核':
        st.subheader('一题一页：答案 → 证据 → 执行 → 评分')
        ids=[r['run_id'] for r in selected];target=st.session_state.get('workbench_target')
        rid=st.selectbox('案例 / 方案 / 重复 / 轮次',ids,index=ids.index(target) if target in ids else 0,
            format_func=lambda x:f"{index[x]['case']} / {NAMES[index[x]['scheme']]} / {index[x]['repetition']} / {index[x]['turn']}")
        row=index[rid];item=findex[rid];rec=row['record'];out=rec['output']
        st.caption('运行ID：'+rid);st.write('**问题：** '+rec['case']['question'])
        st.write((out.get('draft') or {}).get('answer') or '没有可用答案。')
        st.caption('当前结果：'+{'pass':'已通过','fail':'失败','pending':'待复核'}[row['outcome']])
        fields=rec.get('automatic',{}).get('field_checks',{})
        table(st,[{'目标字段':k,'自动检查':'通过' if v is True else '未通过' if v is False else '未判定'} for k,v in fields.items()]) if fields else st.info('本题没有自动字段标准，需要人工判断是否完成要求。')
        with st.expander('核对证据与结构化引用'):
            st.json(out.get('draft',{}));st.json(out.get('evidence',{}));st.json(rec.get('reference',{}))
        with st.expander('查看每一步执行与耗时'):
            table(st,[{'阶段':t.get('stage'),'工具/请求':str(t.get('tool',t.get('call',''))),'耗时（秒）':t.get('elapsed_seconds'),
                '参数':json.dumps(t.get('arguments'),ensure_ascii=False) if t.get('arguments') is not None else '',
                '错误':t.get('error',t.get('error_code',''))} for t in out.get('trace',[])])
            st.json(timing(row));st.json(rec.get('automatic',{}).get('request_measurements',[]))
            st.json(out.get('trace',[]))
        st.markdown('**任务评分**')
        try:old=task_review(Path(row['path']).parent)
        except ValueError as exc:old={};st.warning(str(exc))
        with st.form('task_'+rid):
            reviewer=st.text_input('复核人',value=old.get('reviewer',''))
            labels={'task_correct':'任务是否完成','evidence_support':'完整回答是否有证据支持','text_complete':'文字是否完整'};scores={}
            for k,label in labels.items():scores[k]=st.selectbox(label,[None,'pass','fail'],index=[None,'pass','fail'].index(old.get('scores',{}).get(k)),format_func=lambda v:{None:'待评','pass':'通过','fail':'失败'}[v])
            notes=st.text_area('依据与问题说明',value=old.get('notes',''))
            if st.form_submit_button('保存任务评分',type='primary'):
                try:save_task(row['path'],reviewer,scores,notes);st.rerun()
                except ValueError as exc:st.error(str(exc))
        with st.expander('调用复核与文字陈述评分（细粒度）'):
            try:oldfine=fine_review(row['path'])
            except ValueError:oldfine={}
            st.caption('默认只自动确认重复摘要/重复参数调用；无关查询与no_match需看任务判断。文字陈述需逐条填写，不用格式校验代替。')
            with st.form('fine_'+rid):
                reviewer=st.text_input('细粒度复核人',value=oldfine.get('reviewer',''));decisions={}
                for c in item['calls']:
                    st.write(f"轨迹 {c['trace_index']} · {c['tool']} · {c['arguments']}")
                    st.caption('自动提示：'+c['reason'])
                    value=st.selectbox('调用类别',VERDICTS,index=VERDICTS.index(c['verdict']),format_func=lambda v:{'pending':'待核查','necessary':'必要','redundant':'冗余','failed':'执行失败'}[v],key=f"call_{rid}_{c['trace_index']}")
                    note=st.text_input('分类理由',value=c['note'],key=f"note_{rid}_{c['trace_index']}")
                    if note.strip() or value!=c['verdict']:decisions[str(c['trace_index'])]={'verdict':value,'note':note}
                st.caption('陈述格式：[{"claim":"平均RR约0.986秒","verdict":"supported","basis":"证据ID /calculation_facts/mean_rr_seconds"}]。verdict可为pending/supported/unsupported。')
                claims=st.text_area('逐条陈述 JSON',value=json.dumps(oldfine.get('claims',[]),ensure_ascii=False,indent=2))
                whole=st.checkbox('已覆盖答案中全部可核查陈述',value=oldfine.get('claims_complete',False))
                if st.form_submit_button('保存调用与陈述评分'):
                    try:
                        parsed=json.loads(claims)
                        if not isinstance(parsed,list):raise ValueError('陈述必须是JSON数组')
                        save_fine(row['path'],reviewer,decisions,parsed,whole);st.rerun()
                    except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
            st.write(f"陈述支持/不支持/待评：{item['supported_claims']}/{item['unsupported_claims']}/{item['pending_claims']}")
            st.write('已评陈述支持率：'+pretty_rate(item['claim_support_rate'])+'；完整覆盖：'+str(item['claims_complete']))
        guide(st,['coverage','structure','claims','redundancy','latency'])
        st.download_button('下载此轮完整记录',json.dumps(rec,ensure_ascii=False,indent=2),file_name=rid+'.json',mime='application/json')
    else:
        st.subheader('同题、同配置的优化前后对照')
        older=st.selectbox('优化前批次',[b for b in batches if b!=batch]) if len(batches)>1 else None
        if not older:st.info('需要另一个批次；当前批次视为优化后。');return
        try:oldmanifest,oldrows,olditems,oldwarnings=snapshot(runs,older)
        except Exception as exc:st.error(str(exc));return
        for w in oldwarnings:st.warning('优化前：'+w)
        pairs,excluded,only_before,only_after=paired([r for r in oldrows if r['scheme'] in schemes and r['kind'] in kinds],selected)
        st.caption(f'可配对 {len(pairs)} 轮 · 不兼容 {len(excluded)} 轮 · 仅优化前 {only_before} 轮 · 仅优化后 {only_after} 轮')
        st.info('按案例、方案、重复编号、轮次配对，核对模型、端点、数据、题库、参数、输入哈希和参考值。允许代码哈希变化，但应说明干预内容；这不能排除服务端别名升级或负载变化。')
        if excluded:
            with st.expander('排除原因'):table(st,excluded)
        oldindex={x['run_id']:x for x in olditems};comparison=[]
        for p in pairs:
            a,b=p['before'],p['after'];fa,fb=oldindex[a['run_id']],findex[b['run_id']]
            comparison.append({'案例':a['case'],'方案':NAMES[a['scheme']],'轮次':a['turn'],
                '结果（前→后）':a['outcome']+' → '+b['outcome'],
                '字段（前→后）':f"{fa['field_correct']}/{fa['field_total']} → {fb['field_correct']}/{fb['field_total']}",
                '查询（前→后）':f"{fa['query_requests']} → {fb['query_requests']}",
                '模型请求（前→后）':f"{a['model_calls']} → {b['model_calls']}",
                '全文证据支持率（前→后）':(pretty_rate(fa['claim_support_rate']) if fa['claims_complete'] else '未完整复核')+' → '+(pretty_rate(fb['claim_support_rate']) if fb['claims_complete'] else '未完整复核'),
                '耗时变化（秒，后−前）':round(b['latency']-a['latency'],2) if a['latency'] is not None and b['latency'] is not None else None})
        table(st,comparison)
        if pairs:
            qa=sum(oldindex[p['before']['run_id']]['query_requests'] for p in pairs);qb=sum(findex[p['after']['run_id']]['query_requests'] for p in pairs)
            ma=sum(p['before']['model_calls'] for p in pairs);mb=sum(p['after']['model_calls'] for p in pairs)
            st.write(f'配对记录查询总数：{qa} → {qb}；模型请求总数：{ma} → {mb}。')
            if any(p['before']['outcome']=='pending' or p['after']['outcome']=='pending' for p in pairs):st.warning('仍有待评，不能据此宣布答案质量保持不变。先完成两批的证据复核。')
            st.download_button('下载配对对照 CSV',pd.DataFrame(comparison).to_csv(index=False).encode('utf-8-sig'),file_name='paired_comparison.csv',mime='text/csv')
        with st.expander('两批配置与干预版本'):st.json({'before':oldmanifest['config'],'after':manifest['config']})
        guide(st,['success','coverage','claims','redundancy','requests','latency'])
