"""Batch task entry for 8505, separate comparable execution records for 8506."""
import json
from pathlib import Path
from src.review.conditional_batch import prepare,run,save,TASK

LABELS={'rules':'固定流程（本地）','agent':'Agent（最多8次模型请求）'}
METRICS={
 'branch_decision_accuracy':('条件执行正确率','应补查且成功、或不应补查且没有尝试的记录/条件可判定记录。缺失证据不算分母。检查误触发与漏补查。'),
 'record_coverage':('记录覆盖率','汇总记录/所选记录。缺少分析也应占一行；低值先看是否收集成功及网关是否中断。'),
 'matched_evidence_rate':('匹配证据率','匹配历史/所选记录。低值看输入、权重、适配器、阈值是否一致；这通常是数据可用性，不是Agent能力。'),
 'supplement_completion':('条件补查完成率','成功补查/应补查记录。无应补查记录为不适用；低值检查遗漏调用、错误图缺失或预算耗尽。'),
 'unnecessary_query_attempts':('不必要补查尝试','条件不满足、越出批次或重复的补查请求数。服务端会拒绝；看工具轨迹和对应条件。'),
 'rejected_calls':('被拒绝调用数','包括过早提交、未收集就查询、非法参数等。先查首次失败；不要只增加预算。'),
 'model_requests':('模型请求数','包括失败请求。证据齐全后仍继续查，或逐条串行查询，会增加请求数；可批量提出独立补查。'),
 'window_query_attempts':('窗口查询尝试数','所有inspect_tail_window调用，含拒绝；与实际执行数分别统计。'),
 'window_computations':('窗口实际执行数','通过条件校验后调用本地窗口计算的次数，含执行失败。不含深度模型推理，本任务不重跑模型。')}


def render(st,root,history,open_record,report,batch):
    with st.expander('批量证据收集＋条件补查（新任务）'):
        st.write(TASK)
        category=st.selectbox('任务对象',['fp','fn','tp','tn','all'],format_func=lambda x:{'fp':'误报','fn':'漏报','tp':'异常检出','tn':'正常判对','all':'全部'}[x],key='conditional_category')
        mode=st.radio('执行方式',list(LABELS),format_func=lambda x:LABELS[x],horizontal=True,key='conditional_mode')
        st.caption('固定流程和Agent共用证据快照与查询工具。结果由本地数据汇总；本阶段评估工具执行，不评估LLM自由文本解释。')
        consent=st.checkbox('允许将所选记录的结构化证据发送到已配置网关',key='conditional_consent') if mode=='agent' else True
        key='conditional_result_'+batch+'_'+category+'_'+mode
        if st.button('执行证据收集与条件补查',disabled=not consent,type='primary',key='conditional_run'):
            gateway=None
            try:
                snapshot,objects=prepare(report,history,category)
                if not snapshot['rows']:st.info('所选类别没有记录，无需执行。')
                else:
                    if mode=='agent':
                        from src.agent.gateway import ToolGateway
                        gateway=ToolGateway(timeout=60,max_tokens=1200)
                    result=run(snapshot,objects,mode,gateway)
                    path=save(root,result)
                    st.session_state[key]=result
                    st.success('已保存执行记录：'+str(path.relative_to(root)))
            except Exception as exc:st.error('未完成：'+type(exc).__name__+' · '+str(exc))
            finally:
                if gateway is not None:gateway.close()
        result=st.session_state.get(key)
        if result:
            st.write('执行状态：'+result['status']+'；错误：'+(result['error'] or '无'))
            st.caption('completed表示收集与必要补查流程完成；不表示所有样本都有匹配证据，也不表示疾病结论。')
            display(st,result)
            for row in result['rows']:
                if row['source_analysis_id'] and st.button('打开样本 '+str(row['index'])+' 的匹配分析',key=key+str(row['index'])):
                    open_record(row['source_analysis_id'])
                    st.session_state['workspace_entry']='历史记录';st.rerun()
            st.download_button('下载本次批量执行记录',json.dumps(result,ensure_ascii=False,indent=2),file_name=result['run_id']+'.json',key=key+'_download')
            st.caption('到8506选择“批量条件补查”，查看记录覆盖、条件执行与固定流程对照。')


def display(st,result):
    rows=[]
    for r in result['rows']:
        w=r.get('window') or {};d=w.get('data') or {};lead=(d.get('leads') or [{}])[0]
        rows.append({'样本':r['index'],'历史状态':r['state'],'候选区域条件':r['condition'],'计划分支':r['branch'],
            '补查状态':r['query_state'],'导联':r.get('lead'),'起点':d.get('start_sample'),'终点':d.get('end_sample'),
            '均值':lead.get('mean'),'最大值':lead.get('maximum'),'峰位置':lead.get('peak_sample')})
    st.caption('最后0.6秒是明确指定的复核窗口，不是新检测出的候选区域，也不是病变定位。')
    if rows:st.dataframe(rows,hide_index=True,use_container_width=True)
    else:st.info('尚未收集到结果。')


def render_evaluation(st,root):
    st.subheader('批量条件补查：执行质量与成本')
    st.caption('独立任务协议，不混入原三题成功率。先运行本地固定流程，再运行Agent；只有相同证据快照可直接对照。')
    records=[]
    for path in (Path(root)/'evaluation/batch_evidence_runs').glob('*/result.json'):
        try:records.append(json.loads(path.read_text(encoding='utf-8')))
        except (ValueError,OSError) as exc:st.warning(path.parent.name+' 无法读取：'+type(exc).__name__)
    if not records:st.info('尚无记录。在8505批量复核页展开“批量证据收集＋条件补查”执行。');return
    records.sort(key=lambda r:r['created_at'],reverse=True)
    byid={r['run_id']:r for r in records}
    rid=st.selectbox('执行记录',list(byid),format_func=lambda i:byid[i]['created_at']+' · '+byid[i]['mode']+' · '+byid[i]['snapshot']['category']+' · '+i[:8])
    r=byid[rid];m=r['metrics']
    st.write('状态：'+r['status']+'；错误：'+(r['error'] or '无'))
    st.caption('记录覆盖100%只表示没有遗漏所选行；请同时看匹配证据率和条件补查完成率。')
    st.dataframe([{'指标':label,'值':'不适用' if m.get(k) is None else str(m[k]),'解释与定位':helptext} for k,(label,helptext) in METRICS.items()],hide_index=True,use_container_width=True)
    st.write('总耗时：'+str(round(r['wall_seconds'],3))+' 秒；含历史快照准备：否（准备阶段在两方案运行前完成）。')
    display(st,r)
    compatible=[x for x in records if x['run_id']!=rid and x['mode']!=r['mode'] and x['snapshot']['fingerprint']==r['snapshot']['fingerprint'] and x['version']==r['version'] and x['config'].get('implementation_sha256')==r['config'].get('implementation_sha256')]
    if compatible:
        otherid=st.selectbox('选择相同快照的另一方案', [x['run_id'] for x in compatible])
        other=byid[otherid]
        st.dataframe([{'方案':x['mode'],'状态':x['status'],'记录覆盖':x['metrics']['record_coverage'],
            '匹配证据率':x['metrics']['matched_evidence_rate'],'条件完成':x['metrics']['supplement_completion'],
            '不必要补查':x['metrics']['unnecessary_query_attempts'],'模型请求':x['metrics']['model_requests'],
            '窗口实际执行':x['metrics']['window_computations'],'耗时秒':x['wall_seconds']} for x in [r,other]],hide_index=True,use_container_width=True)
        st.info('这个明确条件适合固定流程。Agent若只达到相同结果而请求更多，应如实记录；本任务不能单独证明自主规划优于规则。')
    else:st.info('暂无相同快照的另一方案；若新增或改变历史分析，快照不同，需要重跑两方案。')
    with st.expander('执行轨迹、逐次请求与配置'):
        st.json({'trace':r['trace'],'requests':r['requests'],'config':r['config']})
    with st.expander('原始证据与任务快照'):st.json(r)
