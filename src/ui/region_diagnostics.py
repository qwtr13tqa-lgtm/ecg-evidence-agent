"""Replay saved evidence-1.0 region extraction; never use conversation text."""
import numpy as np


def diagnose(error_map, config, saved_regions, sampling_rate):
    required = {'version','region_threshold','min_region_length','merge_gap',
                'top_k_regions','normalization_percentiles'}
    if not isinstance(config, dict) or not required.issubset(config):
        return {'status':'unavailable','reason':'记录未保存完整提取配置，无法确认空区域原因；不会套用当前默认值。'}
    if config['version'] != 'evidence-1.0' or config['normalization_percentiles'] != [5,95]:
        return {'status':'unavailable','reason':'该提取配置版本尚不支持复核。'}
    if error_map is None:
        return {'status':'unavailable','reason':'记录没有保存误差图，无法复核。'}
    x = np.asarray(error_map, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 12 or not x.shape[0] or not np.isfinite(x).all():
        return {'status':'unavailable','reason':'保存的误差图形状或数值无效。'}
    threshold, length, gap, top = (config[k] for k in (
        'region_threshold','min_region_length','merge_gap','top_k_regions'))
    if (not isinstance(threshold,(int,float)) or isinstance(threshold,bool)
            or not np.isfinite(threshold) or not 0 <= threshold <= 1
            or any(type(v) is not int for v in (length,gap,top))
            or length < 1 or gap < 0 or top < 1):
        return {'status':'unavailable','reason':'记录中的提取参数无效，无法复核。'}
    raw_score = np.mean(x,axis=1)
    lo, hi = np.percentile(raw_score,[5,95])
    degenerate = not np.isfinite(lo) or not np.isfinite(hi) or hi-lo < 1e-8
    score = np.zeros_like(raw_score) if degenerate else np.clip((raw_score-lo)/(hi-lo),0,1).astype(np.float32)
    mask = score >= threshold
    edges = np.diff(np.r_[False,mask,False].astype(np.int8))
    raw = list(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))
    merged = []
    for start,end in raw:
        if merged and start-merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start,end])
    kept = [{'start':s,'end':e,'score':float(np.mean(score[s:e]))}
            for s,e in merged if e-s >= length]
    kept.sort(key=lambda r:r['score'],reverse=True)
    selected = kept[:top]
    def coordinate(r):
        return (r['start'],r['end'])
    try:
        matches = ([coordinate(r) for r in saved_regions] == [coordinate(r) for r in selected])
    except (KeyError,TypeError):
        matches = False
    if not matches:
        reason = '复算窗口与已保存窗口不一致，需检查记录及提取版本；本页不覆盖历史结果。'
    elif selected:
        reason = f'当前提取规则返回 {len(selected)} 个候选时间区域。'
    elif degenerate and not raw:
        reason = '跨导联均值的第5与95百分位差过小，归一化结果为零，未形成高分片段。'
    elif not raw:
        reason = '归一化时间分数没有达到记录中的区域阈值，未形成高分片段。'
    else:
        reason = '存在高分片段，但合并后的区间全部短于最小长度，因而没有返回候选区域。'
    result = {'status':'verified' if matches else 'inconsistent', 'reason':reason,
        'source':'saved_error_map_and_saved_configuration_no_llm_no_history_text',
        'raw_regions':len(raw),'merged_regions':len(merged),
        'excluded_short_regions':len(merged)-len(kept),
        'eligible_regions':len(kept),'top_k_omitted':max(0,len(kept)-top),
        'returned_regions':len(selected), 'saved_regions':len(saved_regions),
        'region_threshold':threshold,'min_region_length_samples':length,
        'merge_gap_samples':gap,'normalization_degenerate':bool(degenerate),
        'recomputed_regions':selected, 'configuration':config}
    if isinstance(sampling_rate,(int,float)) and sampling_rate > 0:
        result.update(min_region_length_seconds=length/sampling_rate,merge_gap_seconds=gap/sampling_rate)
    return result


def render_regions(st, result, context):
    regions = context['evidence'].get('temporal_regions') or []
    report = diagnose(result.model.error_map,
        result.provenance.get('evidence_configuration'), regions, result.input.sampling_rate)
    st.subheader('候选时间区域：本地复核')
    st.caption('来自本次分析保存的误差图和提取参数，不读取聊天历史，不调用大模型，不重新运行深度模型。')
    if report['status'] != 'verified':
        st.warning(report['reason'])
    elif regions:
        st.success(report['reason'])
    else:
        st.info(report['reason'])
        st.caption('未返回候选窗口不等于信号没有异常，也不等于整体分类一定正常。')
    if 'raw_regions' in report:
        cols = st.columns(4)
        for col,label,key in zip(cols,['原始高分片段','合并后区间','长度不足排除','复算返回窗口'],
                                 ['raw_regions','merged_regions','excluded_short_regions','returned_regions']):
            col.metric(label,report[key])
        st.caption(f"区域阈值 {report['region_threshold']}（不是整体分类阈值）；最小长度 {report['min_region_length_samples']} 点；合并间隔上限 {report['merge_gap_samples']} 点。")
    if regions:
        st.dataframe([{'起点/采样点':r['start'],'终点/采样点（不含）':r['end'],
                       '起点/秒':r['start']/result.input.sampling_rate,
                       '终点/秒':r['end']/result.input.sampling_rate,
                       '区域分数':r.get('score')} for r in regions],hide_index=True)
    with st.expander('区域筛选明细与配置'):
        st.json(report)
