"""Frozen question-only rules. No case IDs, kinds or expected answers."""
import re


def route(question, resolution=None):
    q=question.strip()
    resolution=resolution or {}
    if resolution.get('status')=='needs_clarification':
        return [], resolution['message']
    if resolution.get('status')=='resolved':
        w=resolution['window']
        return [('inspect_error_window',{k:w[k] for k in ('lead','start_sample','end_sample')})], 'historical_window'
    calls=[]
    # Multiple windows and conditional plans are outside this frozen ruleset.
    leads=re.findall(r'(?<![A-Za-z0-9])(aVR|aVL|aVF|III|II|I|V[1-6])(?![A-Za-z0-9])',q,re.I)
    mapping={x.upper():x for x in ('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')}
    durations=re.findall(r'(?:最后|末尾|末)\s*(\d+(?:\.\d+)?)\s*秒',q)
    if re.search(r'如果|若.*则|否则',q) and 'QT' not in q.upper():
        return [],'规则不支持条件分支，请拆分问题。'
    if len(set(x.upper() for x in leads))>1 or len(durations)>1:
        return [],'存在多个窗口，请逐一指定。'
    if leads and durations:
        name='inspect_recent_rr_alignment' if re.search(r'重叠|对齐',q) else 'inspect_recent_error'
        calls.append((name,{'lead':mapping[leads[0].upper()],'duration_seconds':float(durations[0])}))
    elif re.search(r'窗口|采样点.*(?:到|至|\[)',q) and leads:
        pair=re.search(r'\[\s*(\d+)\s*[,，]\s*(\d+)\s*\)',q)
        if pair:calls.append(('inspect_error_window',{'lead':mapping[leads[0].upper()],'start_sample':int(pair[1]),'end_sample':int(pair[2])}))
        else:return [],'请提供明确导联及窗口坐标或末尾时长。'
    if re.search(r'RR|间隔|候选峰|心率',q,re.I) and not re.search(r'QTc|重叠|对齐',q,re.I):
        calls.append(('inspect_rr_intervals',{'offset':0,'limit':50}))
    if re.search(r'阈值|是否异常|是否超过|模型预测|模型判定',q):calls.append(('get_model_decision',{}))
    return calls,'matched' if calls else 'summary_only_or_unsupported'
