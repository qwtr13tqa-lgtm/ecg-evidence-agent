"""Offline V2 report. Fail closed on duplicate/mismatched records; never modify runs."""
import argparse
import csv
import hashlib
import json
import math
import uuid
from collections import Counter
from pathlib import Path
from src.evaluation.checks_v2 import score_output

VERSION='comparison-export-1.0'

def read(path):
    def reject(x): raise ValueError('Nonfinite JSON')
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=reject)

def failure_kind(rec):
    out=rec['output']; trace=out.get('trace',[])
    if rec.get('runner_failure'): return 'runner_failure'
    if out.get('status')=='completed_draft':return 'completed_draft'
    errors=[t for t in trace if t.get('error_type')]
    last=errors[-1] if errors else {}
    if last.get('error_type')=='APITimeoutError':return 'gateway_timeout'
    if out.get('error')=='GATEWAY_REQUEST_FAILED':return 'gateway_request_failure'
    phase=last.get('failure_phase')
    if phase in ('response_protocol','answer_json','answer_validation'):return phase+'_failure'
    if any(t.get('stage')=='tool' and t.get('ok') is False for t in trace):return 'tool_or_control_failure'
    return 'other_failure'

def collect(root,batch):
    # UUID-only command parameter also prevents path traversal.
    if str(uuid.UUID(batch))!=batch:raise ValueError('Invalid batch UUID')
    root=Path(root); manifest=read(root/(batch+'.manifest.json'))
    expected=[]
    for t in manifest['tasks']:
        for s in t['schemes']:expected.append((t['case_id'],t['repetition'],s))
    if len(expected)!=len(set(expected)):raise ValueError('Duplicate planned entry')
    found={};issues=[]
    for path in sorted(root.glob('*/result.json')):
        try:r=read(path)
        except Exception as exc:
            issues.append({'path':str(path),'error':type(exc).__name__});continue
        if r.get('config',{}).get('batch_id')!=batch:continue
        cfg=r['config']; key=(r['case']['id'],cfg['repetition'],cfg['scheme'])
        if key not in expected or key in found:raise ValueError('Unplanned or duplicate run: '+str(key))
        if any(cfg.get(k)!=v for k,v in manifest['config'].items()):raise ValueError('Manifest configuration mismatch')
        if r.get('run_id')!=path.parent.name:raise ValueError('Run directory mismatch')
        out=r['output']; reference=r['reference']
        new=score_output(r['case'],out,reference)
        duration=out.get('elapsed_seconds',r.get('wall_seconds'))
        if duration is not None and (type(duration) not in (int,float) or not math.isfinite(duration) or duration<0):raise ValueError('Invalid duration')
        trace=out.get('trace',[])
        calls=out.get('model_calls')
        if calls is not None and (type(calls) is not int or calls<0):raise ValueError('Invalid model count')
        acquired=any(t.get('stage')=='tool' and t.get('ok') is True and t.get('tool') in
                     ('inspect_rr_intervals','inspect_recent_error','inspect_error_window') for t in trace)
        row=dict(batch_id=batch,case_id=key[0],repetition=key[1],scheme=key[2],sample_index=r['case']['sample_index'],
                 run_id=r['run_id'],status=out.get('status'),failure_category=failure_kind(r),
                 elapsed_seconds=duration,model_calls=calls,
                 successful_specialist_query=acquired,
                 specialist_query_then_no_draft=acquired and out.get('status')!='completed_draft',
                 trace_tool_attempts=sum(t.get('stage')=='tool' for t in trace),
                 result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 original_score_version=r.get('automatic',{}).get('version'),score_version=new['version'],
                 metrics=new['metrics'])
        found[key]=row
    rows=[]
    for key in expected:
        rows.append(found.get(key,dict(batch_id=batch,case_id=key[0],repetition=key[1],scheme=key[2],
                                      status='missing',failure_category='missing',metrics={})))
    groups={}
    for scheme in sorted({k[2] for k in expected}):
        part=[r for r in rows if r['scheme']==scheme];present=[r for r in part if r['status']!='missing']
        names=sorted({k for r in present for k in r['metrics']})
        metrics={}
        for name in names:
            vals=[r['metrics'].get(name) for r in present]
            if any(v is not None and type(v) is not bool for v in vals):raise ValueError('Invalid metric type')
            yes=sum(v is True for v in vals);no=sum(v is False for v in vals)
            metrics[name]={'true':yes,'false':no,'unassessed':len(vals)-yes-no,
                           'assessed_denominator':yes+no,'pass_rate_assessed':yes/(yes+no) if yes+no else None}
        durations=[r['elapsed_seconds'] for r in present if r['elapsed_seconds'] is not None]
        groups[scheme]={'planned':len(part),'recorded':len(present),'missing':len(part)-len(present),
                        'completed':sum(r['status']=='completed_draft' for r in present),
                        'failure_categories':dict(Counter(r['failure_category'] for r in part)),
                        'mean_elapsed_seconds_including_failures':sum(durations)/len(durations) if durations else None,
                        'elapsed_denominator':len(durations),
                        'model_calls_sum':sum(r['model_calls'] or 0 for r in present),
                        'model_calls_missing':sum(r['model_calls'] is None for r in present),
                        'specialist_query_then_no_draft':sum(r['specialist_query_then_no_draft'] for r in present),
                        'metrics':metrics}
    return {'export_version':VERSION,'scoring_version':'development-checks-2.0','batch_id':batch,
            'source_config':manifest['config'],'rows':rows,'groups':groups,'unreadable_records':issues,
            'semantic_evaluation':'not_performed_by_this_export',
            'notes':['Each batch is reported separately; no cross-configuration pooling.',
                     'V2 recomputed from original output/reference, not unverified sidecars.',
                     'Null means unassessed, not failure or success. Missing runs counted separately.',
                     'Elapsed includes failed waiting, excludes shared inference in normal runs.',
                     'Raw trace tool counts have different bootstrap conventions; not an efficiency ranking.',
                     'No medical accuracy or free-text correctness claim; human reviews are not consumed.',
                     'Specialist query success is execution success, not proof of sufficient evidence.']}

def export(report,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    (dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    metric_names=sorted({k for r in report['rows'] for k in r['metrics']})
    fields=['batch_id','case_id','sample_index','repetition','scheme','run_id','status','failure_category',
            'elapsed_seconds','model_calls','specialist_query_then_no_draft','trace_tool_attempts','result_sha256']
    with (dest/'cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields+metric_names);w.writeheader()
        for r in report['rows']:
            row={k:r.get(k,'') for k in fields}
            row.update({k:'unassessed' if r['metrics'].get(k) is None else str(r['metrics'][k]).lower() for k in metric_names})
            w.writerow(row)
    lines=['# 自动开发评测报告','', '评分：development-checks-2.0。未评估自由文本语义或医学正确性。','',
           '| 方案 | 计划 | 已记录 | 草稿完成 | 缺记录 | 平均耗时含失败/秒 |', '|---|---:|---:|---:|---:|---:|']
    for s,g in report['groups'].items():
        mean=g['mean_elapsed_seconds_including_failures']; value=f'{mean:.2f}' if mean is not None else '未提供'
        lines.append(f"| {s} | {g['planned']} | {g['recorded']} | {g['completed']} | {g['missing']} | {value} |")
    lines+=['','## 自动指标：通过 / 未通过 / 未评估','', '| 方案 | 指标 | 通过 | 未通过 | 未评估 |','|---|---|---:|---:|---:|']
    for s,g in report['groups'].items():
        for k,v in g['metrics'].items():lines.append(f"| {s} | {k} | {v['true']} | {v['false']} | {v['unassessed']} |")
    lines+=['','## 运行结果与失败分类','','| 方案 | 分类 | 次数 |','|---|---|---:|']
    for scheme, group in report['groups'].items():
        for category, count in group['failure_categories'].items():
            lines.append(f"| {scheme} | {category} | {count} |")
    lines+=['','## 解释范围','','- 原始结果和评审文件均未修改。','- 超时保留为运行失败；未评估项不计入指标分母。',
            '- 工具选择指标不作为摘要基线的答案正确率。','- 不跨批次混合配置，工具原始计数不可直接比较。',
            '- 取证后未完成回答表示专业查询成功执行后无草稿，不保证所取证据正确或充分。',
            '- 逐案例及失败阶段见 cases.csv；完整配置、分母、源文件哈希见 report.json。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',action='append',required=True)
    p.add_argument('--runs-dir',default='evaluation/comparison_runs');p.add_argument('--output-root',default='evaluation/reports')
    a=p.parse_args()
    reports=[collect(a.runs_dir,b) for b in dict.fromkeys(a.batch)]
    root=Path(a.output_root)/str(uuid.uuid4())
    for r in reports:
        dest=root/r['batch_id'];export(r,dest);print('Exported:',dest)
        for s,g in r['groups'].items():print(s,'completed',g['completed'],'/',g['recorded'],'missing',g['missing'])
    print('Offline only. Original records unchanged. Open report.md for each batch.')
if __name__=='__main__':main()
