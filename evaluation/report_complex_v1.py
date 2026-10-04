"""Static report; raw results and reviews remain untouched."""
import argparse
import html
import json
import statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def summarize(rows):
    out=[]
    for scheme in sorted({r['config']['scheme'] for r in rows}):
        subset=[r for r in rows if r['config']['scheme']==scheme]
        eligible=[r for r in subset if not r['automatic'].get('ineligible')]
        times=sorted(r['wall_seconds'] for r in eligible)
        out.append({'scheme':scheme,'runs':len(subset),'eligible':len(eligible),
            'automatic_pass':sum(r['automatic'].get('automatic_task_pass',False) for r in eligible),
            'covered':sum(r['automatic']['target_covered'] for r in eligible),'targets':sum(r['automatic']['target_total'] for r in eligible),
            'model_requests':sum(r['automatic'].get('model_requests',0) for r in subset),
            'queries':sum(r['automatic'].get('supplemental_queries',0) for r in subset),
            'extra_windows':sum(r['automatic'].get('extra_window_count',0) for r in subset),
            'duplicates':sum(r['automatic'].get('duplicate_query_count',0) for r in subset),
            'median_seconds':round(statistics.median(times),2) if times else None,
            'errors':dict(Counter(r['output'].get('error') for r in subset if r['output'].get('error')))})
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);a=p.parse_args()
    root=ROOT/'evaluation/complex_runs_v1';manifest=json.loads((root/(a.batch+'.manifest.json')).read_text(encoding='utf-8'))
    rows=[]
    for f in root.glob('*/result.json'):
        r=json.loads(f.read_text(encoding='utf-8'))
        if r['config'].get('batch_id')==a.batch:rows.append(r)
    data=summarize(rows);dest=ROOT/'evaluation/complex_reports_v1'/a.batch;dest.mkdir(parents=True,exist_ok=True)
    (dest/'summary.json').write_text(json.dumps({'batch':a.batch,'expected_runs':len(manifest['tasks']),'found_runs':len(rows),'summary':data},ensure_ascii=False,indent=2),encoding='utf-8')
    esc=lambda x:html.escape(str(x))
    parts=['<!doctype html><meta charset="utf-8"><title>条件查询对照</title><style>body{font:16px system-ui;max-width:1200px;margin:36px auto;color:#243449}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;background:#f3f6fa;padding:16px}details{margin:16px 0}progress{width:100px}small{color:#536273}</style><h1>条件查询 · 一次规划与 Agent 对照</h1>',
           f'<p>批次 {esc(a.batch)} · 已记录 {len(rows)}/{len(manifest["tasks"])} 次。缺失运行不可当作成功；请确认全量跑完。</p>',
           '<p>自动通过是字段、对象、必需查询的联合检查，尚不是人工确认任务成功率。人工复核填写每个运行目录的 review.json；本页不合并人工评分。</p>',
           '<table><tr><th>方案</th><th>自动通过/有效运行</th><th>字段覆盖</th><th>模型请求</th><th>补查</th><th>额外窗口/重复</th><th>中位耗时</th></tr>']
    for s in data:
        parts.append(f'<tr><td>{esc(s["scheme"])}</td><td>{s["automatic_pass"]}/{s["eligible"]}（共{s["runs"]}）</td><td><progress max="{s["targets"] or 1}" value="{s["covered"]}"></progress> {s["covered"]}/{s["targets"]}</td><td>{s["model_requests"]}</td><td>{s["queries"]}</td><td>{s["extra_windows"]}/{s["duplicates"]}</td><td>{s["median_seconds"]}s</td></tr>')
    parts.append('</table><h2>指标如何定位问题</h2><ul><li>字段覆盖：正确且引用到指定对象的目标字段/全部目标字段。低时展开记录，核对计划、导联和窗口。</li><li>自动通过：全部字段正确且所有必需窗口实际查询；失败及超时计入有效运行分母。</li><li>补查：工具 trace 的查询次数，排除启动摘要和答案提交。额外窗口仅表示不在目标集合，不等于已确认冗余。</li><li>延迟：包含失败等待的端到端中位耗时；当前小样本不展示不稳定的P95。检查每次请求耗时区分网关等待与本地执行。</li><li>结构通过不保证所有文字陈述受支持。需核查分支说明、坐标口径及额外陈述。</li><li>rules仅作原有覆盖范围基线；不能用它不支持新任务证明通用规则无法实现。</li></ul><h2>逐题记录与失败定位</h2>')
    for r in sorted(rows,key=lambda r:(r['case']['id'],r['case']['sample_index'],r['config']['repetition'],r['config']['scheme'])):
        text=f"{r['case']['id']} · 样本{r['case']['sample_index']} · 重复{r['config']['repetition']} · {r['config']['scheme']} · {r['output']['status']}"
        parts.append('<details><summary>'+esc(text)+'</summary><p>'+esc(r['case']['question'])+'</p><p>'+esc(r['output'].get('draft',{}).get('answer','无答案'))+'</p><pre>'+esc(json.dumps({'run_id':r['run_id'],'error':r['output'].get('error'),'automatic':r['automatic'],'reference':r['reference'],'plan':r['output'].get('plan'),'trace':r['output'].get('trace')},ensure_ascii=False,indent=2))+'</pre></details>')
    (dest/'report.html').write_text(''.join(parts),encoding='utf-8');print('Report:',dest/'report.html')
if __name__=='__main__':main()
