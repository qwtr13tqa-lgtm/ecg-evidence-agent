"""Offline fine-grained HTML/JSON export. Never updates original records."""
import argparse
import html
import json
from pathlib import Path
from datetime import datetime,timezone
from src.evaluation.threeway_metrics import load_batch
from src.evaluation.fine_metrics import analyze,aggregate,read_review


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest,rows,warnings=load_batch(root/'evaluation/threeway_runs',a.batch)
    items=[]
    for row in rows:
        try:review=read_review(row['path'])
        except ValueError as exc:review={};warnings.append(str(exc))
        items.append(analyze(row,review))
    summary=aggregate(items)
    folder=root/'evaluation/fine_reports'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
    (folder/'fine_metrics.json').write_text(json.dumps(dict(batch=a.batch,summary=summary,items=items,warnings=warnings),ensure_ascii=False,indent=2),encoding='utf-8')
    esc=lambda v:html.escape(str(v))
    s=['<!doctype html><meta charset="utf-8"><title>ECG 细粒度评分</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:16px system-ui;color:#19344a}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}progress{width:220px;height:24px}pre{white-space:pre-wrap}details{margin:16px 0}</style><h1>ECG 细粒度评分与冗余查询</h1><p>批次 '+esc(a.batch)+'</p><p>覆盖率仅针对已定义的结构化目标字段，不等于完整答案成功率。缺失信息题无数值字段，不计入该分母。文字语义未标注时保持缺失。</p>']
    for w in warnings:s.append('<p>'+esc(w)+'</p>')
    s.append('<table><tr><th>方案</th><th>字段覆盖</th><th>查询</th><th>确认冗余</th><th>待核查</th><th>模型请求</th></tr>')
    for row in summary:
        rate=row['field_coverage'];label=f"{row['field_correct']}/{row['field_total']}"
        chart='' if rate is None else f'<progress max="1" value="{rate}"></progress>'
        s.append('<tr><td>'+esc(row['scheme'])+'</td><td>'+chart+' '+label+'</td>'+''.join('<td>'+esc(row[k])+'</td>' for k in ('query_requests','redundant_calls','pending_calls','model_requests'))+'</tr>')
    s.append('</table><p>冗余默认只自动确认同轮重复摘要或相同入参的重复请求；无关查询和空检索需要复核。查询失败单列，不重复扣分。初始摘要与提交答案不计补充查询。</p>')
    for x in items:
        s.append('<details><summary>'+esc(x['case']+' / '+x['scheme'])+'</summary><pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2))+'</pre></details>')
    (folder/'report.html').write_text(''.join(s),encoding='utf-8');print('Report:',folder/'report.html')
if __name__=='__main__':main()
