"""Generate standalone HTML and JSON without changing original results."""
import argparse
import html
import json
from datetime import datetime,timezone
from pathlib import Path
from src.evaluation.threeway_metrics import load_batch,summarize,conversation_outcomes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest,rows,warnings=load_batch(root/'evaluation/threeway_runs',a.batch)
    summary=summarize(rows)
    target=root/'evaluation/threeway_reports'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');target.mkdir(parents=True)
    data={'manifest':manifest,'summary':summary,'warnings':warnings,'conversations':conversation_outcomes(manifest,rows)}
    (target/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    esc=lambda s:html.escape(str(s))
    pieces=['<!doctype html><meta charset="utf-8"><title>ECG 三方案对照</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#183048}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}.bar{display:flex;height:24px;background:#eee;margin:10px 0}.pass{background:#2a9d8f}.fail{background:#e76f51}.pending{background:#d9bc60}details{padding:12px;border:1px solid #ddd;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>ECG 三方案对照实验</h1><p>真实本地批次：'+esc(a.batch)+'</p><p>绿色=确认成功，红色=失败，黄色=待评。未复核不算成功；结构检查不验证文字语义。延迟含失败等待。</p>']
    for w in warnings:pieces.append('<p>'+esc(w)+'</p>')
    for s in summary:
        pieces.append('<h2>'+esc(s['scheme'])+'</h2><div class="bar">'+''.join('<span class="'+k+'" style="width:'+str(100*s[k]/s['runs'])+'%"></span>' for k in ('pass','fail','pending'))+'</div><p>'+esc(f"成功 {s['pass']} / 失败 {s['fail']} / 待评 {s['pending']}")+'</p>')
    if summary:
        cols=['scheme','runs','confirmed_success_rate','possible_success_rate','structure_rate','task_field_rate','median_seconds','p95_seconds','mean_query_calls','mean_model_calls','timeout_rate']
        pieces.append('<table><tr>'+''.join('<th>'+esc(c)+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(round(s[c],3) if isinstance(s[c],float) else s[c])+'</td>' for c in cols)+'</tr>' for s in summary)+'</table>')
    pieces.append('<h2>逐题记录</h2>')
    for r in sorted(rows,key=lambda x:(x['case'],x['repetition'],x['turn'],x['scheme'])):
        out=r['record']['output']
        pieces.append('<details><summary>'+esc(f"{r['case']} / {r['turn']} / {r['scheme']} / {r['outcome']}")+'</summary><p>'+esc(r['record']['case']['question'])+'</p><pre>'+esc((out.get('draft') or {}).get('answer','无回答'))+'</pre><pre>'+esc(json.dumps(out.get('trace',[]),ensure_ascii=False,indent=2))+'</pre></details>')
    (target/'report.html').write_text(''.join(pieces),encoding='utf-8');print('Report:',target/'report.html')
if __name__=='__main__':main()
