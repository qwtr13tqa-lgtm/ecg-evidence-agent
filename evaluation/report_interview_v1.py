"""Offline, standard-library report. Pending reviews never count as passes."""
import argparse,csv,hashlib,html,json,math,statistics
from collections import defaultdict,Counter
from pathlib import Path

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pct(a,b):return a/b if b else None
def quantile(xs,q):
    if not xs:return None
    xs=sorted(xs);k=(len(xs)-1)*q;i=int(k)
    return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(k-i)
def inspect(path):
    r=load(path);o=r.get('output',{});a=r.get('automatic',{});cfg=r.get('config',{});fields=a.get('field_checks',{})
    digest=hashlib.sha256(path.read_bytes()).hexdigest();review={};warnings=[]
    rp=path.with_name('review.json')
    if rp.exists():
        v=load(rp)
        if v.get('result_sha256')==digest and v.get('run_id')==r.get('run_id') and v.get('reviewer','').strip():review=v
        else:warnings.append('人工复核未签名或结果哈希不匹配，按待复核处理')
    checks=[review.get(k) for k in ('task_correct','evidence_support','text_complete')]
    complete=o.get('status')=='completed_draft'
    status='fail' if not complete or a.get('metrics',{}).get('structure_values_references') is False or any(v is False for v in fields.values()) or 'fail' in checks else ('pass' if checks==['pass']*3 else 'pending')
    trace=o.get('trace',[]);tools=[t for t in trace if t.get('stage')=='tool']
    # bootstrap is one shared summary per turn. Submission/routing are not queries.
    supplemental=[t for t in tools if t.get('source')!='bootstrap']
    seen=set();duplicates=0
    for t in supplemental:
        ident=json.dumps([t.get('tool'),t.get('arguments')],sort_keys=True)
        duplicates+=ident in seen;seen.add(ident)
    assessment={};ap=path.with_name('interview_review.json')
    if ap.exists():
        v=load(ap)
        if v.get('result_sha256')==digest and v.get('reviewer','').strip():assessment=v
        else:warnings.append('细粒度复核未签名或哈希不匹配')
    claims=assessment.get('claims',[]);callreviews=assessment.get('calls',[])
    validclaims=[c for c in claims if c.get('verdict') in ('supported','unsupported','contradicted')]
    validcalls=[c for c in callreviews if c.get('classification') in ('necessary','duplicate','already_available','irrelevant')]
    return dict(batch=cfg.get('batch_id','unknown'),scheme=cfg.get('scheme','unknown'),run_id=r.get('run_id'),
      case=r.get('case',{}).get('id'),repetition=cfg.get('repetition',1),turn=r.get('case',{}).get('turn_index',1),
      status=status,completed=complete,fields_ok=sum(v is True for v in fields.values()),fields_n=len(fields),
      structure=a.get('metrics',{}).get('structure_values_references') is True,
      latency=r.get('wall_seconds',o.get('elapsed_seconds')),model_calls=o.get('model_calls',0),
      queries=len(supplemental),query_failures=sum(not t.get('ok') for t in supplemental),duplicates=duplicates,
      error=o.get('error',''),timeout=any(t.get('error_type') in ('APITimeoutError','TimeoutError') for t in trace),
      claims_n=len(validclaims),claims_supported=sum(c['verdict']=='supported' for c in validclaims),
      calls_reviewed=len(validcalls),redundant=sum(c['classification']!='necessary' for c in validcalls),
      warnings=warnings,config=cfg,question=r.get('case',{}).get('question',''),answer=o.get('draft',{}).get('answer',''),
      fields=fields,trace=trace,source=str(path),sha256=digest)

def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['batch'],r['scheme'])].append(r)
    result=[]
    for (batch,scheme),rs in sorted(groups.items()):
        n=len(rs);lat=[r['latency'] for r in rs if isinstance(r['latency'],(int,float))]
        counts=Counter(r['status'] for r in rs);conv=defaultdict(list)
        for r in rs:conv[(r['case'],r['repetition'])].append(r['status'])
        # Completeness of planned turns is checked at manifest level, not assumed here.
        result.append(dict(batch=batch,scheme=scheme,n=n,pass_n=counts['pass'],fail_n=counts['fail'],pending_n=counts['pending'],
          confirmed_success=pct(counts['pass'],n),possible_success=pct(counts['pass']+counts['pending'],n),
          completion=pct(sum(r['completed'] for r in rs),n),structure=pct(sum(r['structure'] for r in rs),n),
          coverage=pct(sum(r['fields_ok'] for r in rs),sum(r['fields_n'] for r in rs)),field_denominator=sum(r['fields_n'] for r in rs),
          supported_claims=pct(sum(r['claims_supported'] for r in rs),sum(r['claims_n'] for r in rs)),claims_denominator=sum(r['claims_n'] for r in rs),
          redundancy=pct(sum(r['redundant'] for r in rs),sum(r['calls_reviewed'] for r in rs)),reviewed_calls=sum(r['calls_reviewed'] for r in rs),
          median_seconds=quantile(lat,.5),p95_seconds=quantile(lat,.95),latency_n=len(lat),
          mean_model_calls=statistics.mean(r['model_calls'] for r in rs),mean_queries=statistics.mean(r['queries'] for r in rs),
          exact_duplicates=sum(r['duplicates'] for r in rs),timeout_rate=pct(sum(r['timeout'] for r in rs),n),
          errors=dict(Counter(r['error'] for r in rs if r['error'])),observed_conversations=len(conv)))
    return result

DEFINITIONS=[
 ('确认成功率','人工三项均pass且无自动字段失败的轮次 / 全部已运行轮次。待复核不算成功。','展开失败轮次→区分工具、网关、引用、任务对象错误。'),
 ('成功率上下界','下界=确认成功/N，上界=(确认成功+待复核)/N；不是统计置信区间。','先完成待复核，不能用上界宣传准确率。'),
 ('字段覆盖率','正确结构化目标字段数 / 全部自动目标字段数；无自动字段的题不入分母。','查看field_checks及工具对象；文字答对但未提交观察需另行说明。'),
 ('陈述证据支持率','人工标注supported陈述 / 全部已标注陈述；未标注显示待复核。','逐条核对证据路径、语义、范围和反例；结构通过不代表语义通过。'),
 ('冗余调用率','人工标注duplicate/already_available/irrelevant调用 / 全部已标注查询；提交不计。','检查相同参数重复、初始证据已有结果及无关工具。'),
 ('补查次数','trace中stage=tool且非bootstrap条数；含失败尝试，不含提交与路由。','比对实际轨迹，避免用各方案口径不同的tool_calls直接比较。'),
 ('延迟中位数/P95','全部有耗时记录的轮次，包括失败；不含共享深度模型预计算。','查看每次模型与工具耗时；少量样本P95不代表稳定尾延迟。'),
 ('超时率','含APITimeoutError/TimeoutError的轮次 / 已运行轮次。','同证据单次生成与完整Agent分开诊断；重试另记，不覆盖首次失败。')]

def generate(root,out,batch=None):
    paths=sorted(Path(root).glob('*/result.json'));rows=[inspect(p) for p in paths]
    if batch:rows=[r for r in rows if r['batch']==batch]
    if not rows:raise ValueError('No matching records')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    summary=aggregate(rows);payload={'scope':'development engineering benchmark, not clinical validation','groups':summary,'rows':rows}
    (out/'metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    def esc(x):return html.escape(str(x))
    def fmt(v,percent=False):return '待复核 / 无分母' if v is None else (f'{v:.1%}' if percent else f'{v:.2f}')
    page=['<!doctype html><meta charset="utf-8"><title>ECG 面试测评 v1</title><style>body{font:16px system-ui;background:#f5f7fb;color:#17263c;max-width:1200px;margin:32px auto;padding:20px}section,details{background:white;padding:20px;margin:16px 0;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#53657b}meter{width:140px}</style><h1>ECG 三方案测评 · 面试版 v1</h1><p>真实运行记录；按批次分别汇总。待复核不当作成功，失败不删除。批次任务另列为可靠性案例。</p>']
    for b in sorted(set(r['batch'] for r in rows)):
        rs=[r for r in rows if r['batch']==b];schemes=set(r['scheme'] for r in rs)
        signatures={(r['config'].get('source_sha256'),r['config'].get('suite_sha256'),r['config'].get('model'),r['config'].get('data_sha256'),r['config'].get('timeout_seconds'),r['config'].get('max_tokens')) for r in rs}
        keys={s:{(r['case'],r['repetition'],r['turn']) for r in rs if r['scheme']==s} for s in schemes}
        paired=schemes=={'summary','rules','agent'} and len(signatures)==1 and all(v==next(iter(keys.values())) for v in keys.values())
        manifest_path=Path(root)/(b+'.manifest.json')
        planned=None;conversation_pass=0;conversation_pending=0
        if manifest_path.exists():
            manifest=load(manifest_path);tasks=manifest.get('tasks',[])
            planned=sum(t.get('turn_count',0) for t in tasks)
            for task in tasks:
                cr=[r for r in rs if r['case']==task['case_id'] and r['scheme']==task['scheme'] and r['repetition']==task['repetition']]
                full=len(cr)==task['turn_count'] and {r['turn'] for r in cr}==set(range(1,task['turn_count']+1))
                conversation_pass+=full and all(r['status']=='pass' for r in cr)
                conversation_pending+=full and not any(r['status']=='fail' for r in cr) and any(r['status']=='pending' for r in cr)
            page.append(f'<p>批次 {esc(b)}：计划{planned}轮，保存{len(rs)}轮；整段任务确认通过{conversation_pass}/{len(tasks)}，待复核{conversation_pending}。缺失记录未算通过。</p>')
        else:page.append('<p>未找到批次manifest：无法判定计划是否全部执行，不能宣称完整批次通过。</p>')
        page.append(f'<section><h2>批次 {esc(b)}</h2><p>三方案记录配对及主要配置检查：{"通过（仍需核对任务和人工评分）" if paired else "不完整或配置不同，禁止直接归因方案优劣"}</p><table><tr><th>方案 / N</th><th>成功 / 失败 / 待复核</th><th>确认成功率</th><th>字段覆盖</th><th>模型 / 补查均次</th><th>中位 / P95秒</th></tr>')
        for g in summary:
            if g['batch']!=b:continue
            page.append(f"<tr><td>{esc(g['scheme'])} / {g['n']}</td><td>{g['pass_n']} / {g['fail_n']} / {g['pending_n']}</td><td><meter min='0' max='1' value='{g['confirmed_success']}'></meter> {fmt(g['confirmed_success'],True)}</td><td>{fmt(g['coverage'],True)}（{g['field_denominator']}字段）</td><td>{fmt(g['mean_model_calls'])} / {fmt(g['mean_queries'])}</td><td>{fmt(g['median_seconds'])} / {fmt(g['p95_seconds'])}</td></tr>")
        page.append('</table><details><summary>完整指标、分母与失败类型</summary><pre>'+esc(json.dumps([g for g in summary if g['batch']==b],ensure_ascii=False,indent=2))+'</pre></details></section>')
    page.append('<section><h2>指标解释与定位</h2><table>')
    for row in DEFINITIONS:page.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>')
    page.append('</table></section><h2>逐轮复核</h2>')
    for r in rows:
        page.append('<details><summary>'+esc(f"{r['scheme']} · {r['case']} · 第{r['turn']}轮 · {r['status']} · {r['error']}")+'</summary><p>'+esc(r['question'])+'</p><pre>'+esc(r['answer'] or '未交付完整答案')+'</pre><p>来源：'+esc(r['source'])+'</p><pre>'+esc(json.dumps({'fields':r['fields'],'warnings':r['warnings'],'trace':r['trace']},ensure_ascii=False,indent=2))+'</pre></details>')
    (out/'report.html').write_text(''.join(page),encoding='utf-8');print(out/'report.html')
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',default='evaluation/interview_runs');p.add_argument('--batch');p.add_argument('--out',required=True);a=p.parse_args();generate(a.runs,a.out,a.batch)
