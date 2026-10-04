"""Report completed, failed, missing and unreviewed runs without selection bias."""
import argparse
import hashlib
import json
import statistics
import uuid
from pathlib import Path
from src.evaluation.records import atomic_json,read_review

ROOT=Path(__file__).resolve().parents[1]


def collect(root,batch):
    root=Path(root);manifest=json.loads((root/(batch+'.manifest.json')).read_text(encoding="utf-8"))
    planned={(t['case_id'],t['scheme'],t['repetition'],i) for t in manifest['tasks'] for i in range(1,t['turn_count']+1)}
    records={};invalid=[];invalid_details=[]
    for path in root.glob('*/result.json'):
        try:r=json.loads(path.read_text(encoding="utf-8"))
        except (ValueError,OSError) as exc:
            invalid.append(str(path))
            detail={'path':str(path),'error_type':type(exc).__name__}
            if isinstance(exc,UnicodeDecodeError):detail.update(encoding=exc.encoding,start=exc.start,reason=exc.reason)
            elif isinstance(exc,json.JSONDecodeError):detail.update(line=exc.lineno,column=exc.colno,reason=exc.msg)
            else:detail['errno']=getattr(exc,'errno',None)
            invalid_details.append(detail)
            continue
        c=r.get('config',{})
        if c.get('batch_id')!=batch:continue
        key=(r['case']['id'],c['scheme'],c['repetition'],r['case']['turn_index'])
        if key not in planned or key in records:raise ValueError('Unplanned or duplicate run')
        records[key]=(r,path)
    groups={};packet=[];case_results=[]
    for key in sorted(planned):
        scheme=key[1];g=groups.setdefault(scheme,{'planned':0,'recorded':0,'completed':0,'missing':0,
            'task_observations':{'pass':0,'fail':0,'unassessed':0},'elapsed':[],'model_requests':0,'logical_input_bytes':0,'query_tool_attempts':0,
            'manual':{k:{'pass':0,'fail':0,'unscored':0} for k in ('task_correct','evidence_support','text_complete')},'failures':{}})
        g['planned']+=1
        if key not in records:
            case_results.append({'case_id':key[0],'scheme':scheme,'repetition':key[2],'turn':key[3],'status':'missing'})
            g['missing']+=1
            for counts in g['manual'].values():counts['unscored']+=1
            continue
        r,path=records[key];g['recorded']+=1;out=r['output'];auto=r.get('automatic') or {}
        g['completed']+=out.get('status')=='completed_draft'
        g['query_tool_attempts']+=sum(t.get('stage')=='tool' for t in out.get('trace',[]))
        case_results.append({'case_id':key[0],'scheme':scheme,'repetition':key[2],'turn':key[3],
            'split':r.get('dataset_role'),'status':out.get('status'),'run_id':r['run_id'],
            'metrics':auto.get('metrics',{}),'field_checks':auto.get('field_checks',{}),'wall_seconds':r.get('wall_seconds')})
        value=auto.get('metrics',{}).get('task_observations')
        g['task_observations']['pass' if value is True else 'fail' if value is False else 'unassessed']+=1
        if type(r.get('wall_seconds')) in (int,float):g['elapsed'].append(r['wall_seconds'])
        measures=auto.get('request_measurements',[]);g['model_requests']+=len(measures)
        g['logical_input_bytes']+=sum(x['logical_input_bytes'] for x in measures)
        if out.get('status')!='completed_draft':
            reason=(r.get('runner_failure') or {}).get('error_type') or auto.get('error_type') or out.get('error') or 'unknown'
            g['failures'][reason]=g['failures'].get(reason,0)+1
        review=read_review(path.parent)
        for k,counts in g['manual'].items():counts[review['scores'].get(k) or 'unscored']+=1
        packet.append({'run_id':r['run_id'],'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'case_id':key[0],'scheme':scheme,'turn':key[3],'question':r['case']['question'],
            'answer':out.get('draft',{}).get('answer'),'status':out.get('status'),
            'reference':r['reference'],'rubric':r['case']['rubric'],
            'evidence':out.get('evidence',{}),'knowledge':out.get('knowledge',{}),
            'reviewer':review.get('reviewer',''),'notes':review.get('notes',''),
            **{k:review['scores'].get(k) for k in ('task_correct','evidence_support','text_complete')}})
    for g in groups.values():
        times=g.pop('elapsed');g['mean_elapsed_recorded']=statistics.mean(times) if times else None
        g['completion_rate_planned']=g['completed']/g['planned']
    return {'batch_id':batch,'groups':groups,'case_results':case_results,'invalid_records':invalid,'invalid_record_details':invalid_details,
        'scope':'engineering capability evaluation; no clinical validation; evaluation_candidate is not an independent patient test set'},packet


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);a=p.parse_args()
    report,packet=collect(ROOT/'evaluation/capability_runs',a.batch)
    dest=ROOT/'evaluation/capability_reports'/str(uuid.uuid4());dest.mkdir(parents=True)
    atomic_json(dest/'report.json',report);atomic_json(dest/'manual_review.json',packet)
    lines=['# 能力对照评测','', '未评分不等于通过；失败和缺失均保留。','',
           '|方案|完成/计划轮数|缺失|平均耗时（含失败）|结构化任务字段通过/失败/未评|',
           '|---|---:|---:|---:|---|']
    for name,g in report['groups'].items():
        f=g['task_observations'];mean=g['mean_elapsed_recorded']
        lines.append(f"|{name}|{g['completed']}/{g['planned']}|{g['missing']}|{mean if mean is not None else '未提供'}|{f['pass']}/{f['fail']}/{f['unassessed']}|")
    if report['invalid_records']:
        lines+=['','注意：存在无法读取的记录；missing包含尚未匹配的记录，不能当作没有运行。具体原因见report.json中的invalid_record_details。']
    lines+=['','人工评分及失败分类见report.json；编辑manual_review.json后用evaluation.review_capability导入。',
        '纯LLM只获得输入元信息；这是专业证据可用性对照，不是等信号输入的异常检测性能比较。',
        '未实现Token计费统计；logical_input_bytes不是Token数。固定方案1请求/轮，Agent最多4请求/轮，非等预算实验。']
    (dest/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Saved',dest)
    if report['invalid_record_details']:print(json.dumps(report['invalid_record_details'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
