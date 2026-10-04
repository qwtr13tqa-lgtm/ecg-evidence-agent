"""Read-only input aggregation. Missing runs/reviews stay explicitly missing."""
import argparse
import json
from pathlib import Path
from src.evaluation.records import read_review, atomic_json

def summarize(root,batch):
    root=Path(root); manifest=json.loads((root/(batch+'.manifest.json')).read_text())
    groups={}; seen=set(); invalid=[]
    for task in manifest['tasks']:
        for scheme in task['schemes']:
            groups.setdefault(scheme,{'planned':0,'recorded':0,'completed':0,'elapsed_seconds_sum':0,
                                     'model_calls':0,'query_tool_attempts':0,
                                     'reviews':{k:{'pass':0,'fail':0,'unscored':0} for k in ('task_correct','evidence_support','text_complete')}})['planned']+=1
    for path in sorted(root.glob('*/result.json')):
        try:
            rec=json.loads(path.read_text()); cfg=rec['config']
            if cfg.get('batch_id')!=batch: continue
            scheme=cfg['scheme']; key=(rec['case']['id'],cfg['repetition'],scheme)
            if key in seen: raise ValueError('duplicate paired run')
            expected=any(t['case_id']==key[0] and t['repetition']==key[1] and scheme in t['schemes'] for t in manifest['tasks'])
            if not expected: raise ValueError('unplanned run')
            for k,v in manifest['config'].items():
                if cfg.get(k)!=v: raise ValueError('configuration mismatch')
            review=read_review(path.parent)
            out=rec['output']; g=groups[scheme]; seen.add(key)
            g['recorded']+=1; g['completed']+=out.get('status')=='completed_draft'
            g['elapsed_seconds_sum']+=out.get('elapsed_seconds',rec.get('wall_seconds') or 0)
            g['model_calls']+=out.get('model_calls',0)
            g['query_tool_attempts']+=sum(x.get('stage')=='tool' for x in out.get('trace',[]))
            for k,v in g['reviews'].items():
                score=review.get('scores',{}).get(k)
                v[score if score in ('pass','fail') else 'unscored']+=1
        except Exception as exc: invalid.append({'file':str(path),'error_type':type(exc).__name__})
    for g in groups.values():
        g['missing_runs']=g['planned']-g['recorded']
        g['completion_rate_recorded']=g['completed']/g['recorded'] if g['recorded'] else None
        g['mean_elapsed_seconds_recorded']=g['elapsed_seconds_sum']/g['recorded'] if g['recorded'] else None
    return {'batch_id':batch,'groups':groups,'invalid_records':invalid,
            'scope':'development; manual scores required; incomplete pairs are not a fair final comparison'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runs-dir',default='evaluation/comparison_runs');p.add_argument('--batch',required=True)
    a=p.parse_args(); report=summarize(a.runs_dir,a.batch)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    atomic_json(Path(a.runs_dir)/(a.batch+'.summary.json'),report)
if __name__=='__main__':main()
