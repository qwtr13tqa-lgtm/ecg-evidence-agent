"""Offline sidecar scoring, preserves original result and review hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from src.evaluation.checks_v2 import score_output

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True)
    p.add_argument('--runs-dir',default='evaluation/comparison_runs');a=p.parse_args()
    found=0
    for path in sorted(Path(a.runs_dir).glob('*/result.json')):
        raw=path.read_bytes();rec=json.loads(raw)
        if rec.get('config',{}).get('batch_id')!=a.batch:continue
        new=score_output(rec['case'],rec['output'],rec['reference'])
        payload={'run_id':rec['run_id'],'result_sha256':hashlib.sha256(raw).hexdigest(),
                 'automatic':new,'original_version':rec.get('automatic',{}).get('version')}
        dest=path.parent/'automatic_v2.json'
        text=json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)
        if dest.exists():
            if json.loads(dest.read_text())!=payload:raise ValueError('Existing v2 sidecar differs: '+str(dest))
        else:
            with dest.open('x',encoding='utf-8') as f:f.write(text)
        found+=1
        print(json.dumps({'scheme':rec['config'].get('scheme'),'run_id':rec['run_id'],
                         'old_task_observations':rec['automatic']['metrics'].get('task_observations'),
                         'new_task_observations':new['metrics'].get('task_observations'),
                         'rr_field_checks':new.get('rr_field_checks')},ensure_ascii=False,indent=2))
    if not found:raise SystemExit('No matching batch records')
    print('Offline rescore complete. result.json and review.json unchanged; comparison summary still uses original metrics.')
if __name__=='__main__':main()
