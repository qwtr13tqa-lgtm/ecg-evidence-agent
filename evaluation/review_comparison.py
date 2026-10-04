"""Inspect a run or save explicit human ratings; never modify result.json."""
import argparse
import json
from pathlib import Path
from src.evaluation.records import read_review, atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--reviewer');p.add_argument('--notes',default='')
    for k in ('task-correct','evidence-support','text-complete'):p.add_argument('--'+k,choices=('pass','fail'))
    a=p.parse_args(); path=Path(a.run); read_review(path)
    rec=json.loads((path/'result.json').read_text())
    print(json.dumps({'case':rec['case'],'scheme':rec['config'].get('scheme'),'draft':rec['output'].get('draft'),
                      'automatic':rec['automatic']},ensure_ascii=False,indent=2))
    values={k:getattr(a,k) for k in ('task_correct','evidence_support','text_complete')}
    if any(v is not None for v in values.values()):
        if not a.reviewer or not a.reviewer.strip():p.error('--reviewer required')
        review=json.loads((path/'review.json').read_text());review.update(reviewer=a.reviewer,notes=a.notes)
        review.update({k:v for k,v in values.items() if v is not None})
        atomic_json(path/'review.json',review);print('Human review saved; original result unchanged.')
if __name__=='__main__':main()
