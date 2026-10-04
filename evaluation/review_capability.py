"""Import explicit human scores, bound to immutable result hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from src.evaluation.records import atomic_json
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--file',required=True);a=p.parse_args()
    items=json.loads(Path(a.file).read_text(encoding="utf-8"));writes=[];seen=set()
    for item in items:
        rid=item['run_id']
        import uuid
        if str(uuid.UUID(rid))!=rid or rid in seen:raise ValueError('Invalid/duplicate run ID')
        seen.add(rid);path=ROOT/'evaluation/capability_runs'/rid
        if hashlib.sha256((path/'result.json').read_bytes()).hexdigest()!=item['result_sha256']:raise ValueError('Result changed')
        scores={k:item[k] for k in ('task_correct','evidence_support','text_complete')}
        if any(v not in (None,'pass','fail') for v in scores.values()):raise ValueError('Use null/pass/fail')
        if all(v is None for v in scores.values()):continue
        if not isinstance(item['reviewer'],str) or not item['reviewer'].strip():raise ValueError('Reviewer required')
        if not isinstance(item['notes'],str):raise ValueError('Notes must be text')
        writes.append((path/'review.json',{'run_id':rid,'result_sha256':item['result_sha256'],
             'reviewer':item['reviewer'],'notes':item['notes'],**scores}))
    for path,review in writes:atomic_json(path,review)
    print('Imported',len(writes),'human reviews. Original result.json unchanged.')

if __name__=='__main__':main()
