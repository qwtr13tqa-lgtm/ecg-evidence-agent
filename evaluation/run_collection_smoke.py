"""Explicit opt-in live smoke tests. Each failure retained; no automatic retry."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import uuid
from src.review.collection_runtime import run_collection
from src.review.collection_ops import base_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--allow-external',action='store_true');args=p.parse_args()
    path=Path(args.report);raw=path.read_bytes();report=json.loads(raw);rows=base_rows(report)
    questions=[('FN_TOP5','找出所有漏报中异常分数最低的5条记录。'),('CATEGORY_COUNTS','按FN、TP、FP、TN分组，统计本批每类记录数量。'),('AMBIGUOUS','找出所有漏报中最值得关注的几条记录。')]
    print('At most 3 model requests; report counts and questions sent; no ECG waveform; no retries.')
    if not args.allow_external:
        print('Preview only. Add --allow-external to run.');return
    from src.agent.gateway import ToolGateway
    directory=Path('evaluation/collection_runs')/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);directory.mkdir(parents=True,exist_ok=False)
    gateway=ToolGateway(timeout=90,max_tokens=2500)
    try:
        results=[]
        for case,question in questions:
            out=run_collection(None,gateway,'smoke-'+uuid.uuid4().hex,question,report,hashlib.sha256(raw).hexdigest(),None,max_model_calls=1)
            ev=[e['data'] for e in out['evidence'].values() if e.get('scope')=='batch_collection']
            checks={'terminal_draft':out['status']=='completed_draft','at_most_one_model':out['model_calls']<=1,'no_record_queries':out['tool_calls']==0}
            if case=='FN_TOP5':
                expected=sorted([r for r in rows if r['category']=='fn'],key=lambda r:(r['score'],r['index']))[:5]
                actual=[r for d in ev for r in d.get('rows',[])]
                checks['exact_indices']=[r['index'] for r in actual]==[r['index'] for r in expected]
                checks['exact_scores']=[r.get('score') for r in actual]==[r['score'] for r in expected]
            elif case=='CATEGORY_COUNTS':
                actual={g['group_value']:g['count'] for d in ev for g in d.get('groups',[])}
                expected={c:sum(r['category']==c for r in rows) for c in ('fn','tp','fp','tn') if any(r['category']==c for r in rows)}
                checks['exact_counts']=actual==expected
            else:checks['clarification']=len(ev)==1 and ev[0].get('query_status')=='needs_clarification'
            record={'case':case,'question':question,'model':gateway.model,'report_sha256':hashlib.sha256(raw).hexdigest(),'output':out,'automatic':checks,'manual_required':['query_conditions_match','clarification_appropriate']}
            (directory/(case+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');results.append({'case':case,**checks,'model_calls':out['model_calls'],'seconds':out['elapsed_seconds']})
            print(case,json.dumps(checks,ensure_ascii=False))
        (directory/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:gateway.close()
    print('Report:',directory)
if __name__=='__main__':main()
