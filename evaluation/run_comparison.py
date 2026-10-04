"""Paired development comparison, unique immutable records. No calls by default."""
import argparse
import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path
from src.evaluation.cases import load_cases
from src.evaluation.records import RunRecord, atomic_json
from src.evaluation.checks import build_reference, score_output
from src.evaluation.baselines import SCHEMES, run_baseline

ROOT=Path(__file__).resolve().parents[1]

def fingerprint():
    h=hashlib.sha256()
    for folder in ('src','evaluation'):
        for p in sorted((ROOT/folder).rglob('*.py')):
            h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()

def plan(cases, schemes, repeat, seed):
    rng=random.Random(seed); tasks=[]
    for rep in range(1,repeat+1):
        for case in cases:
            order=list(schemes); rng.shuffle(order)
            tasks.append((rep,case,order))
    return tasks

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases-file',default=str(ROOT/'evaluation/coverage_cases.jsonl'))
    p.add_argument('--case',action='append'); p.add_argument('--all',action='store_true')
    p.add_argument('--schemes',nargs='+',choices=SCHEMES,default=list(SCHEMES))
    p.add_argument('--repeat',type=int,default=1); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--allow-external',action='store_true')
    p.add_argument('--runs-dir',default=str(ROOT/'evaluation/comparison_runs'))
    a=p.parse_args(); cases,suite_hash=load_cases(a.cases_file)
    if a.case and a.all: p.error('Choose --case or --all')
    if not 1<=a.repeat<=10 or len(set(a.schemes))!=len(a.schemes): p.error('Invalid repetition or duplicate scheme')
    ids=set(a.case or [])
    if ids-{c['id'] for c in cases}: p.error('Unknown case')
    chosen=cases if a.all else [c for c in cases if c['id'] in ids]
    if not chosen:
        for c in cases: print(c['id'],'| sample',c['sample_index'],'|',c['question'])
        return 0
    tasks=plan(chosen,a.schemes,a.repeat,a.seed)
    bound=len(tasks)*sum(4 if s=='agent' else 1 for s in a.schemes)
    print(f'{len(tasks)*len(a.schemes)} runs; at most {bound} model requests; no retries.')
    if not a.allow_external:
        print('Preview only. Add --allow-external to send structured real analysis.'); return 0
    os.environ['LANGSMITH_TRACING']='false'; os.environ['LANGCHAIN_TRACING_V2']='false'
    import numpy as np
    from src.analysis.pipeline import ECGAnalysisPipeline
    from src.agent.gateway import ToolGateway
    from src.agent.evidence_agent import ECGEvidenceAgent
    from src.knowledge.retriever import BM25Retriever
    corpus=ROOT/'data/knowledge/ecg_knowledge.jsonl'
    batch=str(uuid.uuid4()); root=Path(a.runs_dir); root.mkdir(parents=True,exist_ok=True)
    config=dict(protocol='comparison-1.0',batch_id=batch,seed=a.seed,suite_sha256=suite_hash,
                source_sha256=fingerprint(),corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                model=os.getenv('ECG_MODEL','DeepSeek-V4-Flash-0731-W8A8'),temperature=0,
                timeout_seconds=180,max_tokens=1800,sdk_retries=0,
                endpoint_sha256=hashlib.sha256(os.getenv('ECG_BASE_URL','http://aigw.dlut.edu.cn/v1').encode()).hexdigest())
    manifest={'config':config,'tasks':[{'repetition':r,'case_id':c['id'],'schemes':s} for r,c,s in tasks]}
    atomic_json(root/(batch+'.manifest.json'),manifest)
    print('Batch:',batch,flush=True)
    pipeline=ECGAnalysisPipeline(); retriever=BM25Retriever.from_jsonl(corpus)
    data=np.load(ROOT/'data/Processed_PTBXL/test.npy',mmap_mode='r')
    for repetition,case,order in tasks:
        # One inference snapshot shared by all schemes in this case/repetition.
        result=None; reference={}; analysis_failure=None
        try:
            if case['sample_index']>=len(data): raise ValueError('sample out of range')
            result=pipeline.analyze(data[case['sample_index'],100:4900,:],source_id='test.npy',
                                    sample_index=case['sample_index'],crop_start_sample=100)
            reference=build_reference(result)
        except Exception as exc:
            analysis_failure={'stage':'analysis','error_type':type(exc).__name__}
        for scheme in order:
            record=RunRecord(root,case,{**config,'scheme':scheme,'repetition':repetition,
                'max_model_calls':4 if scheme=='agent' else 1,'max_tool_calls':6 if scheme=='agent' else 5})
            print(case['id'],scheme,'run_id='+record.run_id,flush=True)
            started=time.perf_counter(); gateway=None; failure=analysis_failure; interrupted=False
            output={'analysis_id':reference.get('analysis_id'),'status':'failed','error':'ANALYSIS_FAILED',
                    'draft':{},'evidence':{},'knowledge':{},'trace':[],'model_calls':0,'tool_calls':0}
            if failure is None:
                try:
                    record.checkpoint('generation'); gateway=ToolGateway(timeout=180,max_tokens=1800)
                    if scheme=='agent':
                        output=ECGEvidenceAgent(pipeline.store,retriever,gateway).run(result.analysis_id,case['question'],allow_external=True)
                    else:
                        output=run_baseline(pipeline.store,retriever,gateway,result.analysis_id,case['question'],scheme)
                except (Exception,KeyboardInterrupt) as exc:
                    failure={'stage':'generation','error_type':type(exc).__name__}; output['error']='RUNNER_FAILURE'
                    interrupted=isinstance(exc,KeyboardInterrupt)
                finally:
                    if gateway:
                        try: gateway.close()
                        except Exception: pass
            try:
                scores=score_output(case,output,reference)
                # Tool selection is a policy-specific diagnostic, never a common answer score.
                scores['comparison_note']='Do not rank summary baselines by required_tool_selected; use manual answer scores.'
            except Exception as exc:
                scores={'metrics':{},'scoring_error_type':type(exc).__name__}
            record.finish(output,reference,scores,failure=failure,wall_seconds=time.perf_counter()-started)
            print('Saved',record.path,'status='+output['status'],flush=True)
            if interrupted: return 130
        if result: pipeline.store.discard(result.analysis_id)
    return 0

if __name__=='__main__': raise SystemExit(main())
