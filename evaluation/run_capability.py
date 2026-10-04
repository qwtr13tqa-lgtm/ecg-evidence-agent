"""Preview-first capability benchmark. Unique records, explicit external authorization."""
import argparse
import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path
from src.evaluation.records import RunRecord,atomic_json
from src.evaluation.capability import SCHEMES,EmptyRetriever,run_fixed,reference,score

ROOT=Path(__file__).resolve().parents[1]


def load_suite(path):
    cases=[json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({c['id'] for c in cases})!=len(cases):raise ValueError('Duplicate case IDs')
    for c in cases:
        if type(c['sample_index']) is not int or c['sample_index']<0 or not c['turns']:raise ValueError('Invalid case')
        for t in c['turns']:
            if not t.get('question') or len(t['question'])>4000:raise ValueError('Invalid question')
    return cases


class Meter:
    def __init__(self,gateway):self.gateway=gateway;self.requests=[]
    def complete(self,messages,tools):
        row={'logical_input_bytes':len(json.dumps({'messages':messages,'tools':tools},ensure_ascii=False).encode()),'status':'started'}
        self.requests.append(row);tick=time.perf_counter()
        try:
            out=self.gateway.complete(messages,tools);row['status']='returned';return out
        except Exception as exc:row.update(status='failed',error_type=type(exc).__name__);raise
        finally:row['elapsed_seconds']=time.perf_counter()-tick


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases-file',default=str(ROOT/'evaluation/capability_cases.jsonl'))
    p.add_argument('--case',action='append');p.add_argument('--all',action='store_true')
    p.add_argument('--schemes',nargs='+',choices=SCHEMES,default=list(SCHEMES[:5]))
    p.add_argument('--repeat',type=int,default=1);p.add_argument('--allow-external',action='store_true')
    a=p.parse_args();cases=load_suite(a.cases_file)
    if a.all and a.case:p.error('Choose --all or --case')
    if not 1<=a.repeat<=5 or len(set(a.schemes))!=len(a.schemes):p.error('Invalid repeat or duplicate schemes')
    if set(a.case or [])-{c['id'] for c in cases}:p.error('Unknown case ID')
    chosen=cases if a.all else [c for c in cases if c['id'] in (a.case or [])]
    if not chosen:
        for c in cases:print(c['id'],c['split'],len(c['turns']),'turns',c['turns'][0]['question'])
        return
    bound=sum(len(c['turns']) for c in chosen)*a.repeat*sum(4 if s.startswith('agent') else 1 for s in a.schemes)
    print('At most',bound,'model requests. No retries. Local inference once per case/repetition.')
    if not a.allow_external:print('Preview only. --allow-external sends structured ECG context/questions to your configured gateway.');return
    os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    import numpy as np
    from src.analysis.pipeline import ECGAnalysisPipeline
    from src.agent.gateway import ToolGateway
    from src.agent.evidence_agent import ECGEvidenceAgent
    from src.agent.conversation import new_conversation,begin_turn,finish_turn,ConversationGateway
    from src.knowledge.retriever import BM25Retriever
    root=ROOT/'evaluation/capability_runs';root.mkdir(parents=True,exist_ok=True)
    batch=str(uuid.uuid4());rng=random.Random(17);tasks=[]
    for rep in range(1,a.repeat+1):
        for c in chosen:
            order=list(a.schemes);rng.shuffle(order)
            for scheme in order:tasks.append({'case_id':c['id'],'scheme':scheme,'repetition':rep,'turn_count':len(c['turns'])})
    h=hashlib.sha256()
    for f in sorted(list((ROOT/'src').rglob('*.py'))+list((ROOT/'evaluation').glob('*.py'))):h.update(f.relative_to(ROOT).as_posix().encode());h.update(f.read_bytes())
    config={'batch_id':batch,'protocol':'capability-1.0','source_sha256':h.hexdigest(),
        'suite_sha256':hashlib.sha256(Path(a.cases_file).read_bytes()).hexdigest(),
        'model':os.getenv('ECG_MODEL',''),'timeout_seconds':180,'max_tokens':1800,
        'sdk_retries':0,'temperature':0,'memory':'existing last-3-successful-turns adapter',
        'endpoint_sha256':hashlib.sha256(os.getenv('ECG_BASE_URL','').encode()).hexdigest()}
    atomic_json(root/(batch+'.manifest.json'),{'config':config,'tasks':tasks});print('Batch:',batch,flush=True)
    pipeline=ECGAnalysisPipeline();data=np.load(ROOT/'data/Processed_PTBXL/test.npy',mmap_mode='r',allow_pickle=False)
    corpus=ROOT/'data/knowledge/ecg_knowledge.jsonl'
    real_retriever=BM25Retriever.from_jsonl(corpus) if 'agent_rag' in a.schemes else None
    byid={c['id']:c for c in chosen};previous=None;result=None;ref={};failure=None
    for task in tasks:
        c=byid[task['case_id']];key=(task['repetition'],c['id'])
        if key!=previous:
            if result is not None:pipeline.store.discard(result.analysis_id)
            result=None;ref={};failure=None;previous=key
            try:
                result=pipeline.analyze(data[c['sample_index'],100:4900,:],source_id='test.npy',sample_index=c['sample_index'],crop_start_sample=100)
                ref=reference(result,**c.get('reference_window',{}))
            except Exception as exc:failure={'stage':'analysis','error_type':type(exc).__name__}
        session=new_conversation(result.analysis_id if result else 'unavailable')
        for ordinal,t in enumerate(c['turns'],1):
            case={**c,'question':t['question'],'turn_index':ordinal,'expectation':{'kind':t['kind']}}
            record=RunRecord(root,case,{**config,**task,'max_model_calls':4 if task['scheme'].startswith('agent') else 1,
                'input_sha256':result.provenance.get('input_sha256') if result else None,
                'scoring_profile':result.provenance.get('score_profile') if result else None,
                'corpus_sha256':hashlib.sha256(corpus.read_bytes()).hexdigest() if real_retriever else None})
            record.base['dataset_role']=c['split']
            out={'analysis_id':ref.get('analysis_id'),'status':'failed','error':'ANALYSIS_FAILED','draft':{},'evidence':{},'trace':[],'model_calls':0,'tool_calls':0}
            gateway=None;meter=None;local_failure=failure;interrupted=False;tick=time.perf_counter()
            try:
                if result is not None and failure is None:
                    rid=begin_turn(session,result.analysis_id,t['question'])
                    gateway=ToolGateway();meter=Meter(gateway);wrapped=ConversationGateway(meter,session,result.analysis_id)
                    if task['scheme'].startswith('agent'):
                        out=ECGEvidenceAgent(pipeline.store,real_retriever if task['scheme']=='agent_rag' else EmptyRetriever(),wrapped).run(result.analysis_id,t['question'],allow_external=True)
                    else:out=run_fixed(pipeline.store,result.analysis_id,t['question'],task['scheme'],wrapped)
                    finish_turn(session,result.analysis_id,rid,out)
            except (Exception,KeyboardInterrupt) as exc:
                local_failure={'stage':'runner','error_type':type(exc).__name__};out.update(status='failed',error='RUNNER_FAILURE',draft={},validation={})
                interrupted=isinstance(exc,KeyboardInterrupt)
                if session.get('pending'):
                    finish_turn(session,session['analysis_id'],session['pending']['request_id'],{**out,'analysis_id':session['analysis_id']})
            finally:
                if gateway:
                    try:gateway.close()
                    except Exception:pass
            scored=score(t['kind'],out,ref) if ref else {'metrics':{'completed_draft':False},'scope':'analysis unavailable'}
            scored['request_measurements']=meter.requests if meter else []
            record.finish(out,ref,scored,failure=local_failure,wall_seconds=time.perf_counter()-tick)
            print(c['id'],task['scheme'],'turn',ordinal,'run',record.run_id,out['status'],flush=True)
            if interrupted:return
    if result:pipeline.store.discard(result.analysis_id)

if __name__=='__main__':main()
