"""Controlled complex tasks. Preview by default; no changes to production or old records."""
import argparse
import hashlib
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from src.evaluation.complex_cases_v1 import QUESTIONS,reference,score
from src.evaluation.complex_plan_v1 import ALLOWED,run_plan,response_metadata,PROTOCOL_VERSION
ROOT=Path(__file__).resolve().parents[1]

class Meter:
    def __init__(self,gateway):self.gateway=gateway;self.requests=[];self.queries=0
    def complete(self,messages,tools):
        # Identical visible query tools; production Agent may have more tools outside this benchmark.
        tools=[t for t in tools if t['function']['name'] in ALLOWED|{'submit_answer','submit_plan'}]
        offered={t['function']['name'] for t in tools}
        row={'offered_tools':sorted(offered),'logical_input_bytes':len(json.dumps([messages,tools],ensure_ascii=False).encode()),'status':'started'}
        self.requests.append(row);tick=time.perf_counter()
        try:
            if len(self.requests)>4:raise ValueError('MODEL_BUDGET_EXHAUSTED')
            reply=self.gateway.complete(messages,tools)
            row['response_protocol']=response_metadata(reply)
            if not isinstance(reply,dict):raise ValueError('REPLY_NOT_OBJECT')
            calls=reply.get('tool_calls',[])
            if not isinstance(calls,list):raise ValueError('TOOL_CALLS_NOT_LIST')
            for call in calls:
                if not isinstance(call,dict) or not isinstance(call.get('function'),dict):raise ValueError('TOOL_CALL_SCHEMA_INVALID')
                name=call['function'].get('name')
                if not isinstance(name,str) or name not in offered:raise ValueError('TOOL_NOT_OFFERED_IN_THIS_PHASE')
                if name in ALLOWED:self.queries+=1
            if self.queries>6:raise ValueError('TOOL_BUDGET_EXHAUSTED')
            row['status']='returned';return reply
        except Exception as exc:
            row.update(status='failed',error_type=type(exc).__name__)
            if re.fullmatch('[A-Z_]{1,100}',str(exc)):row['error_code']=str(exc)
            raise
        finally:row['elapsed_seconds']=time.perf_counter()-tick


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',action='append',choices=list(QUESTIONS));p.add_argument('--index',type=int,action='append')
    p.add_argument('--schemes',nargs='+',choices=['plan','agent','rules'],default=['plan','agent'])
    p.add_argument('--repeat',type=int,default=1);p.add_argument('--allow-external',action='store_true')
    a=p.parse_args();cases=a.case or list(QUESTIONS);indices=a.index or [0]
    if not 1<=a.repeat<=5 or any(i<0 for i in indices):p.error('Invalid repeat or index')
    if len(set(a.schemes))!=len(a.schemes):p.error('Duplicate scheme')
    count=len(cases)*len(indices)*a.repeat
    print('Cases:',cases,'indices:',indices,'schemes:',a.schemes)
    print('At most',count*sum({'plan':2,'agent':4,'rules':1}[s] for s in a.schemes),'model requests; 6 supplemental queries/run; no SDK retries.')
    if not a.allow_external:
        print('Preview only. --allow-external sends questions and structured evidence to configured gateway.');return
    from src.analysis.pipeline import ECGAnalysisPipeline
    from src.agent.gateway import ToolGateway
    from src.agent.evidence_agent import ECGEvidenceAgent
    from src.evaluation.capability import EmptyRetriever
    from src.evaluation.threeway_baselines import run_fixed
    from src.evaluation.records import RunRecord,atomic_json
    import numpy as np
    os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    root=ROOT/'evaluation/complex_runs_v1';root.mkdir(parents=True,exist_ok=True)
    batch=str(uuid.uuid4());path=ROOT/'data/Processed_PTBXL/test.npy'
    data=np.load(path,mmap_mode='r',allow_pickle=False)
    if any(i>=len(data) for i in indices):p.error('Index outside data')
    h=hashlib.sha256()
    for f in sorted((ROOT/'src').rglob('*.py')):h.update(f.relative_to(ROOT).as_posix().encode());h.update(f.read_bytes())
    cfg={'batch_id':batch,'protocol':'complex-v1.2','plan_protocol_version':PROTOCOL_VERSION,'model':os.getenv('ECG_MODEL','DeepSeek-V4-Flash-0731-W8A8'),
         'timeout_seconds':90,'max_tokens':6000,'source_sha256':h.hexdigest(),'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'endpoint_sha256':hashlib.sha256(os.getenv('ECG_BASE_URL','http://aigw.dlut.edu.cn/v1').encode()).hexdigest(),
         'tool_scope':sorted(ALLOWED),'sdk_retries':0,'rules_role':'frozen coverage baseline; not general rules capability',
         'plan_role':'one LLM plan + bounded deterministic execution + one LLM answer',
         'agent_role':'installed production policy, at most4 requests including repairs; same query cap6',
         'primary_unit':'case/sample/repetition','temperature':0}
    tasks=[];rng=random.Random(17)
    for repeat in range(1,a.repeat+1):
        for index in indices:
            for kind in cases:
                schemes=list(a.schemes);rng.shuffle(schemes)
                for scheme in schemes:tasks.append(dict(kind=kind,index=index,repetition=repeat,scheme=scheme))
    atomic_json(root/(batch+'.manifest.json'),{'config':cfg,'tasks':tasks});print('Batch:',batch,flush=True)
    pipeline=ECGAnalysisPipeline();previous=None;result=None;ref=None;ineligible=None
    for task in tasks:
        key=(task['kind'],task['index'],task['repetition'])
        if key!=previous:
            if result is not None:pipeline.store.discard(result.analysis_id)
            result=None;ref=None;ineligible=None;previous=key
            try:
                result=pipeline.analyze(data[task['index'],100:4900,:],source_id='test.npy',sample_index=task['index'],crop_start_sample=100)
                ref=reference(result,task['kind'])
            except Exception as exc:ineligible={'error_type':type(exc).__name__,'reason':str(exc)[:160]}
        question=QUESTIONS[task['kind']]
        record=RunRecord(root,{'id':task['kind'],'question':question,'sample_index':task['index']},
            {**cfg,**task,'analysis_provenance':result.provenance if result is not None else None})
        out={'analysis_id':result.analysis_id if result else None,'status':'failed','error':'REFERENCE_UNAVAILABLE','draft':{},'evidence':{},'trace':[]}
        gateway=None;meter=None;tick=time.perf_counter()
        try:
            if ref is not None:
                gateway=ToolGateway(timeout=90,max_tokens=6000);meter=Meter(gateway)
                if task['scheme']=='plan':out=run_plan(pipeline.store,result.analysis_id,question,meter)
                elif task['scheme']=='agent':
                    out=ECGEvidenceAgent(pipeline.store,EmptyRetriever(),meter,max_model_calls=4,max_tool_calls=10).run(result.analysis_id,question,allow_external=True)
                else:out=run_fixed(pipeline.store,result.analysis_id,question,'rules',meter)
        except Exception as exc:out.update(status='failed',error='RUNNER_FAILURE',failure_type=type(exc).__name__)
        finally:
            if gateway:
                try:gateway.close()
                except Exception:pass
        scored=score(out,ref) if ref else {'ineligible':ineligible,'automatic_task_pass':False,'target_covered':0,'target_total':0}
        scored['request_measurements']=meter.requests if meter else []
        scored['model_requests']=len(meter.requests) if meter else 0
        record.finish(out,ref or {},scored,failure=ineligible,wall_seconds=time.perf_counter()-tick)
        print(task['kind'],task['scheme'],record.run_id,out['status'],flush=True)
    if result is not None:pipeline.store.discard(result.analysis_id)
    print('Report: python -m evaluation.report_complex_v1 --batch',batch)
if __name__=='__main__':main()
