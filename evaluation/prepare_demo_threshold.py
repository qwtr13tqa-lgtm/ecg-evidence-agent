"""Compute and optionally activate a disclosed development threshold; no LLM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from evaluation.select_anomaly_threshold import select_threshold


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def activate(config,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        backup=path.with_name(path.name+'.backup-'+str(uuid.uuid4()))
        backup.write_bytes(path.read_bytes())
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='threshold-',suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(config,f,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/Processed_PTBXL/test.npy')
    p.add_argument('--labels',default='data/Processed_PTBXL/label.npy')
    p.add_argument('--activate',action='store_true')
    a=p.parse_args()
    import numpy as np
    from src.analysis.pipeline import ECGAnalysisPipeline
    data=np.load(a.data,mmap_mode='r',allow_pickle=False);labels=np.load(a.labels,allow_pickle=False)
    if data.ndim!=3 or data.shape[1:]!=(5000,12):p.error('Expected (N,5000,12) signals')
    if labels.shape!=(len(data),) or set(labels.tolist())!={0,1}:p.error('Expected (N,) binary labels, 0=normal, 1=anomaly')
    # Snapshot hashes before and after scoring prevent unnoticed data replacement.
    dh,lh=digest(a.data),digest(a.labels)
    root=Path('evaluation/threshold_runs')/str(uuid.uuid4());root.mkdir(parents=True)
    pipeline=ECGAnalysisPipeline();profile=None;rows=[];started=time.perf_counter()
    print('Development data reused for checkpoint selection. No independent test claim.',flush=True)
    with (root/'scores.jsonl').open('x',encoding='utf-8') as f:
        for i in range(len(data)):
            result=pipeline.analyze(data[i,100:4900,:],source_id=Path(a.data).name,sample_index=i,crop_start_sample=100)
            current=result.provenance['score_profile']
            if profile is not None and current!=profile:raise ValueError('Scoring profile changed')
            profile=current
            row={'sample_index':i,'label':int(labels[i]),'score':result.model.anomaly_score}
            rows.append(row);f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            pipeline.store.discard(result.analysis_id)
            if i==0 or (i+1)%25==0 or i+1==len(data):
                elapsed=time.perf_counter()-started
                print(f'{i+1}/{len(data)} elapsed={elapsed:.1f}s estimated remaining={elapsed/(i+1)*(len(data)-i-1):.1f}s',flush=True)
    if (dh,lh)!=(digest(a.data),digest(a.labels)):raise ValueError('Source files changed; no activation')
    record={'split':'development_reused','used_for_checkpoint_selection':True,
        'split_id':'existing_training_monitor_set','profile':profile,'rows':rows,
        'dataset_sha256':dh,'labels_sha256':lh}
    scorepath=root/'scores.json';scorepath.write_text(json.dumps(record,indent=2,allow_nan=False))
    config=select_threshold(record);config['selection']['scores_file_sha256']=digest(scorepath)
    (root/'threshold.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False))
    print('Raw threshold:',config['threshold'],'artifacts:',root)
    if a.activate:
        activate(config,'configs/anomaly_threshold.json')
        print('Activated. Re-run local analysis in UI. Prediction uses this disclosed development threshold.')
    else:print('Candidate only. Use --activate to compute and activate explicitly.')

if __name__=='__main__':main()
