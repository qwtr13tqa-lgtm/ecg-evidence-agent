"""Offline local inference over a user-designated validation split. No LLM calls."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--labels',required=True)
    p.add_argument('--split-id',required=True);p.add_argument('--confirm-validation-split',action='store_true')
    p.add_argument('--output',default='evaluation/validation_scores.json')
    a=p.parse_args()
    if not a.confirm_validation_split:p.error('Confirm these are validation samples, separate from training and final test')
    dest=Path(a.output)
    if dest.exists():p.error('Output exists; choose a new filename')
    import numpy as np
    from src.analysis.pipeline import ECGAnalysisPipeline
    data=np.load(a.data,mmap_mode='r',allow_pickle=False);labels=np.load(a.labels,allow_pickle=False)
    if data.ndim!=3 or data.shape[1:]!=(5000,12):p.error('Expected validation ECG shape (N,5000,12)')
    if labels.shape!=(len(data),) or set(labels.tolist())!={0,1}:p.error('Expected binary (N,) labels, 0=normal 1=anomaly, both classes')
    pipeline=ECGAnalysisPipeline();rows=[];profile=None
    for i in range(len(data)):
        result=pipeline.analyze(data[i,100:4900,:],source_id=Path(a.data).name,sample_index=i,crop_start_sample=100)
        current=result.provenance['score_profile']
        if profile is not None and current!=profile:raise ValueError('Scoring configuration changed during collection')
        profile=current;rows.append({'sample_index':i,'label':int(labels[i]),'score':result.model.anomaly_score})
        pipeline.store.discard(result.analysis_id)
        if i%25==0:print('Scored',i+1,'/',len(data),flush=True)
    record={'split':'validation','split_id':a.split_id,'profile':profile,'rows':rows,
            'dataset_sha256':digest(a.data),'labels_sha256':digest(a.labels),
            'split_provenance':'user_declared; patient-disjointness and prior model-selection usage not automatically verified'}
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('Saved',dest)

if __name__=='__main__':main()
