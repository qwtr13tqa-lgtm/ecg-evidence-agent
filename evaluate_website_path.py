"""Evaluate the SAME detector as the website. No threshold fitting or history edits."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/Processed_PTBXL/test.npy')
    p.add_argument('--labels',default='data/Processed_PTBXL/label.npy')
    p.add_argument('--checkpoint',default='ckpt_shape_guided_shapex_gate/best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt')
    p.add_argument('--threshold',type=float,default=-0.9182019432385763)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    import numpy as np
    from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
    from src.inference.sgrf_adapter import SGRFDetector
    data=np.load(args.data,mmap_mode='r',allow_pickle=False)
    labels=np.load(args.labels,allow_pickle=False)
    if data.ndim!=3 or data.shape[1:]!=(5000,12):raise ValueError('Expected N x 5000 x 12')
    if labels.shape!=(len(data),) or not set(labels.tolist()) <= {0,1}:raise ValueError('Expected aligned binary labels')
    if not np.isfinite(args.threshold):raise ValueError('Invalid threshold')
    detector=SGRFDetector(args.checkpoint,device=args.device)
    rows=[]
    for i in range(len(data)):
        x=np.asarray(data[i,100:4900],dtype=np.float32)
        score=detector.predict(x).anomaly_score
        rows.append(dict(index=i,label=int(labels[i]),score=float(score),prediction=int(score>=args.threshold),
                         input_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()))
        print(f'{i+1}/{len(data)} score={score:.8f}',flush=True)
    y=[r['prediction'] for r in rows]; scores=[r['score'] for r in rows]
    tn,fp,fn,tp=confusion_matrix(labels,y,labels=[0,1]).ravel().tolist()
    report=dict(scope='website_adapter_fixed_threshold_development_evaluation_not_independent_test',
        preprocessing='adapter.predict: crop[100:4900] then per-lead minmax then STFT',
        checkpoint_sha256=digest(args.checkpoint),data_sha256=digest(args.data),labels_sha256=digest(args.labels),
        adapter_sha256=digest('src/inference/sgrf_adapter.py'),threshold=args.threshold,device=args.device,
        metrics=dict(tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tn+tp)/len(rows),
            auc=float(roc_auc_score(labels,scores)) if len(set(labels.tolist()))==2 else None,
            precision=float(precision_score(labels,y,zero_division=0)),recall=float(recall_score(labels,y,zero_division=0)),
            f1=float(f1_score(labels,y,zero_division=0))),rows=rows)
    folder=Path('evaluation/website_path_reports');folder.mkdir(parents=True,exist_ok=True)
    target=folder/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report['metrics'],indent=2));print('Report:',target)

if __name__=='__main__':main()
