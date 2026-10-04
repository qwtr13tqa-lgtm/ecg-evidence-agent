"""Choose a raw-score threshold on declared validation data, never test metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def select_threshold(records):
    rows=records['rows']
    if records.get('split') not in ('validation','development_reused') or not records.get('split_id') or not records.get('profile'):
        raise ValueError('Validation split and scoring profile required')
    if records.get('split')=='development_reused' and records.get('used_for_checkpoint_selection') is not True:
        raise ValueError('Development reuse disclosure required')
    scores=[r['score'] for r in rows];labels=[r['label'] for r in rows]
    if not scores or any(type(s) not in (int,float) or not math.isfinite(s) for s in scores):
        raise ValueError('Finite raw scores required')
    if any(type(y) is not int or y not in (0,1) for y in labels) or set(labels)!={0,1}:
        raise ValueError('Both classes required: normal=0, anomaly=1')
    if len(set(scores))<2:raise ValueError('Constant scores cannot supply a useful separating threshold')
    ids=[r['sample_index'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate sample indices')
    positives=sum(labels);negatives=len(labels)-positives
    # Group tied scores and update cumulative counts: O(n log n), same >= rule as inference.
    grouped={}
    for s,y in zip(scores,labels):
        g=grouped.setdefault(s,[0,0]);g[y]+=1
    candidates=[];tp=fp=0
    upper=math.nextafter(max(scores),math.inf)
    if math.isfinite(upper):candidates.append((0.0,upper,0.0,0.0))
    for threshold in sorted(grouped,reverse=True):
        n,p=grouped[threshold];tp+=p;fp+=n
        tpr=tp/positives;fpr=fp/negatives
        candidates.append((tpr-fpr,threshold,tpr,fpr))
    j,threshold,tpr,fpr=max(candidates,key=lambda r:(r[0],r[1]))
    return {'enabled':True,'threshold':threshold,'score_space':'raw','comparator':'>=',
        'profile':records['profile'],'selection':{'split':records['split'],'used_for_checkpoint_selection':records.get('used_for_checkpoint_selection',False),'split_id':records['split_id'],
        'method':'youden_j_raw','dataset_sha256':records['dataset_sha256'],
        'labels_sha256':records['labels_sha256'],'sample_count':len(rows),
        'normal_count':negatives,'anomaly_count':positives,'validation_tpr':tpr,
        'validation_fpr':fpr,'validation_youden_j':j,'tie_policy':'highest_threshold',
        'independent_test_evaluated':False}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',required=True);p.add_argument('--output',default='evaluation/threshold_candidate.json')
    a=p.parse_args();source=Path(a.scores)
    config=select_threshold(json.loads(source.read_text()))
    config['selection']['scores_file_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='utf-8') as f:json.dump(config,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('Saved threshold candidate:',dest,'raw threshold:',config['threshold'])
    print('Not activated. Independent test performance has not been measured.')

if __name__=='__main__':main()
