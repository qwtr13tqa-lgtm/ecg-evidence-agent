"""Bridge a selected report row to a verified single-record history snapshot."""
import math
import time
from src.review.batch import validate, signal, sha_file

PREPROCESS='adapter.predict: crop[100:4900] then per-lead minmax then STFT'


def compatible(result,report,row):
    p=result.provenance; d=p.get('model_decision',{})
    return (p.get('input_sha256')==row['input_sha256']
        and p.get('checkpoint_sha256')==report['checkpoint_sha256']
        and p.get('review_adapter_sha256')==report['adapter_sha256']
        and p.get('sample_index')==row['index']
        and p.get('crop_start_sample')==100
        and d.get('status')=='configured' and d.get('threshold')==report['threshold']
        and d.get('prediction')==('model_anomaly' if row['prediction'] else 'model_normal')
        and math.isclose(result.model.anomaly_score,row['score'],rel_tol=0,abs_tol=1e-6))


def open_analysis(report,index,data_path,root,history,pipeline_factory):
    validate(report)
    if report.get('preprocessing')!=PREPROCESS:raise ValueError('报告预处理不受此入口支持。')
    matches=[r for r in report['rows'] if r['index']==index]
    if len(matches)!=1:raise ValueError('记录不属于所选批次。')
    row=matches[0]
    raw=signal(report,row,data_path)
    # Old snapshots without adapter provenance are deliberately not assumed compatible.
    offset=0
    while True:
        records=history.list_analyses(limit=100,offset=offset)
        for record in records:
            if record['sample_index']!=index:continue
            result,_,_=history.load_analysis(record['analysis_id'])
            if compatible(result,report,row):return result.analysis_id,True
        if len(records)<100:break
        offset+=100
    if sha_file(root/'src/inference/sgrf_adapter.py')!=report['adapter_sha256']:
        raise ValueError('当前适配器与报告版本不同，拒绝重算；请使用匹配版本或重新评测。')
    pipeline=pipeline_factory()
    if pipeline.checkpoint_sha256!=report['checkpoint_sha256']:
        raise ValueError('当前权重与报告不同，拒绝重算。')
    started=time.perf_counter();result=None
    try:
        result=pipeline.analyze(raw,source_id='batch_verified_test.npy',sample_index=index,crop_start_sample=100)
        result.provenance['review_adapter_sha256']=report['adapter_sha256']
        if not compatible(result,report,row):
            raise ValueError('重算分数、预测或阈值与报告不一致，未保存；请先核对推理配置。')
        result.provenance['batch_review']={'data_sha256':report['data_sha256'],
            'labels_sha256':report['labels_sha256'],'reported_label':row['label'],
            'reported_score':row['score'],'reported_threshold':report['threshold']}
        history.save_analysis(result,raw,time.perf_counter()-started)
        return result.analysis_id,False
    finally:
        if result is not None:
            pipeline.store.discard(result.analysis_id)
