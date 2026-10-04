"""Frozen raw-score decision contract. No probability or clinical calibration claim."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCORE_VERSION='sgrf-combined-raw-v1'
FORMULA='mean_over_strategies_time_leads(exp(-clamp(sigma,-5,5))*(reconstruction-target)^2 + clamp(sigma,-5,5) + 0.15*(1-cosine_similarity_over_time))'


def score_profile(result, detector):
    sources={}
    for name in ('src/inference/sgrf_adapter.py','lib/SGRFNet.py','lib/modules.py','lib/modules2.py'):
        p=ROOT/name
        sources[name]=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return {'score_version':SCORE_VERSION,'checkpoint_sha256':result.provenance.get('checkpoint_sha256'),
        'implementation_sha256':sources,'sampling_rate':result.input.sampling_rate,
        'num_samples':result.input.num_samples,'num_leads':result.input.num_leads,
        'crop_start_sample':result.provenance.get('crop_start_sample'),
        'mask_ratio_time':getattr(detector,'mask_ratio_time',None),'mask_ratio_spec':getattr(detector,'mask_ratio_spec',None),
        'normalization':'per_lead_minmax_minus1_plus1','direction':'higher_is_more_anomalous','score_space':'raw'}


def evaluate_decision(score, profile, config):
    base={'version':'model-decision-1.0','score':score,'score_version':SCORE_VERSION,
        'formula':FORMULA,'score_space':'raw','direction':'higher_is_more_anomalous',
        'status':'unconfigured','prediction':None,'threshold':None,'comparator':'>=',
        'probability':None,'clinical_validation':False,'reason':'THRESHOLD_NOT_CONFIGURED'}
    if type(score) not in (int,float) or not math.isfinite(score):
        return dict(base,status='invalid',reason='NONFINITE_SCORE',score=None)
    if not config or config.get('enabled') is False:return base
    base['config_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if config.get('enabled') is not True:return dict(base,status='invalid',reason='INVALID_ENABLED')
    if config.get('score_space')!='raw' or config.get('comparator')!='>=':
        return dict(base,status='invalid',reason='SCORE_SPACE_OR_COMPARATOR_MISMATCH')
    if not profile or config.get('profile')!=profile:
        return dict(base,status='invalid',reason='MODEL_OR_PREPROCESSING_MISMATCH')
    threshold=config.get('threshold')
    if type(threshold) not in (int,float) or not math.isfinite(threshold):
        return dict(base,status='invalid',reason='INVALID_THRESHOLD')
    source=config.get('selection') or {}
    if (source.get('split') not in ('validation','development_reused') or source.get('method')!='youden_j_raw'
            or not source.get('dataset_sha256') or not source.get('split_id')):
        return dict(base,status='invalid',reason='MISSING_VALIDATION_PROVENANCE')
    if source.get('split')=='development_reused' and source.get('used_for_checkpoint_selection') is not True:
        return dict(base,status='invalid',reason='MISSING_DEVELOPMENT_DISCLOSURE')
    return dict(base,status='configured',prediction='model_anomaly' if score>=threshold else 'model_normal',
                threshold=threshold,reason='RAW_SCORE_THRESHOLD_COMPARISON',selection=source)


def freeze_decision(result, detector, path=None):
    profile=score_profile(result,detector)
    try:
        p=Path(path) if path else ROOT/'configs/anomaly_threshold.json'
        config=json.loads(p.read_text()) if p.exists() else None
        decision=evaluate_decision(result.model.anomaly_score,profile,config)
    except (ValueError,TypeError,OSError):
        decision=evaluate_decision(result.model.anomaly_score,profile,None)
        decision.update(status='invalid',reason='THRESHOLD_CONFIG_UNREADABLE')
    result.provenance['score_profile']=profile
    result.provenance['model_decision']=decision
    return decision


def get_model_decision(result):
    frozen=getattr(result,'provenance',{}).get('model_decision')
    if frozen is None:return evaluate_decision(getattr(getattr(result,'model',None),'anomaly_score',None),None,None)
    # Snapshot score must still match its record.
    if frozen.get('score')!=result.model.anomaly_score:raise ValueError('Decision score mismatch')
    return json.loads(json.dumps(frozen,allow_nan=False))


def render_decision(d):
    text=f"异常检测分数：{d['score']}。这是加权重构项与形状项的组合分数，可以为负，不是异常概率。"
    if d['status']=='configured':
        label='异常' if d['prediction']=='model_anomaly' else '正常'
        text+=f"\n\n阈值：{d['threshold']}；规则：分数 ≥ 阈值为模型预测异常，否则为模型预测正常。本次模型预测：{label}。这不是临床诊断。"
        if (d.get('selection') or {}).get('split')=='development_reused':
            text+='\n\n开发集演示阈值：选阈值数据已用于checkpoint选择；当前分类不代表独立测试性能。'
    else:
        text+='\n\n当前没有可用的匹配阈值，不能给出模型正常/异常二分类。原因：'+d['reason']+'。RR统计不能替代该阈值判定。'
    return text
