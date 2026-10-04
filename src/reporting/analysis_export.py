"""Bound local export; no gateway calls and no full signal arrays."""
import hashlib
import json
from datetime import datetime, timezone
from src.analysis.rr_facts import build_rr_facts, render_rr_facts
from src.agent.answer_validator import validate_answer


def request_matches(current, analysis_id, request_id):
    return (isinstance(current, dict) and current.get('analysis_id') == analysis_id
            and current.get('request_id') == request_id)


def build_export(result, *, question=None, output=None, request_id=None, configuration=None):
    aid=result.analysis_id
    if not isinstance(aid,str) or not aid:raise ValueError('Missing analysis ID')
    context=result.to_llm_context()
    if context.get('provenance',{}).get('analysis_id') not in (None,aid):
        raise ValueError('Context analysis mismatch')
    # Explicit configuration allowlist; never serialize environment variables or clients.
    config={k:v for k,v in (configuration or {}).items() if k in
            ('model','timeout_seconds','max_tokens','max_model_calls','max_tool_calls','sdk_retries','endpoint_sha256')}
    check={'status':'not_requested','scope':'structure_values_references_only',
           'semantic_support_checked':False,'medical_correctness_checked':False}
    answer=None
    if output is not None:
        if output.get('analysis_id')!=aid:raise ValueError('Answer analysis mismatch')
        if not isinstance(question,str) or not question.strip() or not request_id:
            raise ValueError('Missing question/request binding')
        for eid,item in output.get('evidence',{}).items():
            if item.get('analysis_id')!=aid or not eid.startswith(aid+':') or item.get('evidence_id')!=eid:
                raise ValueError('Evidence analysis mismatch')
        if output.get('status')=='completed_draft':
            try:
                validate_answer(output.get('draft'),output.get('evidence',{}),output.get('knowledge',{}))
                check['status']='passed'
            except Exception as exc:
                check.update(status='failed',error_type=type(exc).__name__)
        else:check['status']='no_completed_draft'
        # Keep original draft even when validation fails, clearly labeled, for audit.
        answer={k:output.get(k) for k in ('analysis_id','data_kind','status','error','draft','validation',
                'limitations','requires_review','evidence','knowledge','trace','model_calls','tool_calls','elapsed_seconds','gateway_failure','local_support')}
    payload={'schema_version':'analysis-export-1.0','created_at':datetime.now(timezone.utc).isoformat(),
             'analysis_id':aid,'analysis':context,'rr_facts':build_rr_facts(result),
             'request':{'request_id':request_id,'question':question,'configuration':config} if output is not None else None,
             'agent_output':answer,'export_validation':check,
             'scope':'Local structured analysis and optional original Agent draft; no raw ECG/reconstruction/error arrays.'}
    from src.analysis.model_decision import get_model_decision
    from src.tools.window_alignment import inspect_recent_rr_alignment
    payload['model_decision']=get_model_decision(result)
    try:payload['recent_window_rr_alignment']=inspect_recent_rr_alignment(result,0.6,'V1')
    except (ValueError,TypeError,AttributeError):payload['recent_window_rr_alignment']={'status':'unavailable'}
    return json.loads(json.dumps(payload,ensure_ascii=False,allow_nan=False))


def json_bytes(bundle):
    return json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')


def markdown_bytes(bundle):
    lines=['# ECG 分析记录','', '分析 ID：'+bundle['analysis_id'],'',
           '研究原型。计算核算与结构校验不代表医学正确性或文字语义已验证。','',
           '## RR 程序事实','',render_rr_facts(bundle['rr_facts']),'']
    from src.analysis.model_decision import render_decision
    if bundle.get('model_decision'):
        lines+=['## 异常检测判定（本地程序）','',render_decision(bundle['model_decision']),'']
    alignment=bundle.get('recent_window_rr_alignment') or {}
    if 'window' in alignment:
        w=alignment['window'];lines+=['## V1末尾0.6秒与RR时间对齐','',
            f"窗口 [{w['start_sample']}, {w['end_sample']})；重叠RR数 {alignment['overlapping_rr_count']}。", 
            '仅表示时间重叠，完整统计及RR明细见JSON，不表示医学交叉验证。','']
    req=bundle.get('request');out=bundle.get('agent_output')
    if out is not None:
        lines+=['## 本次问题','',req['question'],'','## 模型原始回答（待复核）','',
                '状态：'+str(out.get('status')),'导出时校验：'+bundle['export_validation']['status'],'',
                (out.get('draft') or {}).get('answer') or '未生成回答。错误：'+str(out.get('error')),'',
                '## 引用与调用记录','', '完整证据、知识、原始校验与工具轨迹见配套 JSON。','']
        for kid,doc in (out.get('knowledge') or {}).items():
            lines += [f"- {kid}：{doc.get('title','')}；来源：{doc.get('source','')}；定位：{doc.get('locator','')}"]
    lines+=['','## 分析来源与配置','','```json',json.dumps(bundle['analysis'].get('provenance',{}),ensure_ascii=False,indent=2),'```']
    return ('\n'.join(lines)+'\n').encode('utf-8')
