"""Revalidate every turn using existing single-analysis export checks."""
from .analysis_export import build_export, json_bytes, markdown_bytes


def build_conversation_export(result, session):
    if result.analysis_id != session['analysis_id']:
        raise ValueError('Conversation analysis mismatch')
    records = []
    for turn in session['turns']:
        if turn['analysis_id'] != result.analysis_id:
            raise ValueError('Turn analysis mismatch')
        records.append(build_export(result, **{k: turn[k] for k in
            ('question', 'output', 'request_id', 'configuration')}))
    return {'schema_version': 'conversation-export-1.0',
        'analysis_id': result.analysis_id, 'conversation_id': session['conversation_id'],
        'local_analysis': build_export(result), 'turns': records,
        'memory_policy': 'last 3 successful turns, up to 9000 characters; requery evidence each turn',
        'semantic_support_checked': False}


def conversation_markdown(bundle):
    text = markdown_bytes(bundle['local_analysis']).decode('utf-8')
    text += '\n# 多轮问答记录\n\n会话 ID：' + bundle['conversation_id'] + '\n'
    for i, record in enumerate(bundle['turns'], 1):
        out = record['agent_output']
        text += f"\n## 第 {i} 轮\n\n问题：{record['request']['question']}\n\n"
        text += '状态：' + str(out.get('status')) + '\n\n'
        text += ((out.get('draft') or {}).get('answer') or '未生成回答：' + str(out.get('error'))) + '\n'
        if out.get('status')!='completed_draft':
            from src.agent.failure_help import failure_help
            from src.analysis.model_decision import render_decision
            help=failure_help(out)
            text += '\n失败说明：'+help['error_type']+'；'+help['message']+'\n'
            text += '\n本地程序补充（不是LLM回答）：\n'+render_decision(record['model_decision'])+'\n'
        text += '\n结构复核：' + record['export_validation']['status'] + '；原始文字待复核。\n'
    return text.encode('utf-8')
