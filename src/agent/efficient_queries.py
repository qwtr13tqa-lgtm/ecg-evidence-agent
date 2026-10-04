"""Reduce unnecessary choices without changing evidence, validation or budgets."""
from copy import deepcopy

VERSION='efficient-queries-1.0'
POLICY='''
查询策略 efficient-queries-1.0：
1. initial.bootstrap 是本轮同一分析已经取得的完整摘要，已含 evidence_id 与 observation_paths，可以直接引用。没有 get_analysis_summary 入口，不必再次读取。
2. 先检查用户实际要求的字段哪些在本轮证据中已有。若全部已有，直接 submit_answer；若缺失，只调用能够补足这些字段的工具。不要为了让答案更长而添加未请求的模型判定或无关数值。
3. get_model_decision 用于用户要求模型分类、分数阈值比较或任务确实依赖判定的情况；单独的窗口统计、RR计数或工具测量缺失问题通常不需要它。用户同时明确要求判定时仍应查询。
4. 每次工具返回后重新检查信息是否足够；足够即提交，不必用摘要或知识检索再确认同一事实。多个互不依赖且确有必要的查询可以同一轮发出，存在结果依赖时继续分步执行。
5. 用户询问现有工具未提供的测量时，区分“未测量”与“正常/没有异常”。现有证据和工具能力已足以确认不能提供该值时，直接说明缺失，不用不相关测量替代，也不用检索来编出该样本的数值。
6. 历史恢复若已提供 current_evidence，这是本轮重新查询的结果，应使用它。历史文本本身仍不是数值证据；不能跳过对象与引用校验。
7. 减少调用不是优先于正确性的要求。仍缺必要证据时继续补查，参数含糊时澄清；保留数值原值、类型、证据路径及必要局限，不因少调用而省略问题要求。
'''


def prepare(messages,tools,retriever):
    messages=deepcopy(messages);tools=deepcopy(tools)
    if not messages or messages[0].get('role')!='system':raise ValueError('System message required')
    disabled={'get_analysis_summary'}
    # Unknown retrievers remain enabled. Do not infer availability from one empty result.
    if getattr(retriever,'knowledge_available',True) is False:disabled.add('search_knowledge')
    allowed=[t for t in tools if t.get('function',{}).get('name') not in disabled]
    messages[0]['content']+=POLICY
    if 'search_knowledge' in disabled:
        messages[0]['content']+='\n本次运行未启用知识检索，故不提供search_knowledge；这不表示样本已测量任何缺失指标。'
    return messages,allowed


def complete(gateway,messages,tools,retriever):
    messages,tools=prepare(messages,tools,retriever)
    return gateway.complete(messages,tools)


def policy_trace(retriever):
    return {'stage':'query_policy','version':VERSION,
            'bootstrap_summary_available':True,
            'knowledge_tool_enabled':getattr(retriever,'knowledge_available',True) is not False,
            'mechanism':'tool_visibility_and_prompt; no_cache; no_auto_answer; original_budgets'}
