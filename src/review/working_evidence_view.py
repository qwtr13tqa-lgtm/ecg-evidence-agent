"""Compact per-request evidence view, preserving canonical JSON pointer prefixes."""
import json
from src.review.final_evidence_view import selected_rows
from src.review.cross_context import recent_context

FIELDS=('index','sample_index','category','label','prediction','threshold','score','score_margin',
        'reconstruction_error','shape_error','region_count','heart_rate_bpm','mean_rr_seconds',
        'rr_cv','measurement_status','history_status','top_lead','region_coordinates')

def build_working_messages(system,question,evidence,old_messages,trace,recent=None,final_only=False):
    views=[];before=after=0
    for eid,e in evidence.items():
        data=e['data'];body={k:v for k,v in data.items() if k not in ('rows','assumptions','batch_sha256')}
        view={'evidence_id':eid,'scope':e.get('scope'),'data':body}
        rows=data.get('rows')
        if isinstance(rows,list) and all(isinstance(r,dict) for r in rows):
            # Fully returned small result sets remain complete; larger sets are explicit excerpts.
            chosen=selected_rows(rows,limit=8)
            # Explicit tool requests expose the requested page (up to 50), not another 8-row sample.
            requests=[t for t in trace if t.get('evidence_id')==eid and (
                t.get('stage')=='evidence_reuse' or (t.get('stage')=='tool' and t.get('tool')=='query_batch_collection' and t.get('ok') and not t.get('cache_hit')))]
            if requests:
                last=requests[-1]
                chosen=(last.get('row_indices') if last.get('stage')=='evidence_reuse' else list(range(len(rows))))[:50]
            before+=len(rows);after+=len(chosen)
            view['row_excerpts']=[{'original_path':f'/rows/{i}',
                'data':{k:v for k,v in rows[i].items() if k in FIELDS}} for i in chosen]
            view['row_selection']={'total_rows':len(rows),'shown_rows':len(chosen),
                'omitted_rows':len(rows)-len(chosen),'policy':'requested page, capped at 50' if requests else 'metric extremes then original order; not proven support or counterexamples'}
        views.append(view)
    # Retain actual actions/errors, not old assistant prose or duplicate tool payloads.
    actions=[{k:t[k] for k in ('stage','tool','arguments','ok','error','evidence_id','hint','cache_hit','executed') if k in t}
             for t in trace if t.get('stage') in ('tool','tool_rejected','collection_query')]
    policy='''\n证据视图：每项data的字段保持原始路径；row_excerpts中的original_path是原证据行路径，字段路径须拼接它，例如original_path=/rows/49、字段score对应/rows/49/score，不能引用/row_excerpts。不要把摘录位置当成原数组下标。引用使用原evidence_id和原data路径。
同条件且字段已覆盖的集合查询会复用原证据，按请求展开原始行；单次最多展示50行，更大结果请分页。明细摘录不代表全量；组统计仍基于其标明的有效数。需要摘录外记录时可用集合查询筛选/分页，不重复查已有数据。代表候选不自动构成支持或反例。证据充分立即提交，无法确定因果时说明边界。正文目标600字，支持记录与反例各最多3条；用户要求更多时明示摘录范围和未完成部分。已执行操作及错误见executed_actions，不重复失败操作。'''
    if final_only:policy+='\n本轮只允许提交，禁止查询。缺少的证据必须说明，不能假装补查过。'
    messages=[{'role':'system','content':system+policy},
              {'role':'user','content':json.dumps({'working_evidence':views,'executed_actions':actions,
                'untrusted_recent_context':recent_context(recent)},ensure_ascii=False,separators=(',',':'))},
              {'role':'user','content':question}]
    size=lambda m:len(json.dumps(m,ensure_ascii=False,separators=(',',':')))
    return messages,{'stage':'working_evidence_projection','context_before':size(old_messages),
        'context_after':size(messages),'rows_before':before,'rows_retained':after,'rows_omitted':before-after,
        'original_evidence_preserved':True,'original_pointers_preserved':True,'finalization_only':final_only}
