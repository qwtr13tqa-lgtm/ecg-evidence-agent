"""Read-only diagnostics and matched before/after comparisons."""
import json
from collections import Counter
from src.evaluation.threeway_metrics import load_batch,summarize
from src.evaluation.fine_metrics import analyze,aggregate,read_review

NAMES={'summary':'固定摘要','rules':'规则路由','agent':'Agent'}
METRICS={
 'success':('任务成功率','已确认通过轮次 ÷ 当前筛选已落盘轮次。待评仍在分母，另示可能上界；不是置信区间。','越高越好，但先看待评数量和批次是否完整。','低值先区分失败与未复核，再看漏字段、错误引用或请求失败。','问题定位 → 按原因选运行 → 查看自动检查与证据。'),
 'coverage':('目标字段覆盖率','正确的结构化目标字段 ÷ 已定义目标字段，按字段加总。无字段标准的题不计分母，单列数量。','越高越好；只说明结构化目标完成度，不代表全文正确。','缺字段可能是没有查到所需证据，也可能答案未提交对应observations。','逐题复核 → 对照字段检查、工具参数及回答引用。'),
 'structure':('数值与引用校验通过率','structure_values_references=True的轮次 ÷ 全部已落盘轮次。','越高越好；校验通过不等于文字语义正确。','看错误字段路径、原始值和类型、引用ID；不要为提高通过率放宽校验。','问题定位 → 数值/引用校验 → 展开原始trace。'),
 'redundancy':('冗余查询','已确认冗余查询数；比率分母为补充查询数。失败、必要、冗余、待核查互斥。无查询时比率不适用。','越少越好，前提是覆盖率和任务结果不下降。','重复启动摘要可调整工具暴露；无关调用先核查用户是否要求，空检索不自动判冗余。','逐题复核 → 调用复核 → 写判断理由。'),
 'requests':('模型请求数','Meter实际记录的模型请求次数，包括失败请求。按全部轮次算均值。','同等完成质量下越少越好。','查看每轮是否在取得充分证据后继续查询，或触发提交修复。','逐题复核 → 按顺序查看model/tool/submission_repair。'),
 'queries':('补充查询数','trace中的tool事件，排除标记为bootstrap的共同初始摘要及submit_answer。Agent重复读摘要仍计入。','不宜单独追求低值；少查可能漏答。','检查必要查询、冗余查询与失败查询。工具请求记录不保证等于底层实际计算次数。','逐题复核 → 调用分类。'),
 'latency':('响应耗时（中位数/P95）','runner每轮wall_seconds，含失败等待，不含共享深度模型推理。P95采用线性插值。','越低越好；小样本P95不稳定，默认少于20条不展示P95数值。','模型时间高：查看单次请求、轮次及超时；工具时间高：检查具体工具。剩余时间不是网络时间。','逐题复核 → 请求耗时与工具耗时；跨批比较前核对配置。'),
 'timeout':('超时率','异常类型含Timeout的轮次 ÷ 全部已落盘轮次。其他网关错误另列。','越低越好。','按请求查看异常和等待时间，区分后端失败与工具错误，不直接增加重试或超时。','问题定位 → 超时；检查本地gateway诊断。'),
 'claims':('文字陈述证据支持率','supported ÷ (supported+unsupported)，同时列出pending；只有覆盖全文且无待评时才作为全文指标。','部分标注100%不等于完整答案100%。不使用自动LLM裁判。','结构化值正确但文字越界时，查看具体句子及证据来源，不只检查引用存在。','逐题复核 → 文字陈述评分，填写证据路径或判断依据。'),
 'conversation':('整段会话成功率','所有轮次均通过才算会话通过；任一失败则失败；缺轮或待评不能算成功。','用于多轮任务，不能用前几轮通过掩盖最后指代错误。','核查各轮目标对象、历史恢复、跨分析绑定与证据重新查询。','问题定位 → 对象/指代，查看reference_resolution和窗口参数。'),
}


def snapshot(root,batch):
    manifest,rows,warnings=load_batch(root,batch);items=[]
    for row in rows:
        try:review=read_review(row['path'])
        except (ValueError,TypeError,KeyError) as exc:review={};warnings.append(row['run_id']+': '+str(exc))
        items.append(analyze(row,review))
    return manifest,rows,items,warnings


def issues(row,item):
    r=row['record'];out=r['output'];m=r.get('automatic',{}).get('metrics',{});events=out.get('trace',[])
    found=[]
    def add(name,detail):found.append({'run_id':row['run_id'],'case':row['case'],'scheme':row['scheme'],'category':name,'detail':detail})
    if row['timeout']:add('超时','查看逐次请求耗时和异常类型')
    if out.get('status') not in ('completed_draft',):add('运行未完成',out.get('error') or out.get('status','unknown'))
    if m.get('analysis_binding') is False or 'WINDOW_REFERENCE' in str(out.get('error','')):add('对象或指代错误','检查analysis_id、reference_resolution与窗口参数')
    if m.get('structure_values_references') is False:add('数值或引用校验失败','查看失败路径、原值和类型')
    failed=[k for k,v in r.get('automatic',{}).get('field_checks',{}).items() if v is False]
    if failed:add('目标字段缺失或错误','、'.join(failed))
    if row['outcome']=='pending':add('任务待复核','待评不是失败；填写任务/证据/完整性评分')
    if any(v=='fail' for v in row['review'].values()):add('人工评审未通过','查看review.json说明')
    if item['redundant_calls']:add('已确认冗余查询',str(item['redundant_calls'])+' 次')
    if item['pending_calls']:add('调用待核查',str(item['pending_calls'])+' 次；不自动当冗余')
    if item['failed_calls']:add('工具执行失败',str(item['failed_calls'])+' 次；查看工具参数与返回error')
    if any(t.get('stage')=='submission_repair' for t in events):add('发生提交修复','检查第一次失败原因与修复后的答案')
    if item['unsupported_claims']:add('文字陈述无支持',str(item['unsupported_claims'])+' 条')
    return found


def paired(before,after):
    def key(r):return (r['case'],r['scheme'],r['repetition'],r['turn'])
    a={key(r):r for r in before};b={key(r):r for r in after};pairs=[];excluded=[]
    for k in sorted(a.keys()&b.keys()):
        x,y=a[k],b[k];cx=x['record']['config'];cy=y['record']['config'];reasons=[]
        for name in ('model','endpoint_sha256','data_sha256','suite_sha256','temperature','max_tokens','timeout_seconds'):
            if name not in cx or name not in cy or cx[name]!=cy[name]:reasons.append(name+' 不同或缺失')
        for name in ('scoring_profile','corpus_sha256','sdk_retries'):
            if cx.get(name)!=cy.get(name):reasons.append(name+' 不同')
        if x['record']['case'].get('expectation')!=y['record']['case'].get('expectation'):reasons.append('评分标准不同')
        if x['record']['case']['question']!=y['record']['case']['question']:reasons.append('问题不同')
        if not cx.get('input_sha256') or cx.get('input_sha256')!=cy.get('input_sha256'):reasons.append('输入哈希不同或缺失')
        # Compare frozen evaluation reference values; analysis IDs are necessarily different.
        rx={i:v for i,v in x['record'].get('reference',{}).items() if i!='analysis_id'}
        ry={i:v for i,v in y['record'].get('reference',{}).items() if i!='analysis_id'}
        if not rx or rx!=ry:reasons.append('参考值不同或缺失')
        if reasons:excluded.append({'case':k[0],'scheme':k[1],'turn':k[3],'reason':'；'.join(reasons)});continue
        pairs.append({'key':k,'before':x,'after':y})
    return pairs,excluded,len(a.keys()-b.keys()),len(b.keys()-a.keys())


def timing(row):
    r=row['record'];requests=r.get('automatic',{}).get('request_measurements',[])
    model=sum(x.get('elapsed_seconds',0) or 0 for x in requests)
    tool=sum(t.get('elapsed_seconds',0) or 0 for t in r['output'].get('trace',[]) if t.get('stage')=='tool')
    wall=row.get('latency')
    return {'总耗时':wall,'模型请求耗时合计':model,'已记录工具耗时合计':tool,
            '其余本地开销（非网络时间）':max(0,wall-model-tool) if wall is not None else None}
