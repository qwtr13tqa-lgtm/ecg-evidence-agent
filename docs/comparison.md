# 第三阶段：任务覆盖与基线评测

新增文件均不覆盖现有 Agent、网关、页面和评测结果。
18 个开发任务 = 样本索引 0/1/2 × 窗口、RR、筛选、越界、缺失 QTc、分数解释。
这是小规模开发覆盖，不是独立医学评测集。样本1/2的测量值由运行时计算，不复制样本0参考值；它们是否含排除 RR 尚未知。
工具失败/超时由离线测试注入；这18项不代表已覆盖所有失败类型。

四个方案：
- summary：固定读取摘要，1次模型请求。
- summary_rag：摘要 + 原始用户问题 BM25 top3，1次模型请求。
- fixed_tools：固定摘要、RR第一页(offset0/limit50)、V1末尾0.6秒、原问题BM25 top3，1次模型请求。
- agent：已有按需 Agent，最多4次模型请求、6次工具调用（含提交）。
固定方案仅允许 submit_answer，不通过工具强行实现按需决策。不读取 case.expectation 或 rubric。
固定流程窗口只覆盖本开发套件的窗口范围，后续必须另设导联/时长任务检验泛化。
摘要始终被 Agent 引导读取；Agent现有trace不计bootstrap，而固定方案trace计摘要。因此工具执行次数只作为机制描述，不作为公平优劣评分。模型调用次数与墙钟耗时另外报告。
同一case/repetition四方案共享输入和分析ID；模型/温度/每次max_tokens1800/timeout180一致。总预算不相等，报告质量与成本，不称等预算消融。
按seed随机打乱方案顺序减小时间段偏差。未使用重试、答案修复或隐式成功重跑。手工重跑产生新batch，保留旧batch。

## 操作
```
python -m evaluation.run_comparison
python -m evaluation.run_comparison --case DEV_RR_S0
python -m evaluation.run_comparison --case DEV_RR_S0 --allow-external
```
默认仅列任务；指定任务但不加allow-external只预览请求上限。
第一个四方案试跑最多7次模型请求。全部18项为72条记录、最多126次模型请求，不应第一次直接全跑。
扩展：
```
python -m evaluation.run_comparison --case DEV_WINDOW_S0 --case DEV_MISSING_S1 --case DEV_SCORE_S2 --allow-external
python -m evaluation.run_comparison --all --allow-external
```
--schemes 可选择方案，--repeat 1..10 是计划重复，不是失败重试。
终端打印Batch UUID；comparison_runs/<batch>.manifest.json列出全部计划，进程中断也能识别未完成项。
每条run包含result.json、review.json、progress.json。result保存原始回答、证据、参考、自动检查、配置指纹。初始化模型失败时manifest已保留，但不会自动补齐。

人工评审：
```
python -m evaluation.review_comparison --run evaluation/comparison_runs/RUN_UUID
python -m evaluation.review_comparison --run evaluation/comparison_runs/RUN_UUID --reviewer human --task-correct pass --evidence-support pass --text-complete pass --notes "核查计算依据和引用后评分"
python -m evaluation.compare_results --batch BATCH_UUID
```
评分前检查 result.json 中引用证据与正文，不仅看自动检查；三个分数分别是任务正确、证据支持、文字完整。
如果工具缺失导致无法回答实际任务，可以任务fail但证据支持pass（诚实报告缺失）。无回答的超时可记任务fail，其他项保持null。
程序只更新review，不改result。review哈希校验绑定原始记录。汇总pass/fail/unscored，未评分不是通过。缺记录单列missing_runs，不丢掉失败；未完成配对不可作为最终优劣结论。
自动required_tool_selected等偏向查询机制，不能作为所有方案共用的最终准确率。以统一人工评分评价答案，自动指标定位错误。
在同一批比较结果；不同代码/语料/模型指纹不混合。新任务与重复实验预先规划，避免只挑成功结果。

交付范围：基线与覆盖脚本已离线测试；真实模型运行、人工评分和实际收益结论需在用户数据环境执行后产生。尚无Agent优于基线的实验结论。
