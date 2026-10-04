# 能力评测与消融使用说明

本批次新增独立评测入口，不修改应用、多轮记忆或已有 comparison_runs 记录。先测当前系统，再用同一协议评价后续记忆改进。测试代码不需要权重、GPU或网关；真实运行需要你已有的模型、数据、环境和 ECG_API_KEY。

## 回答哪些问题

- 专业模型摘要能否提供纯文本模型缺少的样本信息？
- 加入节律统计后，能否回答心率、RR相关问题？
- 详细查询工具能否弥补摘要缺少的数值与坐标？
- 按需工具选择与固定工具流程，在任务完成、请求数量和耗时上有何差异？
- 超过最近三轮的追问能否找到此前讨论对象？

这套评测不能证明 SGRF-Net 的 AUROC、峰检测准确率或临床诊断能力。那些需要标签、参考峰标注、独立数据划分及相应模型评测。当前 test.npy 已参与历史 checkpoint 选择的风险仍然存在。

## 六个方案

|方案|本轮可获得的信息|模型请求预算|
|---|---|---|
|pure_llm|样本长度、导联数、采样率、时长；无波形、模型分数、RR和知识|1次|
|sgrf|输入元信息、模型分数、证据摘要、绑定阈值判定；无RR|1次|
|sgrf_rhythm|上述信息加节律摘要；无RR明细/任意窗口查询|1次|
|fixed_tools|固定收集摘要、阈值判定、前50个RR、V1最后0.6秒误差及该窗口RR对齐|1次|
|agent|现有Agent按问题选择专业工具；知识检索返回空结果|最多4次|
|agent_rag|现有Agent加现有BM25知识库|最多4次|

默认前五个方案。agent_rag 需显式指定；不自动扩展知识库。关闭知识的agent仍保留search_knowledge接口，但返回空结果，可能产生无效查询成本。

pure_llm 是“没有专业样本证据”的对照，并非让文本模型直接读取同一份完整ECG波形。因此结果只能支持专业证据可用性的收益，不能表述为“SGRF-Net诊断准确率超过纯LLM”。

sgrf_rhythm 和 fixed_tools 的差别不仅是工具，还包括明细证据量；fixed_tools与agent预算不同，不能宣称公平等预算胜负。固定窗口始终为V1最后0.6秒，不从答案或评分参考选择工具；它可能覆盖不了其他窗口。若需论证Agent优于更强固定流程，应另外预注册覆盖更广、预算可比的固定方案。

所有方案复用现有 ConversationGateway：最多最近3轮成功且结构通过的回答，9000字符预算；最多20轮会话。此次没有改变这些限制。

## 任务覆盖

48个案例是8类任务 × 6个样本，不是48种独立任务。单轮42个案例，加6个六轮会话，共78轮/方案。

|ID前缀|任务|自动检查|
|---|---|---|
|CAP_DECISION|分数、阈值、模型分类；阈值缺失明确说明|数值及状态字段|
|CAP_WINDOW|V1末0.6秒均值/最大值/起止坐标|坐标、导联、数值|
|CAP_RR|候选峰、原始/保留/排除RR、平均RR和心率|明细或等价事实路径|
|CAP_LONGEST|最长原始RR及两端坐标，含被筛除的间隔|同一RR行的索引、时长、坐标|
|CAP_ALIGN|窗口与RR的时间重叠|窗口及重叠数量|
|CAP_MISSING|询问没有测量的QTc|人工检查是否编造、是否明确缺失|
|CAP_SCORE|分数是不是概率|人工检查解释与证据支持|
|CAP_MEMORY|第一轮V2末1.2秒，四轮其他问题，第六轮追问原窗口|第一/六轮窗口事实；其余人工|

样本0—2标记development，3—5标记evaluation_candidate。它们来自相同test.npy，不能称为独立患者测试集。冻结代码、提示词、评分规则之后才能用未用于调参的问题做最终评测；若已看过这些样本，应重新选择候选数据。

多轮的第六轮需要记忆第一轮对象。现有最近3轮策略可能无法保留它；适当澄清可以在证据支持上通过，但未直接解决窗口任务时，任务完成项仍不通过。之前轮次若超时失败，会影响记忆，应结合逐轮状态分析，不只统计第六轮。

## 安装及先运行一例

在项目根目录执行，命令均为单行：

```bash
python install_ecg_capability_evaluation.py --apply
python -m unittest tests.test_capability
python -m evaluation.run_capability --case CAP_RR_S0
python -m evaluation.run_capability --case CAP_RR_S0 --allow-external
```

第三条只预览，第四条明确允许外发当前样本结构化信息。默认5方案，单轮案例最多8次模型请求；无自动重试。ECG_API_KEY沿用当前终端环境变量，不写入评测记录。

先检查这一例后再增加覆盖：

```bash
python -m evaluation.run_capability --case CAP_WINDOW_S0 --case CAP_DECISION_S0 --case CAP_ALIGN_S0 --allow-external
python -m evaluation.run_capability --case CAP_MEMORY_S0 --schemes agent --allow-external
python -m evaluation.run_capability --case CAP_SCORE_S0 --schemes agent agent_rag --allow-external
```

不建议首次就 --all。默认五方案全套为390轮记录，最多624次模型请求，可能产生很长等待和调用费用。--repeat 2 会完整重复所选案例，而不是只重跑失败项。每个案例/重复只做一次本地推理，所有方案共享其快照；方案执行顺序以固定随机种子打乱。

## 汇总与人工复核

每次真实运行打印 `Batch: 实际UUID`。运行 `python -m evaluation.summarize_capability --batch 实际UUID` 时，将“实际UUID”替换成该值，不要原样输入。

结果存放 evaluation/capability_runs；汇总另存 evaluation/capability_reports 下的新目录：

- report.md：完成率和耗时速览。
- report.json：逐案例/轮次、自动字段检查、失败类别、查询工具次数、模型请求数、输入字节量、人工评分计数。
- manual_review.json：包含回答、证据、参考计算和评分要点的人工评审包。

人工阅读 manual_review.json，将 reviewer 填为评审者，notes 写明依据，把以下项由 null 改为 pass 或 fail：

1. task_correct：是否完成问题要求。要求具体RR明细却只说明信息不足，不能算任务完成；CAP_MISSING本来要求识别缺失，恰当说明缺失可通过。
2. evidence_support：主要陈述是否由本轮证据/知识支持，是否混淆相邻作差与筛选、概率与原始分数。没有引用时不能默认通过。
3. text_complete：文字是否完整、无断句或明显残缺。协议通过不代表文字完整。

然后执行 `python -m evaluation.review_capability --file 评审包实际路径`，再次汇总同一批次。导入会核对结果文件哈希，只写 review.json，不改原始结果。所有条目验证通过后才开始写入。没有人工评审就保留null；不代填通过。无回答的任务无法完成，评审时不要跳过超时病例。

自动评分独立从数组、峰坐标和配置重算参考值，只检查结构化observations。相同指标可来自摘要或calculation_facts，但必须通过本轮证据数值和归属检查。窗统计必须匹配导联及坐标，最长RR必须来自正确一行。它不自动判定自由文本解释、拒答语义或医学正确性。

汇总完成率以计划轮数为分母，超时和缺失都保留；耗时均值包含已记录的失败，未运行项不伪造耗时。本地模型推理在各方案前共享，不计入逐轮问答耗时。记录逻辑输入字节量，不把它当token或费用。按case_results检查不同任务和开发/候选样本；整体表不代表同预算实验或统计显著性。

## 复现与边界

运行记录包含代码/任务集哈希、输入哈希、模型配置、分数profile、批次/轮次身份、证据、回答及调用轨迹。参考计算和评分要点只进入评测记录，不传给模型。RAG可选组记录知识库哈希。

没有改动旧的comparison_runs、评分v2及应用历史记录。网关180秒超时仍可能发生；此次实现不声称修复网关。如果更换模型、预算或提示词，应创建新批次，完整保存旧失败记录。先用小批验证协议，再扩大到更多样本与任务。
