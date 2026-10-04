# 第三阶段第一批：任务评测与结果留存

本批只新增文件，复用现有 SGRF Pipeline、Store 和 ECGEvidenceAgent，不覆盖已有调试修改。
任务文件 evaluation/agent_development_cases.jsonl：六类已知开发问题，全部默认样本0。
这些不是独立测试样本，不得把六个问题称为六名患者，也不能宣称统计泛化能力。

## 安装

```bash
python install_ecg_stage3.py
python install_ecg_stage3.py --apply
python -m unittest tests.test_development_evaluation
python -m evaluation.run_development --list
```

离线测试不需要 Torch、CUDA、LangGraph、Token 或网络。真实运行使用现有环境。

## 运行一个任务

```bash
python -m evaluation.run_development --case DEV_RR --allow-external
```

复用当前 ECG_API_KEY、ECG_MODEL、ECG_BASE_URL，发送当前真实分析结构化信息。
每个任务新建分析ID与运行ID。数据与checkpoint仍在本机；网关接收范围与现有Agent一致。
默认4次模型请求、6次工具预算（包含submit_answer），每次180秒，不自动重试。
先单个任务核对；随后可运行整批：

```bash
python -m evaluation.run_development --all --allow-external
```

整批最多24次模型请求，可能耗时较长并产生费用；不会因为某个Agent回答失败自动重跑。
数据加载/网关初始化等runner失败会保存记录后停止；普通Agent失败会保存并进入下一任务。
显式重复：--case DEV_RR --repeat 3 --allow-external。重复运行不覆盖旧记录，不是新增独立样本。
运行前导入或命令参数错误尚未进入run生命周期；硬退出会留下progress.json，不伪造完成状态。

## 每次运行文件

evaluation/runs/<run_id>/progress.json：开始运行前即保存，记录当前阶段。
result.json：问题、评分要点、配置与代码/知识库哈希、输入与checkpoint哈希（在reference.context.provenance）、
分析ID、结构化参考、Agent回答/证据/轨迹/耗时、自动检查和失败类型。
review.json：人工评分。与run_id及result.json文件哈希绑定，避免评审错配。
result.json完成后不由runner覆盖；每次重试新建目录。不是数据库、多用户存储或完整数组归档。
不保存Token、任意环境变量、请求消息历史或隐藏推理内容。保存的是现有Agent返回的草稿与工具证据。
不要手改result.json；会使review的哈希绑定失效。

## 自动检查与人工判断

| 项目 | 自动范围 |
|---|---|
| completed_draft | 流程是否提交草稿 |
| structure_values_references | 独立重新执行现有校验器，不信任输出自称passed |
| analysis_binding | 结果及证据属于本次分析 |
| required_tool_selected | 是否获取任务所需的专业证据，允许不同合理路径 |
| task_observations | 是否把任务要求的数值作为可校验observation提交 |
| task_evidence_cited | 是否引用本次目标证据 |
| no_successful_substitute_window | 越界问题是否被偷偷替换成合法窗口查询 |

工具返回错误不自动等于任务失败。例如20秒请求可以直接拒绝，也可以工具拒绝后说明。
无QTc和异常概率两个问题的语义，需要人工阅读。字符串包含某关键词不等于回答正确。
DEV_SCORE检查是否搜索与引用资料，不证明引用支持语义。
DEV_FILTER当前样本没有排除项，尚不能覆盖“存在排除项”的困难场景，后续扩展开发数据。
自动false不必然等于文字事实错误：如模型只在answer写出数值，却没有提交对应observation。
自动null表示未评估/不适用，不计为通过；评测输出不提供未经定义的总准确率。

## 人工填写

打开一次运行的review.json，只编辑reviewer、notes和三个评分字段；评分用字符串pass或fail，未评留null。

```json
{
  "reviewer": "你的姓名或代号",
  "notes": "计数和引用正确，但有重复残句",
  "task_correct": "pass",
  "evidence_support": "pass",
  "text_complete": "fail"
}
```

上面只展示可编辑字段，不能删除文件原有的run_id和result_sha256。
task_correct：按case.rubric判断是否回答全部要点，不把格式通过当事实正确。
evidence_support：引用是否实际支持相应陈述；没有依据或编造判fail。
text_complete：有无残句、重复乱码、单位缺失等影响阅读的问题。
失败或没有回答的运行通常应标记task_correct=fail；另外两项可留null并在notes说明无草稿。
所有指标均为工程与内容评审，不是医学正确性评估。

## 汇总

```bash
python -m evaluation.summarize_development
```

生成evaluation/runs/summary.json和summary.md，不改review。
每项显示通过数、失败数、未评分/不适用数、已评分分母；无人工评分时不会显示100%。
不同模型、代码、语料、环境/参数按配置分组。开发任务同一组内重复运行会计为多次尝试。
未完成与评审错配记录单独列出，不静默丢弃。

剩余：独立任务集、摘要/RAG/固定工具/Agent基线对照、带专家依据的准确性验证、UI展示与正式报告整合。
