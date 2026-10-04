# SGRF-Net ECG分析与证据复核工作区

基于SGRF-Net、本地信号计算和受约束工具调用的ECG研究原型。自然语言用于查询与复核，数值由本地工具计算；系统不作临床诊断。

本说明按2026-09-16提供的目录、源码及运行结果整理。目录中存在文件不等于功能已完成真实端到端验收。

## 文档入口

- [Agent架构、评测与边界](README_AGENT.md)
- [网站启动说明](Readme_网站)
- [用户操作说明](docs/user_guide.md)：与网站本地帮助共用，由产品导航补丁提供。
- [原始SGRF-Net说明](docs/SGRF_MODEL_ORIGINAL.md)：原文保留模型摘要、作者、论文声明、训练参数、实验表、引用和致谢。本次未重新核验发表信息或复现实验。

## 当前主入口

`app_ecg.py`是当前统一网站入口，使用8505端口。侧栏管理新建分析、历史和批量复核；单条记录主导航为“查看信号、向本记录提问、查看分析依据、导出记录”。

`app_batch_review.py`是保留的独立批量页面，不必与主网站同时启动。`ui_ecg_*`、`ui_sgrfnet_three_entry.py`等为旧版或专用入口，不作为默认启动入口。`install_*.py`和补丁目录用于版本迁移，不要按目录顺序全部重装。

## 快速启动：已有Windows环境

在项目根目录运行：

```powershell
.\.venv-ecg\Scripts\python.exe scripts/check_agent_environment.py
.\.venv-ecg\Scripts\python.exe -m pip check
.\.venv-ecg\Scripts\python.exe -m streamlit run app_ecg.py --server.address 127.0.0.1 --server.port 8505 --browser.gatherUsageStats false
```

浏览器访问 http://127.0.0.1:8505 。本地分析、固定筛选、波形及常见产品帮助不需要API Key；外部模型问答需要配置网关并在界面授权，见[启动说明](Readme_网站)。

这是已有环境的启动流程，不是经过干净机器验证的一键安装流程。`requirements-agent-observed.txt`是观测依赖记录，不是完整依赖锁。

## 本地资产与输入

- 数据：`data/Processed_PTBXL/test.npy`及对应`label.npy`。
- 权重：`ckpt_shape_guided_shapex_gate/best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt`。
- 知识：`data/knowledge/ecg_knowledge.jsonl`。
- 阈值：`configs/anomaly_threshold.json`。

网站示例输入由`sample[100:4900, :]`取得，500Hz、4800点、12导联，共9.6秒。导联顺序为I、II、III、aVR、aVL、aVF、V1–V6。窗口为左闭右开区间；原记录采样点需加裁剪起点100。

历史使用过2160条数据；本次提供的网站路径评测是100条。数据数量与索引对应关系应读取实际文件并校验哈希，不可仅凭同名文件或索引假定是同一记录。

上传入口的CSV/NPZ支持情况以当前部署页面为准；当前仅接受500Hz、完整12导联。原始上传信号滤波与训练流程的一致性仍未验证。持有处理后的npy不代表能复现原始PTB-XL预处理。

## 功能与执行方式

| 功能 | 当前实现 |
|---|---|
| 本地分析 | 模型分数、重构、误差图、导联与候选时间区域、RR统计 |
| 波形复核 | 输入与重构叠加、绝对差、组合分数、窗口高亮 |
| 单条问答 | 有预算的工具循环、证据引用、历史窗口对象恢复 |
| 批量筛选 | 读取报告，按标签与预测筛选、按分数排序 |
| 批次对话 | 显式选择批次，单次工具选择后逐条检查兼容历史并读取摘要 |
| 批量到单条 | 匹配历史则复用；否则由用户点击运行本地分析，核验后保存 |
| 历史管理 | SQLite持久化、搜索、重命名、归档与确认删除 |
| 产品帮助 | 常见使用问题读取本地文档，零模型调用；保守规则分流 |
| 步骤指引 | 侧栏步骤卡，按事件推进，部分步骤人工确认 |

批次对话不是通用多步规划器；单条分析权限没有被全局放开。产品帮助不能保证识别所有问法。所有新UI的实际验收以部署环境结果为准。

## 架构与目录

| 路径 | 职责 |
|---|---|
| `lib/`、`train.py` | 原模型与训练代码 |
| `src/inference/` | 网站模型适配器 |
| `src/analysis/` | 分析流程、记录、判定、历史持久化 |
| `src/evidence/`、`src/features/` | 模型区域与节律统计 |
| `src/tools/` | 本地窗口、RR、时间对齐工具 |
| `src/agent/` | 网关、执行循环、记忆与提交校验 |
| `src/knowledge/` | BM25及知识来源 |
| `src/review/` | 批次查询、历史匹配、受限批次对话 |
| `src/ui/`、`src/reporting/` | 界面、帮助、图形和导出 |
| `evaluation/`、`tests/` | 评测记录与工程测试 |
| `local_history/` | 单用户历史数据库 |

## 当前可引用的结果

报告：`evaluation/website_path_reports/20260915_131219_469735.json`。

| 样本数 | TN | FP | FN | TP | Accuracy | AUC | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 47 | 3 | 16 | 34 | 0.81 | 0.9032 | 0.9189 | 0.68 | 0.7816 |

本报告阈值为−0.9182019432385763，分数≥阈值预测异常；误报按分数降序为10、37、1。该结果是开发用途评估，不是独立临床性能。原模型README中的AUC0.887等实验值及`test2.py`其他指标属于不同实验口径，不能混为一组。

## 运行评测

网站路径评测，不调用LLM，结果另存新报告：

```powershell
.\.venv-ecg\Scripts\python.exe .\evaluate_website_path.py --device cpu
```

该脚本默认阈值来自这次报告，不自动跟随网站配置；配置变化时显式传入`--threshold`。它复用网站适配器，不能据此声称训练预处理已完全一致。

单样本推理对照：

```powershell
.\.venv-ecg\Scripts\python.exe .\check_inference_consistency.py --index 10
```

Agent案例先预览，再自行决定是否允许外部请求：

```powershell
.\.venv-ecg\Scripts\python.exe -m evaluation.run_capability --case CAP_MEMORY_S0 --schemes agent
```

加`--allow-external`才执行真实模型请求。旧批次与失败记录保留，补丁自测不能替代真实Agent评测。不要把全量unittest发现当成离线测试承诺；部分现有测试依赖本地权重或外部服务。

## 已知限制与后续工作

1. 已定位离线TestSet未归一化、网站归一化导致分数不同；新增网站路径评测统一了部署与评测调用，但checkpoint训练预处理及完整记录/裁剪后归一化顺序仍须确认。
2. 模型分数不是概率；阈值数据参与模型选择，不能声称独立测试结果。
3. 峰检测尚未验证；RR不是经过窦性筛选的NN，rr_std不能称SDNN。
4. 结构和数值通过校验不等于自由文本语义或医学结论正确。
5. Agent减少查询次数不等于更快、更省；尚未证明整体优于固定流程。
6. SQLite工作区按单用户、单服务进程设计，不是多租户权限系统。不要直接公开部署。
7. 没有长期健康趋势验证、通用时间序列验证或相似病例检索验收。

下一步优先完善预处理规范、独立评估及任务级对照；扩展能力以实际复核需求为依据。
