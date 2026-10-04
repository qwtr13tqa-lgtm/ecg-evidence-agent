# 第二阶段：专业证据 Agent

本阶段新增原生工具调用的 LangGraph 循环。复用第一阶段 Store、Executor 和三个 ECG 工具，复用 BM25。
已有 workflow.py、reporting/generator.py 和旧页面保持可运行；新增的是带证据引用的问答草稿接口，尚未迁移旧 ECGReport schema。

## 文件职责

| 文件 | 作用 |
|---|---|
| src/agent/tool_protocol.py | 四个工具的参数 schema、解释边界、最终 JSON 协议 |
| src/agent/gateway.py | 原生 tools 请求和 tool_calls 响应适配，SDK 自动重试关闭 |
| src/agent/evidence_agent.py | 当前分析绑定、决策/执行循环、预算、状态和轨迹 |
| src/agent/answer_validator.py | 复制标量、JSON Pointer、证据 ID 和知识引用检查 |
| src/agent/demo_analysis.py | 明确标识的虚构协议测试数据，不运行模型 |
| run_ecg_agent.py | 虚构协议检查及真实 ECG 命令行入口 |
| ui_ecg_stage2.py | 本地分析与同一样本问答页面 |
| tests/test_stage2_agent.py | 无 Token、无请求的失败与循环测试 |

## 安装和离线测试

在项目根目录运行，使用已经安装 LangGraph/OpenAI 的 Python：

```bash
python install_ecg_stage2.py
python install_ecg_stage2.py --apply
python -m unittest tests.test_stage2_agent tests.test_analysis_isolation tests.test_ecg_tools tests.test_agent_workflow
```

可先用 `python install_ecg_stage2.py --extract stage2_review` 解包到一个新目录阅读所有源码。
安装器只新增文件，遇到不同内容的同名文件即停止，不覆盖你的修改。
开发验证环境：Python 3.12.13、langgraph 1.2.11、openai 3.8.0。代码使用 Python 3.10 兼容语法；用户现有 3.10 环境仍需运行上述测试。不要求升级已经可工作的依赖。

## 验证当前网关的原生工具调用

```bash
export LANGSMITH_TRACING=false
export LANGCHAIN_TRACING_V2=false
export ECG_BASE_URL=http://aigw.dlut.edu.cn/v1
export ECG_MODEL=DeepSeek-V4-Flash-0731-W8A8
read -rsp "请输入 Token: " ECG_API_KEY
export ECG_API_KEY
python run_ecg_agent.py
```

默认仅发送虚构摘要及按需查询的虚构统计。问题明确要求调用误差窗口工具。
成功应有 status=completed_draft、tool_calls >= 1、trace 中的工具执行及最后 PASS。
普通 JSON 网关调用成功不等于原生 tools 协议可用；本脚本用于补上这一检查。模型不调用工具时不会给出协议 PASS。
不支持 tools 的网关可能返回错误；不自动假装调用成功，也不自动切换纯文本解析协议。

## 真实 ECG 问答

确认允许将该样本的结构化信息发送到配置网关后，在本机执行：

```bash
python run_ecg_agent.py --real --allow-external --sample 0 --question "心率依据哪些 RR？V1 最后 0.6 秒的误差是多少？请查询后解释。"
```

真实分析与查询均来自本次 Pipeline 的 Store。不会标为虚构。
发送：问题、分析 ID、摘要、解释限制、按需窗口统计/RR 明细、检索条目。
不发送：原始 ECG、完整误差图、重构数组；摘要去掉 provenance 中本地路径等任意元数据。
知识 source/locator 作为引用会发送，可能包含工程文件相对路径。工具选取后返回的是有限统计，不是全部信号。
allow_external 是调用方声明，并不自动识别数据性质；真实调用由命令行或页面显式开启。

## 页面

```bash
python -m streamlit run ui_ecg_stage2.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

先停止占用同一端口的旧页面，或改用 8502。
选择样本 → 本地分析 → 查看摘要和 II 导联波形 → 开启本次数据外发 → 提问 → 查看回答/轨迹/证据。
切换样本立即清除当前分析选择与旧回答；重新分析前清除旧结果；回答只有 analysis_id 匹配时显示。
页面是本地单用户原型，不是多租户服务。Store 默认 32 条且重启清空；容量满时需重启或由应用显式清理。
相同样本的每次提问独立运行，不自动携带上一轮历史，避免隐含复用旧证据。

## 图与停止规则

START → decide；模型返回工具调用则进入 tools，再回 decide；返回合格 JSON 草稿则 END。
最多 4 次模型请求、6 次工具尝试（不含本地初始摘要读取）。bootstrap 摘要始终有证据 ID。
同名同参数重复调用停止；参数失败作为工具错误返回，模型可在预算内修正。
模型超时、截断、非法响应、最终引用或数值校验失败直接结束，不自动重试或重写。
每次请求默认 180 秒超时；这是单次请求的限制，不是整个工作流 180 秒。四轮最坏可能接近 12 分钟外加本地耗时。
消息字符总量 90000、单工具响应 24000，超限终止或返回工具错误；不是精确 token 预算。

## 回答与校验

最终 draft 包括 answer、evidence_ids、knowledge_ids、observations。
observations 中的 path 相对于对应工具返回的 data，例如 `/leads/0/mean`。
value 必须复制实际标量且类型相同。分析 ID 和必需 limitations 由程序附加，不交给模型改写。
evidence/knowledge 是本次已获取资料集合，可显示引用原文；trace 仅存调用信息，不存隐藏思维链。
模型耗时/工具耗时/调用数用于以后评测。失败输出 draft 为空，不复用上一轮成功回答。

自动验证仅涵盖结构、复制标量和引用成员资格。
它不会证明 answer 中每个数字正确、引用支持全部语义、选对了工具或医学结论正确。
模型可能只引用 bootstrap 而未充分使用细节；需在后续任务评测中单独评价证据充分性。

## 方案选择与剩余工作

选择原生 tool_calls，是为了保留标准请求→工具结果→后续请求协议；暂不采用任意文本动作解析。
选择独立新增 Agent 与页面，是为了保留已有固定报告流程并可对照评测。
本阶段未实现：旧完整报告 schema 迁移、多轮聊天记忆、自由规划调用 SGRF-Net、持久化、专业准确性验证、独立任务集对照评测。
SGRF-Net 仍由本地 Pipeline 确定性运行；Agent 自主选择的是证据补查和知识检索。

参考接口：https://docs.langchain.com/oss/python/langgraph/graph-api
