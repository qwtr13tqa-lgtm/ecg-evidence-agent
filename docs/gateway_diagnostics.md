# 网关逐请求诊断

补丁替换src/agent/gateway.py并新增gateway_diagnostics.py，保持complete(messages,tools)接口与请求参数不变。
不改变180秒默认超时、1800输出上限、temperature=0、tool_choice=auto、max_retries=0。
每次请求开始先保存元数据，结束后原子更新同一文件。默认目录evaluation/gateway_diagnostics。
ECG_GATEWAY_DIAG_DIR可指定其他目录；自定义目录时inspect_gateway用--directory指定相同路径。

字段：本地request UUID、client session UUID、可提取的analysis UUID、UTC时间、总耗时、消息规模、工具schema规模、
模型、输出上限、结束原因、已返回的token用量、provider request ID、错误类型、可用HTTP状态。
不记录Token、base URL明文（仅哈希）、消息正文、模型答案、工具参数正文或异常正文。
成功response_received表示收到了可适配响应，不表示最终回答通过校验。length仍是长度截断，不自动当成功答案。
request_failed是SDK请求异常；response_adapter_failed是收到响应但无法适配；started长期未更新可能进程中断。
请求ID和usage可能缺失，保留null，不推测。
logical_payload_utf8_bytes是规范JSON编码大小，非SDK实际线路字节；字符/字节均不是token数。
非流式总耗时无法区分排队、生成、网络；time_to_first_token_seconds固定null。

```bash
python install_ecg_gateway_diagnostics.py
python install_ecg_gateway_diagnostics.py --apply
python -m unittest tests.test_gateway_diagnostics
python -m evaluation.run_development --cases-file evaluation/agent_local_cases.jsonl --case DEV_RR_LOCAL --allow-external
python -m evaluation.inspect_gateway --last 4
```

可用 --analysis-id UUID筛选某次分析；analysis UUID不是run UUID，前者见result.json中的output.analysis_id。
离线测试不会调用网关。只需一次带诊断的新运行，无需批量或自动重试。
旧记录没有请求级元数据，无法追溯补齐；保留旧记录，不把新旧配置组强行合并。
仅替换gateway.py，不触碰你手改过的evidence_agent.py诊断打印。
诊断输出GATEWAY_REQUEST / GATEWAY_RESULT便于对应文件；磁盘日志失败打印类名，不影响正常请求，也不新增重试。
