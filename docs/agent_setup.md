# 配置与故障处理

## 现有环境

优先继续使用已验证环境，不为生成说明文件重装依赖。运行 `python -m pip check` 查看依赖一致性；若报冲突请保留输出。
`requirements-agent-observed.txt`是主要包版本清单，不包含全部传递依赖，也不固定PyTorch轮子的下载渠道。全新环境应先按硬件与驱动选择PyTorch构建，再安装清单并运行检查、离线测试与本地样本测试。不要把未验证的新环境描述为已复现。

## 网关

逐行在终端执行；输入Token时输入不回显。不要把完整Token贴到聊天、源代码或版本控制。

```bash
read -rsp "请输入网关 Token: " ECG_API_KEY
export ECG_API_KEY
printf '\n'
export ECG_BASE_URL=http://aigw.dlut.edu.cn/v1
export ECG_MODEL=DeepSeek-V4-Flash-0731-W8A8
export LANGSMITH_TRACING=false
export LANGCHAIN_TRACING_V2=false
python -m streamlit run app_ecg.py --server.address 127.0.0.1 --server.port 8504 --browser.gatherUsageStats false
```

默认HTTP是现有网关地址；此前用户已接受该传输方式。应用不会自动加载`.env`，`agent.env.example`只供参考。修改终端变量后需重启进程才能继承新值。

## 状态与错误

- GATEWAY_REQUEST_FAILED：请求失败，需在调用流程或诊断记录查看具体error_type，不能仅凭页面判断超时。
- APITimeoutError：请求等待超时，不代表本地ECG计算错误。不要清除失败记录后只保留成功重跑。
- AuthenticationError：检查当前终端导出的Token是否完整、有无空白或Bearer前缀。
- 503：网关/模型通道不可用或服务错误，具体以返回错误为准。
- completed_draft：结构化草稿完成，不代表医学或正文正确。

诊断只读：`python -m evaluation.inspect_gateway --last 4`。
失败回答也可通过“准备导出”保留，JSON中status与导出校验应标明未完成。页面本地统计可继续使用。

## 验证交付

1. 选样本0本地分析并导出JSON/Markdown。
2. 切换样本1确认旧回答和旧下载清除，来源sample_index和analysis_id一致。
3. 可选提交一次问题，查看中文证据和调用流程，再准备导出。
4. 下载记录与画面对应；失败不能显示为成功回答。

README记录的测试成功来自既有测试和用户运行，新增中文卡片此前仅经过模块测试；用户最近提供的页面是失败请求状态，不能当作完整成功问答卡片的视觉验收。
