# 时间窗口接口修复

已观察失败：500Hz 输入最后0.6秒，被模型错误换算成 [4200,4800)，实际1.2秒。
新增 inspect_recent_error(duration_seconds, lead=None)，执行器从已绑定分析取采样率和长度，计算末尾窗口。
4800点、500Hz、0.6秒 → 300点 → [4500,4800) → 9.0–9.6秒。
原始记录裁剪起点100，对应原始 [4600,4900)。

秒数必须有限且为正，不接受布尔值/字符串。超过片段时长直接拒绝，不静默裁剪。
离散化采用最近采样点，恰好半个点向上取整。小于半个采样周期且舍入到0时拒绝。
返回 requested_duration_seconds、actual_duration_seconds、window_num_samples 和 rounding_policy。
旧 inspect_error_window 保留，供明确采样点范围查询。原模型、误差公式、RR算法不变。

## 安装与测试

```bash
python install_ecg_window_fix.py
python install_ecg_window_fix.py --apply
python -m unittest tests.test_recent_error tests.test_stage2_agent tests.test_analysis_isolation tests.test_ecg_tools tests.test_agent_workflow
python -m tests.test_recent_error_live
```

最后一个命令运行真实 ECG 本地推理和窗口统计比对，不调用网关。
允许当前真实样本结构化结果发送到现有网关后：

```bash
python -m tests.test_recent_error_live --with-agent --allow-external --sample 0
```

命令先校验真实误差数组切片，再让 Agent 回答固定问题。
不需要反复加载模型分别做本地和网关测试，可以直接使用带 --with-agent 的命令。
测试会检查窗口 [4500,4800)、导联 V1、引用正确窗口以及均值 observation。
它仅针对这个固定任务，并不理解所有自然语言请求；自由文本正确性仍需复核。
若模型选择其他时长/错误窗口，任务回归失败，而不是以协议成功掩盖错误。

`python run_ecg_agent.py` 默认问题也改为自然语言末尾0.6秒查询，新增同样的固定任务检查。
自定义 --question 只输出协议/结构成功，不自动声称该问题已回答正确。
虚构模式增加专用 system 说明及 SYNTHETIC_DATA 限制：预设数组，不是 SGRF-Net 实测。

旧页面的自然语言问题会使用新的工具 schema；停止并重新启动 Streamlit 以加载新代码。
这次没有迁移旧报告格式或实现全文语义校验。
Token 环境变量沿用现有配置。不要重新运行旧 stage2 安装器覆盖补丁。
