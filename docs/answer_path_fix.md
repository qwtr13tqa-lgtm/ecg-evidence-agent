# 回答路径诊断补丁

真实窗口已正确选择 [4500,4800)。第三轮 KeyError 尚不能从旧日志确定缺失字段。
本补丁为模型提供从实际工具 data 枚举的 observation_paths（最多200条），并说明复制路径。
校验区分缺失字段、数组索引错误、穿过标量、非法转义及数值不一致。
失败轨迹增加 failure_phase、error_code，路径类错误增加 observation_index 和受限 requested_path。
不输出原始异常正文或整段无效模型文本；失败仍不产生草稿，不自动重试。
不忽略错误观察，不自动删除 /data 前缀，不放宽数值类型检查。
知识未命中不是本次 KeyError 的已证实原因，本补丁未修改检索算法。

```bash
python install_ecg_answer_path_fix.py
python install_ecg_answer_path_fix.py --apply
python -m unittest tests.test_answer_paths tests.test_recent_error tests.test_stage2_agent tests.test_analysis_isolation tests.test_ecg_tools tests.test_agent_workflow
python -m tests.test_recent_error_live --with-agent --allow-external --sample 0
```

若仍失败，提供 error 和 trace 尾部即可；不要发送 Token。
本地离线测试验证错误分型和路径提示构造，不能保证网关下一次生成必然正确。
