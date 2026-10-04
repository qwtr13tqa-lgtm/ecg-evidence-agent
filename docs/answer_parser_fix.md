# 回答解析模块

模型返回的是文本。parse_answer 将文本变成 dict；validate_answer 随后校验字段、引用、数值。
解析成功不是回答正确。代码围栏属于展示包装，本补丁仅移除整条回复的完整 json/无语言代码围栏。
允许纯 JSON 对象及完整独立代码围栏，支持 LF/CRLF。
拒绝围栏外说明、多个块、损坏 JSON、重复键、NaN/Infinity、浮点溢出、非对象。
不自动补逗号、引号，不截取任意大括号，不改变工具参数解析，不增加模型重试。

新增 answer_parser.py；evidence_agent.py 仅将最终回答改用 parse_answer。
错误轨迹增加 parse_reason、content_length、has_code_fence、outer_fence_removed。
JSON 语法错误增加 json_line/json_column/json_position，均相对于去掉外层包装后的解析正文。
不记录模型原文、Token 或原始网关错误正文。
旧日志未保留模型正文，因此不能断言上次失败确实由围栏导致；若再次失败，新诊断能区分类型。

```bash
python install_ecg_answer_parser_fix.py
python install_ecg_answer_parser_fix.py --apply
python -m unittest tests.test_answer_parser tests.test_answer_paths tests.test_recent_error tests.test_stage2_agent tests.test_analysis_isolation tests.test_ecg_tools tests.test_agent_workflow
```

然后重新执行真实 RR 问答命令。既有真实窗口通过结果不用为了本补丁重复调用网关。
