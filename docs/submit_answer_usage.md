# 最终回答通过原生工具提交

已确认：模型在 content 开头返回中文分析文字，违反最终 JSON 协议。
不从这类文本截取大括号内容。新增 submit_answer 原生工具，由其 arguments 提交最终回答。
结构、路径、数值、引用仍用现有校验器；有效提交直接结束，不再向模型发送一次“整理答案”请求。
同轮不得混合查询和提交，避免提交未执行的证据；截断和坏参数继续拒绝。
submit_answer 占用现有工具预算的一个名额；最多6个工具调用包括提交，默认最多4轮模型请求不变。
原有合法 JSON content 路径保留兼容；若模型既不提交也不给合法 JSON，仍失败。
本改动复用已经验证的工具调用通道，但不能保证当前网关对新增 schema 的每次生成都成功。

安装前删除之前手动增加的 MODEL_OUTPUT_PREFIX 临时打印，使该处恢复为：

```python
                phase = "answer_json"
                answer = parse_answer(reply.get("content"))
```

```bash
python install_ecg_submit_fix.py
python install_ecg_submit_fix.py --apply
python -m unittest tests.test_submit_answer tests.test_answer_parser tests.test_answer_paths tests.test_recent_error tests.test_stage2_agent tests.test_analysis_isolation tests.test_ecg_tools tests.test_agent_workflow
```

再次运行原真实RR问题。成功轨迹应包含 stage=submission, tool=submit_answer, ok=true。
仍需人工检查 answer 是否说明9个RR、10个候选峰、保留9排除0、平均约0.986秒及心率约60.85 bpm。
