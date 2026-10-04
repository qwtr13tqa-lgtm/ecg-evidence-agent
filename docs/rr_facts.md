# RR 事实核算与确定性展示

本补丁核算存储结果的算术一致性，不验证 R 峰检测准确性，也不自动验证 LLM 自由文本。

- src/analysis/rr_facts.py：检查峰顺序和输入范围、峰数、相邻作差/采样率、完整 RR 数、配置范围/掩码/原因、保留值、均值与心率。失败返回 invalid，不展示确定性计算结论。
- src/tools/ecg_tools.py：RR 工具先核算再返回；calculation_facts 是完整序列的事实，不受 offset/limit 影响，作为工具证据供 Agent 引用。原明细字段保留。非法 trace 的查询失败。
- ui_ecg_rr_facts.py：新增页面，按当前 analysis_id 展示程序核算的事实和表格。LLM 原始回答独立展示，仍需复核；不覆盖旧页面和旧评测结果。
- tests/test_rr_facts.py：算术、边界、篡改、分页测试。
- tests/test_ecg_tools.py：原测试将 4800 样本提取的峰挂在 100 样本元数据上；修正 RR 测试的元数据，不放宽校验。
- tests/test_rr_facts_live.py：真实样本本地核算，不创建网关。

float64 重算与现有 float32 摘要比较使用 rtol=1e-6、atol=1e-7，允许数值表示差异，不更改原始测量值。
verified 仅指已实现的 RR 计数、筛选、均值和 HR 一致；不包括 RR 标准差或文字语义。
无足够峰或全部间隔排除时均值和心率保持 None。缺 trace 标记 unavailable。

运行：
```
python -m unittest tests.test_rr_facts tests.test_ecg_tools tests.test_analysis_isolation
python -m tests.test_rr_facts_live --sample 0
python -m streamlit run ui_ecg_rr_facts.py --server.address 127.0.0.1 --server.port 8503 --browser.gatherUsageStats false
```

本地事实展示不调用 LLM。仅在页面勾选外发并点击提问后运行原有 Agent。
本补丁不声称修好了网关偶发慢响应或残句。它将可计算的事实从自由文本中分离展示。
修改了工具返回内容，后续开发评测应作为新代码配置分组；不要覆盖之前失败记录。
