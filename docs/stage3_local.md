# 本地RR任务开发评测

新增独立任务DEV_RR_LOCAL，原六类任务与两条DEV_RR失败记录不变。
--cases-file选择任务文件，结果保存相应问题与文件哈希；汇总自动按配置/文件哈希分组。
本批不修改Agent权限：模型仍能看见检索工具。提示词要求不检索，评分no_knowledge_query核对实际执行轨迹。
这不是严格屏蔽检索工具的消融实验，也不能据一次成功推断RAG导致超时。

```bash
python install_ecg_stage3_local.py
python install_ecg_stage3_local.py --apply
python -m unittest tests.test_development_evaluation tests.test_local_evaluation
python -m evaluation.run_development --cases-file evaluation/agent_local_cases.jsonl --case DEV_RR_LOCAL --allow-external
python -m evaluation.summarize_development
```

先核对result.json中的automatic和output.draft，再编辑review.json人工评分。
no_knowledge_query=true只表示未查询知识，不能替代任务完成检查。
旧DEV_RR超时不覆盖；没有草稿的人工task_correct记fail，其余无从评审的字段保留null。
原开发集其他任务可逐个运行：--case DEV_WINDOW / DEV_FILTER / DEV_OVERSIZE / DEV_MISSING / DEV_SCORE。
不要为了追求PASS反复只保留成功运行。用户终端的旧run_ecg_agent输出不会自动导入评测记录。
