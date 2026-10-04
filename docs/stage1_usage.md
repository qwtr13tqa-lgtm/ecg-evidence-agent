# 第一阶段：分析身份、隔离存储、专业工具

保留 analyze() -> ECGAnalysisResult；增加 result.analysis_id 和 pipeline.store。
不调用网关，不改变候选峰检测算法、RR 筛选范围或证据打分公式。

## 调用

```python
pipeline = ECGAnalysisPipeline()
result = pipeline.analyze(ecg, sample_index=0, crop_start_sample=100)
executor = ECGToolExecutor(pipeline.store, result.analysis_id)
response = executor.execute("inspect_error_window", {
    "start_sample": 4500, "end_sample": 4800, "lead": "V1"
})
rr = executor.execute("inspect_rr_intervals", {"offset": 0, "limit": 50})
record = pipeline.store.get_record(result.analysis_id)
```

导入 ECGAnalysisPipeline 自 src.analysis.pipeline，ECGToolExecutor 自 src.tools.executor。
工具使用输入片段的半开区间采样坐标，返回秒与原始记录坐标。
RR 原始数据和筛选详情在 result.rhythm.rr_details；旧 to_dict() 不扩展字段，保持旧测试兼容。
时间窗口只统计已有误差图，不重跑模型，不产生病理判断。
evidence_id 绑定 analysis_id、工具名与参数；同一不可变记录的相同查询返回同一 ID。
工具执行历史由 executor.history() 获取，包含参数、耗时、状态与证据 ID，无完整数组。

## 存储与边界

进程内最多 32 条记录（包括失败记录），重启即丢失；满时拒绝新分析，不自动淘汰。
可通过 AnalysisStore(capacity=...) 传给 Pipeline；确实不再需要时显式 store.discard(id)。
大数组在本地快照中，读写深拷贝避免调用方污染，代价是内存和复制开销。
记录生命周期为 running -> succeeded/failed，禁止修改终态。
创建 Pipeline 时的权重缺失和存储已满不创建分析记录；analyze 中异常会记录错误类型并原样抛出。
失败 ID 可从 store.list_records() 查询；没有跨进程持久化、用户鉴权或 UI 乱序任务防护。
执行器由可信应用代码构造；本阶段不是安全沙箱，禁止把整个 store 暴露给 LLM。
权重 SHA256 初始化计算一次；输入 hash 基于本次验证后的 float32 C 顺序字节。
来源、形状、采样率、裁剪和关键算法配置独立记录。不是完整实验环境锁定。
未校准的 relative_evidence_score 保留兼容；注释已修正，不将其升级为概率。

## 验证

离线：
```bash
python -m unittest tests.test_analysis_isolation tests.test_ecg_tools tests.test_rhythm_features tests.test_rhythm_boundaries tests.test_analysis_rhythm tests.test_interpretation
```

真实数据本机联调（运行两个样本，无网关调用）：
```bash
python -m tests.test_stage1_live
```

在交付环境完成 35 项离线测试；真实推理未执行，交付环境没有 Torch、权重或数据。
原有 UI 继续兼容，但 UI 尚未展示记录或提供工具选择；这是第二阶段工作。
