# 用现有模型选择数据计算开发演示阈值

## 核对结论

已读取上传源码 train.py、dataloader.py、test1.py：
- TrainSet只加载train.npy，训练循环对它更新参数。
- TestSet加载test.npy；train.py另外加载label.npy，每个epoch计算AUC，选择最佳checkpoint并早停。
- 因此test.npy实际用于模型选择；名称叫test不能证明它是独立最终测试集。
- test1.py用同一label.npy，在整批scores的minmax归一化空间按Youden J选阈值，然后在同一数据上计算F1等。其返回字典没有保存阈值，checkpoint保存字段也没有threshold。
- 上传的两个压缩包不包含data/preprocess.py。尚未据此确认原始病人/strat_fold划分；不能说代码已经证明病人独立。

之前要求只接受独立验证集的配置过窄。现在额外支持development_reused，允许用于界面演示，明确复用数据事实，不虚称独立测试性能。

## 前提

先安装上一补丁install_ecg_decision_alignment.py，再安装本脚本。

```bash
python install_ecg_demo_threshold.py --apply
python -m unittest tests.test_demo_threshold tests.test_decision_alignment tests.test_interpretation
python -m evaluation.prepare_demo_threshold --activate
```

默认读取data/Processed_PTBXL/test.npy和label.npy，二分类标签0=正常、1=异常。使用当前Pipeline和checkpoint，裁剪100:4900，评分方式与页面一致。不调用LLM、不需要API Key。不直接沿用旧脚本的归一化阈值。

这是整批本地推理，2160条数据可能需要较长时间；脚本每25条打印进度与按已用时间估算的剩余时间。不要把网关180秒超时当成本脚本超时。

scores.jsonl保留逐样本进度；完整成功后生成scores.json、threshold.json到evaluation/threshold_runs/唯一ID目录。失败不会激活部分数据阈值。当前未提供断点续跑，失败记录保留用于诊断。

--activate在全部评分完成、数据哈希未变化后备份并替换configs/anomaly_threshold.json。没有--activate只生成候选文件。配置记录checkpoint/评分代码/预处理身份、数据与标签哈希、分数记录哈希、Youden J规则和数据复用声明。

## 显示结果

计算完成后回到页面，重新点击“本地分析”，即可冻结新配置并展示：
- 异常检测原始分数。
- 数值阈值；score >= threshold预测异常，否则预测正常。
- 开发集演示阈值说明。

旧分析不会被新阈值暗中改写，必须重新分析。若服务仍运行旧源码，在原启动终端Ctrl+C后重启。

```bash
python -m streamlit run app_ecg.py --server.address 127.0.0.1 --server.port 8505 --browser.gatherUsageStats false
```

阈值是全片段模型分类阈值，不适用于局部窗口均值，也不是AUC或患病概率。对同一数据计算的分类指标不用于宣称泛化效果。后续若用于正式模型性能评估，需重新审查训练/验证/测试独立性与模型选择流程。
