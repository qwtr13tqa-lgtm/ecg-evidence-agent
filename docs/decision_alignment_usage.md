# 异常分数、阈值、时间对齐及网关失败处理

## 安装与启动

基于已安装的多轮版本，安装器检查修改文件的原始SHA256，不覆盖未知改动。默认预览，--apply备份后安装。

```bash
python install_ecg_decision_alignment.py --apply
python -m unittest tests.test_decision_alignment tests.test_analysis_export tests.test_conversation_export tests.test_ecg_tools tests.test_interpretation tests.test_analysis_isolation
python -m evaluation.check_gateway_connection
python -m streamlit run app_ecg.py --server.address 127.0.0.1 --server.port 8505 --browser.gatherUsageStats false
```

先在旧Streamlit服务的终端按Ctrl+C，之后重启。不要盲目杀掉未知端口进程。需重新本地分析以冻结本次阈值配置；旧内存记录没有新配置。

## 网关失败

GATEWAY_REQUEST_FAILED不是“样本异常”，也不能仅据它判断超时。新页面显示trace中的异常类型，并保留安全HTTP状态（如果SDK提供）。

- APITimeoutError：等待超时。
- AuthenticationError：Token认证失败，在启动服务的终端设置Key后重启。
- APIConnectionError：网络连接问题。
- InternalServerError：网关/上游故障或通道问题。
- unknown：旧导出信息不足，不能推断原因。

本地分析和模型阈值状态始终可查看，不需要API Key。网关失败时本地程序补充不会替换原始回答或把status改成成功。不自动重试，避免重复请求；用户可手动再次发送。

单次虚构连接检查（会发1次网关请求，30秒超时，128token上限）：

```bash
python -m evaluation.check_gateway_connection --send
python -m evaluation.inspect_gateway --last 2
```

连接检查成功只说明这一条请求返回，不保证完整Agent请求成功。本补丁不声称已修复远端网关超时。实际故障需结合本机诊断。没有自动换模型或自动改用其它网关。

## 分数与阈值

异常分数是加权重构项加0.15倍形状项，跨时间/导联/掩码策略平均。它不是MSE、概率或AUC，可为负。模型方向为分数越高越异常。

configs/anomaly_threshold.json 初始 enabled=false，避免猜测阈值。系统显示“无匹配阈值，不能二分类”。这不是还没接好界面，而是尚无验证集选出的阈值资产。

启用配置后：score >= threshold -> model_anomaly，否则 model_normal。仅表示模型分类，不代表临床判断或独立测试性能已确认。阈值与checkpoint、输入形状/采样率/裁剪、掩码参数、模型和预处理代码哈希绑定，变化后拒绝分类。配置在每次分析时冻结，修改文件不会改变旧分析。

旧test1.py对整批scores做minmax后选阈值，没有保存对应原始尺度阈值。不要填AUC=0.8871，也不要直接填归一化阈值。如只有归一化阈值，必须知道选阈值时同一批分数的min/max及完全匹配的评分流程才能换算；推荐用下面的新流程生成原始阈值。

## 生成匹配阈值

需要真正独立的验证集信号与二分类标签，0=正常、1=异常。不能把已用于最终测试的test.npy改名为validation后宣称独立验证。脚本确认标志是数据划分声明，不会自动证明病人独立或未参与模型选择。

数据形状(N,5000,12)，标签(N,)。命令中的validation.npy、validation_labels.npy是示例名称，须替换为实际文件路径，不能直接当作已存在文件。

```bash
python -m evaluation.collect_validation_scores --data validation.npy --labels validation_labels.npy --split-id validation_v1 --confirm-validation-split --output evaluation/validation_scores.json
python -m evaluation.select_anomaly_threshold --scores evaluation/validation_scores.json --output evaluation/threshold_candidate.json
```

前一步仅本地SGRF推理，不调用LLM；耗时取决于样本数。后一步用验证集Youden J选择原始分数阈值；同分取更高阈值，单一类别/常量分数拒绝生成。记录数据、标签、分数文件哈希及验证集TPR/FPR，不能将它们报告成独立测试指标。

检查候选配置后，在项目根目录显式激活：

```bash
cp configs/anomaly_threshold.json configs/anomaly_threshold.before_activation.json
cp evaluation/threshold_candidate.json configs/anomaly_threshold.json
```

重新本地分析；如果配置来源与当前模型一致，页面显示实际阈值和预测。下一步需在未用于训练/选择模型/选阈值的测试集测量性能。本任务没有提供验证集文件，因此未生成或激活数值阈值。

## 时间对齐

模型证据页支持末尾0.6秒或现有模型高分窗口，选择导联后本地显示误差统计和重叠RR。

半开窗口[a,b)，RR由[left_peak,right_peak)表示，仅当left_peak < b且right_peak > a计入。完整RR时长与窗口内重叠时长分别显示。边界处没有下一候选峰，不推断未形成的RR。

样本0现有数据中，V1最后0.6秒为[4500,4800)，与RR索引8的[4277,4769)重叠。完整RR=.984秒，实际重叠[4500,4769)=.538秒。RR来自II导联，不能称V1节律测量。无重叠不表示正常；误差高与RR稳定不矛盾。

全片段异常阈值不能直接用于局部窗口均值。时间对齐不是临床交叉验证，没有自动融合分类器。

Agent新增get_model_decision、inspect_recent_rr_alignment、inspect_window_rr_alignment。现有Agent控制器不被覆盖；调用预算保持原样。

## 导出与验收

重新分析样本0，查看“模型如何判定”和“模型窗口与RR时间对齐”。可问“这个样本有异常吗？”或“V1最后0.6秒与哪些RR重叠？”

导出JSON/Markdown新增本地判定；JSON包含默认V1末尾0.6秒对齐结果，Agent调用的其他窗口保留在该轮evidence。页面临时选择的其它窗口不会自动替换默认导出窗口。

失败轮次在Markdown保留failed，并显示具体异常类型及本地判定，不再只有一个错误代码。原始自由文本仍需复核。
