# 公开配置与环境准备

## 当前验证范围

当前项目在用户 Windows / Python 3.10 / PyTorch 2.6.0+cpu 环境运行过。`environment-windows-observed-20261004.json` 保存 2026-10-04 用户提供的安装版本清单，不包含密钥。

这份清单不是锁文件：缺少安装源、wheel 哈希及完整安装过程。`requirements-app.txt` 是按源码整理的直接依赖清单，未固定版本；`requirements-agent-observed.txt` 保留更早环境观测，不代表当前全量依赖。

**目前没有完成干净机器端到端复现。** 不要在已有可用环境中直接升级全部依赖。以下新环境操作用于后续复现验证，不能宣称一次安装必定成功。

## 已有环境：无需重装

在项目根目录运行：

```powershell
.\.venv-ecg\Scripts\python.exe -m pip check
.\.venv-ecg\Scripts\python.exe -m unittest discover -s tests -p test_public_config.py -v
```

网关现在必须显式配置三个变量：`ECG_API_KEY`、`ECG_BASE_URL`、`ECG_MODEL`。缺失时在创建网关客户端前报错，不再自动使用学校网关或旧模型名称。

```powershell
.\start_ecg_app.ps1
```

脚本保留终端中已有配置，仅询问缺失项。Token 隐藏输入，不写到文件；仍会存在当前进程环境中。服务地址支持 HTTP/HTTPS，实际使用应遵循所选网关的传输要求。格式检查不代表连接或模型权限检查。

只运行本地功能：

```powershell
.\start_ecg_app.ps1 -LocalOnly
```

该模式清除当前终端的 ECG_API_KEY，避免沿用已配置密钥；之后开启问答需要重新输入。它不会删除已保存的分析。

`agent.env.example` 仅用于说明字段，不自动加载。旧报告生成入口也读取同一组环境变量。配置错误信息不会包含 Token 或完整 URL。

## 新环境：安装候选流程

先独立创建环境，保留原 `.venv-ecg`：

```powershell
py -3.10 -m venv .venv-repro
.\.venv-repro\Scripts\python.exe -m pip install --upgrade pip
.\.venv-repro\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-repro\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv-repro\Scripts\python.exe -m pip check
```

这里 CPU 安装命令用于贴近已观测环境，尚未在本次环境实际执行。CUDA 用户应按自己的驱动与 PyTorch 安装渠道选择匹配版本，不要混装 CPU/CUDA 包。网络可用性、未来依赖版本和平台 wheel 会影响安装结果。

原始数据预处理与部分旧实验还需要 `requirements-research.txt` 中的 wfdb、torchvision。torchvision 必须与所选 PyTorch 版本匹配；这部分不属于网站最小复现的承诺范围。

完成独立验证后再导出该环境快照并记录安装来源，不用未经验证的 `pip freeze` 代替复现验证。

## 数据和模型资产

| 文件 | 作用 | 缺失时 |
|---|---|---|
| `data/Processed_PTBXL/test.npy` | ECG 数组，通常为 N×5000×12 | 无法执行真实样本推理 |
| `data/Processed_PTBXL/label.npy` | 与样本一一对应的 0/1 标签 | 不能确定 FN、TP、FP、TN |
| `ckpt_shape_guided_shapex_gate/best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt` | 本项目使用的模型权重 | 无法创建新的模型分析 |
| `configs/anomaly_threshold.json` | 已绑定开发阈值及来源 | 不应虚构二分类结果 |
| `data/knowledge/ecg_knowledge.jsonl` | 工程与 ECG 背景知识 | 检索能力不可用或不完整 |

仓库不附带真实 ECG 数据和权重，没有可保证可用的公开权重下载入口。应由有权提供这些资产的人分发，并保留原始许可、文件哈希和处理配置。不要通过重命名任意 checkpoint 假装兼容。

默认导联顺序 I、II、III、aVR、aVL、aVF、V1–V6；500 Hz；从 `[100,4900)` 取得 4800 点。网站适配器与训练/旧脚本的归一化差异仍需单独核验。上传原始数据不代表已经重现训练时滤波。

批次复核还需要对应评测报告和兼容的历史分析。历史缺失应先在网站执行本地分析；不能把其他批次同索引的记录当作当前证据。

## 分层验证

1. **无需第三方包、数据或网关**：`test_public_config.py` 检查配置校验。
2. **无需真实数据/权重/网关，但需要项目依赖**：`test_cross_record_v2.py`、`test_reliability_closeout.py` 使用合成对象和假网关检查执行约束。
3. **需要数据与权重，不需要 LLM**：网站本地分析、`check_inference_consistency.py`。
4. **需要显式授权外发**：真实问答与 `cross_record_eval_v3 run ... --allow-external`。

```powershell
.\.venv-repro\Scripts\python.exe -m unittest discover -s tests -p test_public_config.py -v
.\.venv-repro\Scripts\python.exe -m unittest discover -s tests -p test_cross_record_v2.py -v
.\.venv-repro\Scripts\python.exe -m unittest discover -s tests -p test_reliability_closeout.py -v
```

`scripts/check_agent_environment.py` 仍用于检查早期观测版本与本地资产；资产缺失会返回非零，不代表纯软件测试必然失败。不要直接运行全量测试并假设它们都不访问外部服务。

## 评测与版本变化

评测脚本的默认网关/模型值已去除，以显式环境变量记录配置。冻结前设置好模型名和地址；冻结与运行时必须一致。

本次更改源码会改变冻结评测所记录的源文件哈希。旧记录保留用于审计，不改旧 manifest 来绕过一致性检查；需要新实验时另建冻结目录。无需为了文档发布重新跑旧批次。

## 验证后仍未完成的工作

干净环境安装、真实网关连通性、CPU 推理数值一致性与 Windows 启动脚本的实际运行，需要在具备相应环境和资产的机器验收。完成这些步骤后，才适合声明“可复现部署”。
