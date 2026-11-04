# FedDiverse 复现状态

最后更新：2026-11-04（依据近期聊天记录与本地文件核对）

## 1. 当前状态

当前优先事项是**依据作者回复，在新分支重训 Spawrious GAI**。配置与训练入口已准备，
离线检查已通过，尚未执行服务器 GPU smoke 或新的正式训练。当前分支为
`codex/retrain-spawrious-gai-author`，HEAD 为 `762cae1`；本次训练改动尚未提交、尚未推送。

近期另外完成了 GAI 源码时序对照、GAI pivot 单因素验证、CMNIST 方向感知 oracle
九组实验与选择频率分析。它们采用不同协议，结果单独记录在第 6.12–6.16 节，
不能与下表直接混合求均值。当前没有证据证明方向感知方法在估计属性条件下有效，
动态训练效用模块也仅处于讨论阶段。

以下是 **2026-09 已冻结的四组论文复现实验**，比较对象是 Random，不能据此判断
FedDiverse 相对所有其他算法的排名：

1. CMNIST GSC：主要数值、方法排序和相对提升均已复现；
2. Spawrious GSC：FedDiverse 绝对性能接近论文，相对 Random 的优势未复现；
3. Spawrious GCI：FedDiverse 相对 Random 的平均优势已基本复现。
4. Spawrious GAI：Random 基线接近论文，但 FedDiverse 的绝对性能和相对优势
   均未复现。

以下四组正式实验均使用数据 seed 42、43、44，报告最终模型的 worst-group
accuracy。表中“本次结果”使用总体标准差，不选择中间最佳 checkpoint。

| 数据集 | 本次 Random | 本次 FedDiverse | 本次提升 | 论文 Random | 论文 FedDiverse | 论文提升 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| CMNIST GSC | 91.4942 ± 0.4841 | 94.1505 ± 0.3655 | +2.6563 | 92.00 ± 1.61 | 94.01 ± 0.98 | +2.01 | 基本复现成功 |
| Spawrious GSC | 87.7497 ± 0.0744 | 87.7497 ± 1.3650 | 0.0000 | 86.27 ± 1.12 | 88.01 ± 0.96 | +1.74 | 绝对值接近，相对优势未复现 |
| Spawrious GCI | 89.6425 ± 1.5364 | 91.0620 ± 1.4475 | +1.4196 | 87.59 ± 2.00 | 89.91 ± 1.91 | +2.32 | 主要结论基本复现成功 |
| Spawrious GAI | 86.2250 ± 1.9798 | 84.7003 ± 0.8445 | -1.5247 | 85.86 ± 2.56 | 87.28 ± 1.61 | +1.42 | Random 接近；FedDiverse 及相对优势未复现 |

Spawrious GSC/GCI/GAI 的 checkpoint、日志、配置和元数据均已完成本地备份；
GCI 已冻结，GAI 通过标签 `spawrious-gai-official-rng-v1` 冻结。这批结果继续保留，
不选择性补跑或替换。作者回复后的 GAI 重训是新授权的独立批次，不解除旧结果冻结。

## 2. 仓库、分支与冻结点

| 项目 | 分支 | 提交/标签 | 状态 |
|---|---|---|---|
| CMNIST GSC | `paper-faithful-official-rng` | `e1aeb84`；标签 `cmnist-gsc-official-rng-v1` | 已冻结 |
| Spawrious GSC | `reproduce-spawrious-gsc` | `a2959a6`；标签 `spawrious-gsc-official-rng-v1` | 已冻结并推送 |
| Spawrious GCI | `reproduce-spawrious-gci` | `61a6581`；标签 `spawrious-gci-official-rng-v1` | 已冻结并推送 |
| Spawrious GAI | `reproduce-spawrious-gai` | 标签 `spawrious-gai-official-rng-v1` | 已完成本地备份并冻结 |
| GAI pivot 单因素验证 | `baseline-validation-v1` | `2613047` | 代码已推送；legacy/pivot 六次验证完成 |
| CMNIST 方向感知 oracle | `research/cmnist-direction-aware-oracle` | `9762e55` → `726bc86` → `f47c814` → `762cae1` | 九组实验与选择频率分析已完成，结论有限 |
| 作者回复后的 GAI 重训 | `codex/retrain-spawrious-gai-author` | 从 `762cae1` 创建；训练改动未提交 | 当前工作分支，仅完成本地准备 |

关键实现提交：

- `b71d998`：加入论文式 FedAvg 预训练、DHT-only 收集与模型更新调度；
- `d679cca`：移除额外全局/客户端播种，恢复发布代码的训练 RNG 行为；
- `62bc37c`：加入 Spawrious GSC 正式配置；
- `965d5de`：加入 Spawrious GCI 正式配置；
- `2c250b6`：修复独立 Hydra 检查缺少动态 ConfigStore 注册的问题；
- `61a6581`：记录 GCI 正式结果并作为 GCI 冻结点。
- `9ae89e1`：加入 Spawrious GAI 正式配置和复现计划。
- `2613047`：受控 baseline、pivot/ID 映射/选择器边界的单因素验证入口；
- `9762e55`：CMNIST oracle 联合分布方向感知选择器；
- `726bc86`：CMNIST 九组实验产物审计与汇总；
- `f47c814`：选择覆盖分析；`762cae1`：删除额外实验 Markdown 文档。

仓库位置：

- 本地：`D:/federated/FedDiverse/SpuriousFL`；
- 最近实验使用的服务器项目目录：`/root/SpuriousFL`；
- 远端：`https://github.com/mreason7/SpuriousFL.git`；
- 最近（10–11 月）实验服务器：`root@cpod-1wo0rf930ytu.podtcp.compshare.cn:22211`，
  来自近期用户终端记录与本机 SSH 配置；本次文档更新未验证其当前在线状态；
- 9 月旧服务器：`root@cpod-1uk7kkln33pu.podtcp.compshare.cn:27201`，
  用户在“数据集分析”聊天中明确表示已删除。不得再将此旧地址默认当作当前训练机；
- 论文原文：`D:/zotero_file/storage/VZD6UMI6/Németh 等 - 2025 - FedDiverse Tackling Data Heterogeneity in Federated Learning with Diversity-Driven Client Selection.pdf`。

本地项目已整体迁移到 D 盘，包含 `.git`、模型备份与实验记录，Codex 项目路径已指向
新目录。旧 `SpuriousFL` 目录已移除；`.validation_deps`、`.baseline-validation-checks`、
`paper.txt` 和 `FedDiverse_backup` 目前位于 `D:/federated/FedDiverse/`。
数据压缩包现位于 `D:/federated/FedDiverse/FedDiverse_backup/datasets/`。

## 3. 9 月冻结实验的环境与协议

### 环境

- Conda 环境：`spurious`；
- GPU：NVIDIA GeForce RTX 4090；
- Python 3.10.14；
- PyTorch 2.3.0，torchvision 0.18.0；
- PyTorch CUDA runtime 12.1；
- Flower 1.8.0，Ray 2.6.3；
- NumPy 1.26.4，pandas 2.2.2，Hydra 1.3.2。

服务器 GPU 驱动提供向后兼容能力；训练实际使用 PyTorch 随包安装的 CUDA
12.1 runtime。

### 统一模型与优化配置

- 模型：ImageNet 预训练 MobileNet v2，BatchNorm 替换为 GroupNorm；
- 客户端学习率：0.001；
- batch size：28；
- 本地训练：1 epoch；
- 客户端等权聚合；
- 服务端：FedAvgM，学习率 0.1，`beta_1: 0.95`；
- 总客户端数：CMNIST GSC、Spawrious GSC、Spawrious GCI 均为 24，
  Spawrious GAI 为 25；
- 每轮选择客户端数：9；
- CMNIST GSC、Spawrious GSC/GCI 每客户端训练样本数为 200，联邦训练总样本数
  4800；Spawrious GAI 客户端大小不等，联邦训练总样本数 4600。

### Random 与 FedDiverse 调度

Random：200 个 Flower round，对应 200 次模型更新，不估计 DHT。

FedDiverse：201 个 Flower 事件，对应 200 次模型更新：

1. round 1：所有客户端进行一次标准 FedAvg 预训练更新；
2. round 2：所有客户端只估计并上传 DHT，不更新模型或 FedAvgM 动量；
3. round 3-201：每轮由 FedDiverse 选择 9 个客户端，共 199 次更新。

正式 FedDiverse 配置：

- `client_info: triplets_Npredicted`；
- `selection_method: triplets_stochasticmatrix`；
- `biased_trainer_steps: 50`；
- `left_right_trainer_steps: 10`；
- `generalized_cross_entropy_q: 0.3`；
- `biased_optimizer: ReSample`。

### seed 的准确含义

本节冻结结果中的 `seed=42/43/44` 主要控制客户端数据划分，并非完整训练 seed。
模型初始化、DataLoader、DHT 训练、客户端选择和 Ray worker 的随机状态没有被
统一固定，这是当时本地复现采用的运行方式；不能再等同于作者当年的完整随机性协议。
作者后来说明使用 1、2、42 且共用一个全局 seed，但具体 worker 播种机制尚未确认。
因此本节逐 seed 差值不能视为严格共享训练随机性的配对实验；比较口径是三次运行均值。
10 月研究协议和 11 月作者回复后的协议另见第 6 节。

## 4. 已完成实验与产物索引

### 4.1 CMNIST GSC

配置：

- Random：`conf/cmnist_random.yaml`；
- FedDiverse：`conf/cmnist_feddiverse_paper.yaml`。

| seed | Random 测试准确率 | Random worst-group | FedDiverse 测试准确率 | FedDiverse worst-group | 提升 |
|---:|---:|---:|---:|---:|---:|
| 42 | 95.5400 | 91.8164 | 95.5600 | 93.6926 | +1.8762 |
| 43 | 95.6600 | 91.8563 | 95.7800 | 94.5872 | +2.7309 |
| 44 | 95.1500 | 90.8100 | 95.9200 | 94.1717 | +3.3617 |
| 均值 | 95.4500 | 91.4942 | 95.7533 | 94.1505 | +2.6563 |

正式 checkpoint：

| 方法 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| Random | `20260909-104452/final` | `20260909-112853/final` | `20260909-121342/final` |
| FedDiverse | `20260908-113933/final` | `20260909-025420/final` | `20260909-034311/final` |

产物：

- Random 日志：`logs/random_official_rng/`；
- FedDiverse 日志：`logs/predicted_official_rng/`；
- 最终对照：`results/cmnist_gsc_official_rng_final_comparison.csv`；
- 本地备份：`checkpoints/local_backup/cmnist_gsc_official_rng_v1/`。

结论：FedDiverse 三个数据 seed 均优于 Random；本次 FedDiverse 与论文均值
仅差 `+0.1405` 个百分点，核心数值、排序与结论均已基本复现。

### 4.2 Spawrious GSC

配置：

- Random：`conf/spawrious_gsc_random.yaml`；
- FedDiverse：`conf/spawrious_gsc_feddiverse_paper.yaml`；
- 数据划分：`split_mode: spawrious2`。

| seed | Random 测试准确率 | Random worst-group | FedDiverse 测试准确率 | FedDiverse worst-group | 提升 |
|---:|---:|---:|---:|---:|---:|
| 42 | 92.8628 | 87.8549 | 92.8628 | 89.2744 | +1.4196 |
| 43 | 92.7839 | 87.6972 | 92.5868 | 88.0126 | +0.3155 |
| 44 | 92.9811 | 87.6972 | 92.3896 | 85.9621 | -1.7350 |
| 均值 | 92.8759 | 87.7497 | 92.6130 | 87.7497 | 0.0000 |

正式 checkpoint：

| 方法 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| Random | `20260911-025949/final` | `20260911-035247/final` | `20260911-044304/final` |
| FedDiverse | `20260911-084525/final` | `20260911-093656/final` | `20260911-102916/final` |

产物：

- Random：`logs/spawrious_gsc/random_official_rng/`；
- FedDiverse：`logs/spawrious_gsc/predicted_official_rng/`；
- 选择频率诊断：
  `logs/spawrious_gsc/predicted_official_rng/client_selection_frequency.txt`；
- 本地完整备份：
  `checkpoints/local_backup/spawrious_gsc_official_rng_v1/`，包含 6 个
  checkpoint、两组日志及总结、正式配置源码、Git 元数据和环境信息。

结论：FedDiverse 均值比论文低 `0.2603`，绝对性能接近论文；Random 比论文
高 `1.4797`，导致论文报告的 `+1.74` 相对优势在本次三次均值中变为 0。
流程、配置和 checkpoint 均有效，但相对优势未复现。

### 4.3 Spawrious GCI

配置：

- Random：`conf/spawrious_gci_random.yaml`；
- FedDiverse：`conf/spawrious_gci_feddiverse_paper.yaml`；
- 数据划分：`split_mode: spawrious_GCI`。

| seed | Random 测试准确率 | Random worst-group | FedDiverse 测试准确率 | FedDiverse worst-group | 提升 |
|---:|---:|---:|---:|---:|---:|
| 42 | 93.7697 | 91.7981 | 93.2571 | 89.1167 | -2.6814 |
| 43 | 93.5331 | 88.8013 | 94.0457 | 92.5868 | +3.7855 |
| 44 | 93.2177 | 88.3281 | 93.8091 | 91.4826 | +3.1546 |
| 均值 | 93.5068 | 89.6425 | 93.7040 | 91.0620 | +1.4196 |

总体标准差：Random `1.5364`，FedDiverse `1.4475`。本次 Random 比论文高
`2.0525`，FedDiverse 比论文高 `1.1520`；平均提升比论文少 `0.9004`。

正式 checkpoint：

| 方法 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| Random | `20260914-032515/final` | `20260914-041522/final` | `20260914-050500/final` |
| FedDiverse | `20260914-064256/final` | `20260914-073615/final` | `20260914-082934/final` |

产物：

- Random：`logs/spawrious_gci/random_official_rng/`；
- FedDiverse：`logs/spawrious_gci/predicted_official_rng/`；
- Random 总结：`spawrious_gci_random_official_rng_summary.txt`；
- FedDiverse 总结：
  `spawrious_gci_feddiverse_predicted_official_rng_summary.txt`；
- 本地完整备份：
  `checkpoints/local_backup/spawrious_gci_official_rng_v1/`，包含 6 个
  checkpoint、两组日志及总结、正式配置源码、Git 元数据和环境信息。

结论：FedDiverse 恢复了论文中的平均方法排序和正向优势，主要结论基本复现。
seed 42 的负提升作为正式结果保留，不选择性补跑。

### 4.4 Spawrious GAI

配置：

- Random：`conf/spawrious_gai_random.yaml`；
- FedDiverse：`conf/spawrious_gai_feddiverse_paper.yaml`；
- 数据划分：`split_mode: spawrious_GAI_2`。

| seed | Random 测试准确率 | Random worst-group | FedDiverse 测试准确率 | FedDiverse worst-group | 提升 |
|---:|---:|---:|---:|---:|---:|
| 42 | 93.3360 | 87.8549 | 92.2319 | 84.5426 | -3.3123 |
| 43 | 92.5079 | 83.4385 | 92.9416 | 85.8044 | +2.3659 |
| 44 | 93.2571 | 87.3817 | 92.6656 | 83.7539 | -3.6278 |
| 均值 | 93.0336 | 86.2250 | 92.6130 | 84.7003 | -1.5247 |

总体标准差：Random `1.9798`，FedDiverse `0.8445`。本次 Random 比论文高
`0.3650`，FedDiverse 比论文低 `2.5797`；本次平均提升比论文低 `2.9447`。

正式 checkpoint：

| 方法 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| Random | `20260915-081441/final` | `20260915-090238/final` | `20260915-095113/final` |
| FedDiverse | `20260916-020015/final` | `20260916-025153/final` | `20260916-034207/final` |

产物：

- Random：`logs/spawrious_gai/random_official_rng/`；
- FedDiverse：`logs/spawrious_gai/predicted_official_rng/`；
- 最终对照：`results/spawrious_gai_official_rng_final_comparison.csv`；
- 本地完整备份：
  `checkpoints/local_backup/spawrious_gai_official_rng_v1/`，包含 6 个
  checkpoint、两组日志及总结、正式配置、最终对照、模型哈希和元数据。

结论：六次正式运行的配置、数据、调度和服务器产物检查均通过。Random 基线
接近论文，但 FedDiverse 的绝对性能与相对优势均未复现；三 seed 平均相对
Random 下降 `1.5247` 个百分点。seed 43 的正提升与 seed 42/44 的负提升均按
正式结果保留，不选择性补跑或选择中间 checkpoint。

## 5. 数据与完整性审计

### CMNIST

- 原始训练集 60,000，validation 10,000，test 10,000；
- validation 是当前发布代码中 test 的深拷贝；
- 24 个客户端，每客户端 200 个样本，联邦训练总计 4800；
- 测试群体 `[y0g0,y0g1,y1g0,y1g1]` 为
  `[2571,2568,2505,2356]`，接近均衡。

### Spawrious

- 压缩包：`spawrious224__without_domain_adaptation.tar.gz`；
- 完整压缩包大小：14,393,689,896 bytes；
- 解压 gzip stream：14,496,634,880 bytes；
- 数据目录：`/root/SpuriousFL/datasets/spawrious224/`；
- 使用 beach、snow 两个背景与 labrador、dachshund 两个犬种；
- train 22,808，validation 2,536，test 2,536；
- GSC/GCI 均为 24 个客户端，每客户端 200 个样本，共 4800。
- GAI 使用 `spawrious_GAI_2`，共 25 个客户端，客户端大小不等，联邦训练总计
  4600；群体 `[y0g0,y0g1,y1g0,y1g1]` 为 `[2000,500,2000,100]`，客户端
  分片之间无重叠。

9 月这批采用论文式调度的正式 FedDiverse 运行必须满足：

- `EXIT_CODE=0`；
- 日志出现 `DHT collection round 2 completed without a model update`；
- 最终 checkpoint 为 `checkpoints/<exp_id>/final/torchmodel.pt`；
- `config.yaml`、`client_info.json`、`client_weights.csv` 均存在；
- `client_weights.csv` 为 201 行；
- 三个 seed 的最终模型 SHA-256 不同。

## 6. 按复现顺序整理的问题与结论

### 6.1 额外固定 RNG 导致 CMNIST 结果下降

早期提交 `ef72933` 在 `main` 中调用 `set_seed(seed)`，又在每次创建客户端时
调用 `set_seed(seed + client_id)`。后者会在 Flower/Ray 重建客户端时反复重置
随机状态，改变 DataLoader、局部训练、DHT 和选择轨迹，并由 FedAvgM 动量
累积放大。

固定 RNG 的代表性旧结果：

| 设置 | worst-group 均值 ± 总体标准差 |
|---|---:|
| Random | 89.7855 ± 1.1751 |
| 发布代码顺序 FedDiverse | 88.5296 ± 2.8986 |
| 论文式调度、predicted DHT | 89.5572 ± 1.6708 |
| 论文式调度、Oracle DHT | 90.0998 ± 0.7383 |

处理：提交 `d679cca` 移除两处额外播种。以上旧结果只保留为失败诊断，不得
与 official-unseeded 正式结果混用。

### 6.2 发布代码顺序与论文描述不一致

发布代码原本在首轮训练前估计 DHT，并将该轮同时作为模型更新；论文描述是
先进行 FedAvg 预训练，再估计 DHT。提交 `b71d998` 实现 1 次预训练更新、1 次
DHT-only 收集和 199 次选择更新，使 Random 与 FedDiverse 都有 200 次模型
更新。

### 6.3 DHT 与选择器诊断没有形成正式修复

CMNIST 诊断显示：SC 估计相关系数约 0.93-0.96，CI 估计正确，但 AI 估计
相关性接近 0 或为负，并有向 SC 收缩的趋势。离线重放与实际选择计数相关系数
约 0.998，未发现客户端 ID 映射错误。

Oracle、固定 3 SC + 3 AI + 3 CI 和最佳 checkpoint 均不能稳定解释或恢复
official-unseeded 正式结果。exact-3/3/3 虽使聚合群体更均衡，但 seed 42 最终
worst-group 仅为 `88.5429`，说明 DHT 多样性、群体暴露均衡和最终鲁棒性不是
简单单调关系。这些改动只作为诊断，不进入正式配置。

### 6.4 恢复 official-unseeded 后 CMNIST 成功

移除额外播种并采用论文式调度后，CMNIST FedDiverse 达到 `94.1505`，三个
数据 seed 均优于 Random，确认此前主要阻塞来自随机行为改变，而不是必须使用
Oracle DHT 或修改选择器。

### 6.5 Spawrious 数据集多次下载截断

浏览器首次得到的 5.18 GB 和后续 14.24 GB 文件均被 gzip CRC 与 tar 完整性
检查判定为截断。使用可续传下载补齐到 14,393,689,896 bytes 后，Python gzip
CRC 和 `tar -tzf` 均通过，随后上传服务器并解压。仅看资源管理器显示大小不能
证明压缩包完整，必须同时通过 gzip CRC 和 tar 结构检查。

旧 `spawrious` 包还会检查 `domain_adaptation_ds`；当前项目只使用目录 `0/1`。
服务器在需要兼容检查时使用 `domain_adaptation_ds -> 0` 符号链接，不改变本次
训练数据。

### 6.6 Spawrious GSC 绝对性能接近但相对优势消失

三次正式运行与 checkpoint 全部有效。Random 比论文高 `1.4797`，而
FedDiverse 只比论文低 `0.2603`，较高的本地 Random 基线基本消耗了论文中的
相对差距。

客户端选择总数严格满足 `2×24 + 199×9 = 1839`。三个 seed 的整体选择集中度
接近，但客户端类型构成变化明显：

| seed | CI 选择占比 | AI 选择占比 | 常规 SC 占比 | 反相关 SC 占比 |
|---:|---:|---:|---:|---:|
| 42 | 38.36% | 19.37% | 40.76% | 1.51% |
| 43 | 38.53% | 3.29% | 51.31% | 6.87% |
| 44 | 40.20% | 5.42% | 53.77% | 0.61% |

真实 AI 客户端的预测 AI 分量明显偏低，且 DHT 在 round 2 后不再更新，这可能
是运行方差来源之一。由于数据划分和未固定训练 RNG 同时变化，目前不能建立
单一因果解释。SC 使用归一化互信息，不区分相关方向，因此反相关客户端的低
选择频率本身也不能证明算法错误。

### 6.7 checkpoint 与日志曾被误判

`final` 是目录而不是普通文件；正确检查对象是
`final/torchmodel.pt`。日志中的相同记录出现两遍来自两个日志 handler，不
代表同一轮训练或保存执行了两次。

### 6.8 GCI 独立 Hydra 检查缺少动态配置注册

独立运行 `hydra.compose()` 时曾出现：

```text
Could not load 'job/federated_training'
```

`job/federated_training` 由 `flower_train.py` 通过 ConfigStore 动态注册，不是
磁盘 YAML。正式训练入口不受影响。独立检查脚本需先执行：

```python
ConfigStore.instance().store(
    group="job", name="federated_training", node=Config
)
```

提交 `2c250b6` 已修复复现计划中的配置解析和数据审计脚本。

### 6.9 Spawrious GCI 恢复平均正向提升

GCI 的 Random 和 FedDiverse 均高于论文绝对值；FedDiverse 平均提高
`1.4196` 个百分点，恢复论文的方法排序。seed 42 为负提升，另外两个 seed
为明显正提升。该结果说明 GSC 上没有平均提升不是当前实现对所有 Spawrious
划分都失效的系统性证据。

### 6.10 Spawrious GAI 的 FedDiverse 优势未复现

GAI 六次正式运行均通过配置、数据、退出码、DHT-only 调度和服务器产物检查。
Random 达到 `86.2250 ± 1.9798`，与论文的 `85.86 ± 2.56` 接近；FedDiverse
仅为 `84.7003 ± 0.8445`，比论文低 `2.5797` 个百分点。FedDiverse 相对
Random 的平均差值为 `-1.5247`，与论文的 `+1.42` 方法排序相反。

逐 seed 差值分别为 `-3.3123`、`+2.3659`、`-3.6278`。由于正式协议只固定
数据划分 seed，不能把逐 seed 差值解释为严格配对效应，但三次均值已经足以判定
论文报告的 GAI 平均优势未在本次 official-unseeded 复现中恢复。所有结果均按
最终 checkpoint 保留，不增加选择性补跑。

### 6.11 Ray 启动与 GitHub 网络问题

Ray 2.6.3 曾在第 1 轮前等待 `plasma_store` socket 超时。确认磁盘、内存和
`/dev/shm` 充足后，清理旧 Ray 会话、关闭 Dashboard，并把 object store
显式设为 4 GiB。失败均发生在模型更新前，不计为有效实验。

本地网络也曾能 ping GitHub 但无法建立 TCP 443/22 连接。切换可用网络后推送
成功；这是网络连接问题，不是 Git 仓库或提交损坏。

### 6.12 四数据集分析与 10 月代码审计

已建立“数据异质性 → 客户端互补性 → 实际选择 → 聚合群体覆盖 → 模型性能”的
离线分析链，产物包括：

- `results/four_datasets_dataset_metrics.csv`；
- `results/four_datasets_client_selection.csv`；
- `results/four_datasets_exact_round_selections.csv`；
- `results/four_datasets_round_selection_metrics.csv`；
- `results/code_audit_20261020/evidence.json`。

审计发现 legacy 选择器在全零维度、共线向量、非 3 倍数参与数量等边界上存在问题；
客户端位置与 ID 的映射在候选顺序变化时存在风险；legacy pivot 不等于按伪分组样本数差
选择最平衡类别。早期固定轨迹未发现 ID 错配，不等于一般情况下不存在该风险。
这些是实现/健壮性问题，不能把修正自动视为算法创新或性能提升。

### 6.13 GAI pivot 单因素验证（2026-10-20）

分支 `baseline-validation-v1`，代码提交 `2613047` 已推送。
采用独立 validation、稳定数据顺序和受控 RNG：数据 seed=42/43/44，
训练 seed=数据 seed+100，validation seed=数据 seed+1000。每对方法的数据清单与
初始化哈希一致。该协议不同于 9 月 official-unseeded 结果。

服务器 10 项 baseline 检查及 legacy/pivot/mapping/guarded 四组 smoke 已通过。
正式仅运行 legacy 与 pivot，各三个 seed；mapping/guarded 未扩大正式训练。
本地证据：`results/baseline_validation_20261020/formal_summary.json`。

| 数据 seed | legacy 最终验证 WGA | pivot 最终验证 WGA | pivot − legacy（百分点） |
|---:|---:|---:|---:|
| 42 | 85.4386 | 83.5088 | -1.9298 |
| 43 | 84.7368 | 83.3333 | -1.4035 |
| 44 | 88.2456 | 85.7895 | -2.4561 |
| 均值 | 86.1404 | 84.2105 | -1.9298 |

六次运行最差群体均为 `y1g1`。pivot 修正未带来性能提升。
这些是 validation 指标，不能当作 test 结果与 9 月数值直接比较。
机制诊断资料曾要求从服务器导出；本地尚未发现对应诊断归档，不标为诊断已完成。

### 6.14 CMNIST 方向感知 oracle 研究（2026-10-29）

分支 `research/cmnist-direction-aware-oracle`。此批三方法 × 三 seed 共九次训练，
均使用真实分组信息，区别于 9 月采用预测矩阵的 CMNIST 论文复现：

- Random：`conf/research_cmnist_random.yaml`；
- 真实 DHT：`conf/research_cmnist_oracle_dht.yaml`，`client_info: triplets`；
- 方向感知：`conf/research_cmnist_direction_oracle.yaml`，`joint_direction_oracle`、
  `client_info: joint_oracle`，使用真实联合矩阵。

研究协议：数据 seed=42/43/44，训练 seed=数据 seed+100，validation seed=数据 seed+1000，
从训练数据独立留出 10% validation。同一 seed 下三组数据与初始化哈希一致。
200 次模型更新，使用最终 checkpoint；不依据测试结果调整均匀目标。
方向组通过贪心添加和单客户端交换，使选中客户端平均联合分布接近四群体均匀目标。

聊天中的服务器汇总记录如下，标准差使用 ddof=0：

| 指标 | Random | 真实 DHT | 方向感知 oracle |
|---|---:|---:|---:|
| 最终 validation WGA | 92.5170 ± 1.3273 | 93.2414 ± 0.2228 | 93.6468 ± 0.0807 |
| 最终 test WGA | 92.0825 ± 1.8531 | 93.4375 ± 0.5851 | 93.6267 ± 0.1045 |
| test 平均准确率 | 95.5200 | 95.7533 | 95.9200 |
| 平均联合分布目标误差 | 0.063696 | 0.031602 | 0.001975 |
| 每轮最少群体平均占比 | 10.8440% | 14.3802% | 22.7778% |

| seed | Random test WGA | 真实 DHT test WGA | 方向感知 test WGA |
|---:|---:|---:|---:|
| 42 | 89.6208 | 92.6148 | 93.5748 |
| 43 | 94.0918 | 93.7725 | 93.5329 |
| 44 | 92.5349 | 93.9252 | 93.7725 |

方向感知相对真实 DHT 平均仅提高 `0.1892` 个百分点，逐 seed 差值为
`+0.9600/-0.2395/-0.1528`。群体覆盖明显改善，但不能宣称稳定优于真实 DHT，
更不能据此证明未知属性下的矩阵估计器有效。

服务器 checkpoint 目录（均有 final 模型及验证/测试指标文件）：

| seed | 方向感知 | 真实 DHT | Random |
|---:|---|---|---|
| 42 | `20261029-030744` | `20261029-034655` | `20261029-042632` |
| 43 | `20261029-050624` | `20261029-054556` | `20261029-062600` |
| 44 | `20261029-070616` | `20261029-074601` | `20261029-082543` |

服务器运行 `tools/summarize_cmnist_direction.py` 输出 `Artifact checks passed`，
产物在 `results/cmnist_direction_formal/`：`audit.json`、`runs.csv`、`summary.txt`。
该审计不证明进程退出码及 DHT-only 运行日志，九份日志仍需另行核验。
本次本地未发现该结果目录，不能将聊天中的服务器输出等同于本地产物已完整备份。

选择频率分析（`tools/analyze_cmnist_selection.py`）已在服务器完成：三组均覆盖全部
24 客户端；方向组有效客户端数约 17，真实 DHT 约 21.5，Random 约 23.9；
最常选九人份额分别约 77.9%、59%、42%。方向组不是固定只训练九人，但选择更集中。
方向组客户端 23 在三个 seed 中分别参与 199、199、198 次选择更新。

结合客户端的六类真实分布，当前均匀目标误差 `0.0019753086` 已达到该结构下的下界。
达到相同最优值的组合不唯一，不必总包含客户端 23。集中选择与性能收益小的关系仍是
待验证解释，不能作因果归因。此前建议的额外 15 次训练已暂缓，尚未确认执行。
完整最优组合枚举、方向消融、估计属性的跨客户端方向对齐和动态效用模块均未完成；
“研究动态训练效用优先于继续堆叠均衡指标”是讨论建议，不是已验证结论或已实现功能。

### 6.15 GAI 发布源码时序对照（2026-11-03）

已在最近服务器完成 seeds 42、43、44 的 200 轮实验。
仍使用当前仓库 25 客户端、4600 样本的 `spawrious_GAI_2`，没有恢复作者历史划分。
从 `spawrious_gai_feddiverse_paper` 配置覆盖：
`paper_faithful_pretrain=false`、`pretrain_rounds=1`、`dht_collection_rounds=0`、
`rounds=200`、`update_static_info_rounds=0`。首轮估计 DHT 并训练，没有独立 DHT-only 事件。
未补齐完整训练 RNG，不能将其与另一批结果逐 seed 作严格配对因果比较。

| seed | 最终 test accuracy | 最终 test WGA | 服务器 checkpoint |
|---:|---:|---:|---|
| 42 | 92.8628 | 87.2240 | `20261103-110131/final` |
| 43 | 92.6262 | 85.3312 | `20261103-115135/final` |
| 44 | 92.1530 | 82.0189 | `20261103-124004/final` |
| 均值 ± 总体标准差 | — | 84.8580 ± 2.1511 | — |

聊天中三份日志均有 `Run finished 200 rounds`、最终保存及末轮指标。
服务器日志模式：`logs/source_schedule/gai_seed<seed>_200rounds_*.log`。
此批比历史论文式流程 FedDiverse 均值高 `0.1577`，仍比历史 Random 低 `1.3670`
个百分点；仅切换时序没有恢复 GAI 优势。不能把这个跨批次差值解释为稳定时序效应。
当前未确认这三次运行已下载至本地，不覆盖 9 月冻结归档。

### 6.16 作者回复及新一轮 GAI 重训准备（2026-11-04）

作者 Gergely 表示已失去原实验机器访问权限，可能从备份恢复配置 ID 和 WandB 产物。
目前只“认为找到了一份 GAI 配置”，不是已核实的论文表格完整实验清单。

邮件明确给出：seeds 为 `1、2、42`，仅设置一个 seed，其余共用全局 seed；
DHT 在第一轮后确定，再做 199 轮选择训练；使用最后一个 checkpoint。
作者曾试过每 50 轮刷新 DHT，未观察到改善，但未提供对应日志。

附件：`D:/federated/FedDiverse/config.yaml`，运行 ID `20241114-153425`；
作者给出的提交为 `3c92e8a5a80be836041e1f43bd921ed862156e67`，已在本地只读核对。
该提交的历史 GAI_2 矩阵有 24 个客户端：前四个各 70 样本，后 20 个各 242 样本，
合计 5120，与附件吻合。同名 GAI_2 在当前代码中是另一套 25 客户端、4600 样本划分。
因此此前 GAI 差距至少包含数据划分及 seeds 未对齐的问题，不能仅归因于算法失效。

用户已授权重训，并明确选择**按邮件时机**；随后要求直接使用新分支，
不使用独立训练包、不主动新增说明类 `.md`。训练包与本次额外说明已清理。

当前实现：

- 分支：`codex/retrain-spawrious-gai-author`，从当前研究代码 `762cae1` 创建，
  **不是**直接切换至作者历史提交；改动在工作区，未提交、未推送；
- 配置：`conf/spawrious_gai_author.yaml`；
- 独立划分名：`spawrious_GAI_2_author_20241114`，24 客户端、5120 样本；
  原 `spawrious_GAI_2` 的 25 客户端划分保留；
- 入口：`train_author_gai.py`、`src/author_protocol.py`；
- 顺序运行及汇总：`tools/run_author_gai.py`，默认正式 seeds=1、2、42；
- 检查：`tools/check_author_gai.py`，四项离线检查通过，Python 语法与 Hydra 完整解析通过；
- 事件 1：全 24 客户端模型更新；事件 2：全客户端估计 DHT，不更新模型或动量；
  事件 3–201：每轮选择 9 客户端，共 199 次更新；总计 200 次模型更新；
- `client_info: groupweights_triplets_Npredicted`，保留 legacy pivot、选择器、ID 映射；
- `research.enabled=false`，不套用 10 月研究协议的 seed+100 等派生播种；
- 虽然 YAML 中 `paper_faithful_pretrain=false`、`dht_collection_rounds=0`，
  邮件时序由新入口中的 `AuthorStrategy` 实现，不能改用原 `flower_train.py` 直接运行此配置。

仍需明确保留的假设：附件 `pretrained: ???` 沿用历史工厂默认设为 true；
首轮聚合保留 FedAvgM，邮件未确认应改为普通 FedAvg；全局种子按作者描述重建为
driver/worker 每进程首次初始化时共用一个 seed，不每轮/每客户端反复重置，
但并未证实这等同于作者当年的 worker 播种方式，也不保证逐位确定性。

**当前进度仅到本地准备，尚无这批正式结果。** 本聊天此前误用 9 月旧服务器地址探测，
连接失败不能证明 10–11 月服务器不可用。用户表示会恢复原服务器连接；
开始训练前应先核实实际目标是否为最近使用的 `cpod-1wo0rf930ytu:22211`。

## 7. 下一步与工作边界

1. 优先接续作者回复后的 GAI 新批次。确认实际服务器、项目路径与环境后，
   将本地新分支中选定的代码改动提交并同步到服务器；不要误提交其他聊天遗留的未跟踪文件。
2. 在服务器项目目录依次执行以下命令；任何阶段失败先排查，不自动重试正式实验：

   ```bash
   python tools/run_author_gai.py --stage preflight
   python tools/run_author_gai.py --stage smoke
   python tools/run_author_gai.py --stage formal
   ```

   当前服务器尚未验证这些新入口。正式训练使用最终 checkpoint；预期参与记录为
   `[24,24,9,...,9]` 共 201 行，仅事件 2 不更新模型。日志和汇总保存在 `run_logs/`。
3. 对 10 月 CMNIST 九组和 11 月 GAI 源码时序结果补核日志、退出码及本地备份，
   不能将聊天粘贴的汇总视为完整备份已完成。
4. 9 月 24 次冻结运行继续保留；新作者协议、研究协议、源码时序协议分别报告，
   不混合均值，不按测试分数筛选 checkpoint 或选择性补跑。
5. 方向感知仅有 oracle 机制证据和较小的平均 WGA 增益；暂不追加此前讨论的 15 次训练，
   不声称估计矩阵或动态效用模块已验证。
6. 用户偏好：保留 `README.md` 与现有主目录/备份 `REPRO_STATUS.md`；
   未明确要求时不新增说明类 `.md`，不再提供独立训练包。此次只更新主目录状态文件。

## 8. 新会话接续提示

先阅读本文件并检查 Git 状态。本地根目录是 `D:/federated/FedDiverse/SpuriousFL`。
当前在 `codex/retrain-spawrious-gai-author`，HEAD=`762cae1`；作者重训代码尚未提交、
未推送、未在服务器真实运行。首要任务是在确认实际服务器后运行预检、GPU smoke、
再依次训练 1、2、42。必须使用 `tools/run_author_gai.py` 对应的新入口。

旧四组结果仍冻结；CMNIST oracle 方向研究、GAI pivot 验证、GAI 源码时序对照是其他批次，
不能混用。9 月旧服务器用户已表示删除，最近实验地址为
`root@cpod-1wo0rf930ytu.podtcp.compshare.cn:22211`，当前连通性未知。
不要把本地准备写成已完成训练，不按低分补跑，不挑最佳 checkpoint，不主动新增 `.md`。

本次更新参考了项目聊天“思考_1103”“cmnist二次训练”“cmnist修改”“feddiverse2”
“数据集分析”和当前“邮件回复1”，并以本地分支、配置及 JSON/CSV 记录核对；
其中 CMNIST 九组与 11 月 GAI 源码时序数值来自用户粘贴的服务器输出，未重新连接服务器评估。
