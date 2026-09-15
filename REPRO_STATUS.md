# FedDiverse 复现状态

最后更新：2026-09-15

## 1. 当前状态

已完成论文中以下三组 Random 与 FedDiverse 三数据 seed 正式实验：

1. CMNIST GSC：主要数值、方法排序和相对提升均已复现；
2. Spawrious GSC：FedDiverse 绝对性能接近论文，相对 Random 的优势未复现；
3. Spawrious GCI：FedDiverse 相对 Random 的平均优势已基本复现。

所有正式实验均使用数据 seed 42、43、44，报告最终模型的 worst-group
accuracy。表中“本次结果”使用总体标准差，不选择中间最佳 checkpoint。

| 数据集 | 本次 Random | 本次 FedDiverse | 本次提升 | 论文 Random | 论文 FedDiverse | 论文提升 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| CMNIST GSC | 91.4942 ± 0.4841 | 94.1505 ± 0.3655 | +2.6563 | 92.00 ± 1.61 | 94.01 ± 0.98 | +2.01 | 基本复现成功 |
| Spawrious GSC | 87.7497 ± 0.0744 | 87.7497 ± 1.3650 | 0.0000 | 86.27 ± 1.12 | 88.01 ± 0.96 | +1.74 | 绝对值接近，相对优势未复现 |
| Spawrious GCI | 89.6425 ± 1.5364 | 91.0620 ± 1.4475 | +1.4196 | 87.59 ± 2.00 | 89.91 ± 1.91 | +2.32 | 主要结论基本复现成功 |

Spawrious GSC/GCI 的 checkpoint、日志、配置和元数据均已完成本地备份；
GCI 已冻结。当前阶段是在 `reproduce-spawrious-gai` 分支复现
**Spawrious GAI**。已完成的数据集不再增加正式运行或选择性补跑。

## 2. 仓库、分支与冻结点

| 项目 | 分支 | 提交/标签 | 状态 |
|---|---|---|---|
| CMNIST GSC | `paper-faithful-official-rng` | `e1aeb84`；标签 `cmnist-gsc-official-rng-v1` | 已冻结 |
| Spawrious GSC | `reproduce-spawrious-gsc` | `a2959a6`；标签 `spawrious-gsc-official-rng-v1` | 已冻结并推送 |
| Spawrious GCI | `reproduce-spawrious-gci` | `61a6581`；标签 `spawrious-gci-official-rng-v1` | 已冻结并推送 |
| Spawrious GAI | `reproduce-spawrious-gai` | 起点 `61a6581` | 当前分支，已推送 |

关键实现提交：

- `b71d998`：加入论文式 FedAvg 预训练、DHT-only 收集与模型更新调度；
- `d679cca`：移除额外全局/客户端播种，恢复发布代码的训练 RNG 行为；
- `62bc37c`：加入 Spawrious GSC 正式配置；
- `965d5de`：加入 Spawrious GCI 正式配置；
- `2c250b6`：修复独立 Hydra 检查缺少动态 ConfigStore 注册的问题；
- `61a6581`：记录 GCI 正式结果并作为 GCI 冻结点。

仓库位置：

- 本地：`C:/Users/zy/Documents/ChatGPT/FedDiverse/SpuriousFL`；
- 服务器：`/root/SpuriousFL`；
- 远端：`https://github.com/mreason7/SpuriousFL.git`；
- 服务器：`cpod-1uk7kkln33pu.podtcp.compshare.cn`；
- 论文原文：`D:/zotero_file/storage/VZD6UMI6/Németh 等 - 2025 - FedDiverse Tackling Data Heterogeneity in Federated Learning with Diversity-Driven Client Selection.pdf`。

## 3. 环境与统一实验协议

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
- 总客户端数：CMNIST GSC、Spawrious GSC、Spawrious GCI 均为 24；
- 每轮选择客户端数：9；
- 每个客户端训练样本数：200，联邦训练总样本数 4800。

### Random 与 FedDiverse 调度

Random：200 个 Flower round，对应 200 次模型更新，不估计 DHT。

FedDiverse：201 个 Flower 事件，对应 200 次模型更新：

1. round 1：24 个客户端进行一次标准 FedAvg 预训练更新；
2. round 2：24 个客户端只估计并上传 DHT，不更新模型或 FedAvgM 动量；
3. round 3-201：每轮由 FedDiverse 选择 9 个客户端，共 199 次更新。

正式 FedDiverse 配置：

- `client_info: triplets_Npredicted`；
- `selection_method: triplets_stochasticmatrix`；
- `biased_trainer_steps: 50`；
- `left_right_trainer_steps: 10`；
- `generalized_cross_entropy_q: 0.3`；
- `biased_optimizer: ReSample`。

### seed 的准确含义

正式结果中的 `seed=42/43/44` 主要控制客户端数据划分，并非完整训练 seed。
模型初始化、DataLoader、DHT 训练、客户端选择和 Ray worker 的随机状态没有被
统一固定，这是论文时期发布代码的行为。因此逐 seed 差值不能视为严格共享
训练随机性的配对实验；正式比较口径是三次运行均值。

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

每次正式 FedDiverse 运行必须满足：

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

### 6.10 Ray 启动与 GitHub 网络问题

Ray 2.6.3 曾在第 1 轮前等待 `plasma_store` socket 超时。确认磁盘、内存和
`/dev/shm` 充足后，清理旧 Ray 会话、关闭 Dashboard，并把 object store
显式设为 4 GiB。失败均发生在模型更新前，不计为有效实验。

本地网络也曾能 ping GitHub 但无法建立 TCP 443/22 连接。切换可用网络后推送
成功；这是网络连接问题，不是 Git 仓库或提交损坏。

## 7. 后续工作与边界

已完成 GSC/GCI 独立备份、GCI 结果提交与标签冻结，并已从冻结点建立和推送
`reproduce-spawrious-gai`。后续仅推进 GAI：

1. Spawrious GAI 使用 `split_mode: spawrious_GAI`、25 个客户端、每轮选择
   9 个客户端；
2. 论文目标：Random `85.86 ± 2.56`，FedDiverse `87.28 ± 1.61`，平均提升
   `1.42` 个百分点；
3. 按配置解析、数据审计、Random smoke、FedDiverse smoke、Random 三 seed、
   FedDiverse 三 seed、总结与冻结的顺序执行；
4. 自研阶段另建分支并采用完整受控播种协议，在同一协议下重跑所有基线，不与
   本文档中的 official-unseeded 论文复现结果混用。

## 8. 新会话提示词

请先阅读 `REPRO_STATUS.md` 并检查 Git 状态。已完成 CMNIST GSC、Spawrious
GSC 和 Spawrious GCI 的 Random/FedDiverse 三数据 seed 正式实验。CMNIST
基本复现成功；Spawrious GSC 的 FedDiverse 绝对性能接近论文但相对优势未
复现；Spawrious GCI 恢复了 `+1.4196` 个百分点的平均优势。不要继续补跑上述
数据集或选择最佳 checkpoint。GSC/GCI 完整产物已经分别备份，GCI 已在
`61a6581` 通过标签 `spawrious-gci-official-rng-v1` 冻结。当前位于
`reproduce-spawrious-gai`，下一步复现 Spawrious GAI。
