# FedDiverse 复现状态

最后更新：2026-09-14

## 结论摘要

2025 年 FedDiverse 论文的 CMNIST GSC 核心实验已经基本复现成功。

- 正式 Random（官方未固定训练 RNG 行为）：最差群体准确率
  `91.49420785 ± 0.48410444`。
- 正式 FedDiverse（预测 DHT、原始选择器、论文式调度、官方未固定训练
  RNG 行为）：最差群体准确率 `94.15049962 ± 0.36553036`。
- 论文结果：Random `92.00 ± 1.61`，FedDiverse `94.01`。
- 本次 FedDiverse 均值比论文高 `0.14049962` 个百分点；Random 均值比
  论文低 `0.50579215` 个百分点。
- FedDiverse 相对 Random 提升 `2.65629177` 个百分点；三个数据 seed
  上均为正提升。

因此，论文的核心数值、方法排序和主要结论均已在正常随机波动范围内复现。
不再需要继续运行 CMNIST Random、Oracle DHT、exact-3/3/3 或额外的
seed 42 试验。

Spawrious GSC 的 Random 与 FedDiverse 三数据 seed 正式实验也已完成：

- Random：最差群体准确率 `87.74973712 ± 0.07435403`；
- FedDiverse：最差群体准确率 `87.74973712 ± 1.36495846`；
- 论文结果：Random `86.27 ± 1.12`，FedDiverse `88.01 ± 0.96`；
- 本次 FedDiverse 与论文均值相差 `-0.26026288` 个百分点，但相对本次
  Random 的平均提升为 `0.00000000`。

因此，Spawrious GSC 的执行流程和 FedDiverse 绝对性能接近论文，但论文中
FedDiverse 相对 Random 的优势未在当前三次运行中复现。该结果作为正式结果
保留，不为追求方法排序而选择性补跑或删除 seed。

此前由提交 `ef72933` 加入的全局和客户端确定性播种改变了论文时期官方
代码的随机行为，并显著压低结果。所有包含这两处播种调用的旧结果现统一
归类为“固定 RNG 失败诊断”，不再作为正式论文复现结果。

## 复现目标与范围

使用官方 SpuriousFL 源码复现 FedDiverse 论文的核心实验。当前已经完成：

- CMNIST GSC：Random 与 FedDiverse；
- Spawrious GSC：Random 与 FedDiverse。

两组实验共同采用：

- 对比方法：Random 与 FedDiverse；
- 聚合优化器：FedAvgM；
- 客户端总数：24；
- 每次选择客户端数：9；
- 模型更新次数：200；
- 数据划分：`spawrious2`（GSC）；
- 正式 FedDiverse 使用预测 DHT 和原始选择器，不使用 Oracle 信息。

论文 PDF 目录：`D:/zotero_file/storage/VZD6UMI6/`。

## 仓库、分支与环境

- 本地仓库：`C:/Users/zy/Documents/ChatGPT/FedDiverse/SpuriousFL`
- 服务器仓库：`/root/SpuriousFL`
- Conda 环境：`spurious`
- GPU：NVIDIA GeForce RTX 4090
- CMNIST 冻结分支：`paper-faithful-official-rng`
- CMNIST 冻结标签：`cmnist-gsc-official-rng-v1`（提交 `e1aeb84`）
- Spawrious GSC 冻结分支：`reproduce-spawrious-gsc`
- Spawrious GSC 冻结标签：`spawrious-gsc-official-rng-v1`（提交 `a2959a6`）
- 当前工作分支：`reproduce-spawrious-gci`

关键提交：

- `e0fc5ed`：官方复现基线；
- `ef72933`：加入确定性播种，现仅保留为失败诊断来源；
- `0d6c68f`：Random 配置；
- `db0967b`：发布代码行为的 FedDiverse 配置；
- `b71d998`：论文式预训练、DHT 收集和模型更新调度；
- `379bbda`：sign-invariant exact-3/3/3 选择器诊断，不用于正式结果；
- `d679cca`：从 `flower_train.py` 移除两处额外 `set_seed` 调用。

分支用途：

- `reproduce-paper`：包含确定性播种的早期 Random 与发布代码行为实验；
- `paper-faithful-pretrain`：论文式预训练和 DHT 收集调度；
- `paper-faithful-selector-fix`：exact-3/3/3 负向消融；
- `paper-faithful-official-rng`：当前正式复现分支，恢复论文时期未显式固定
  训练 RNG 的行为。

### 依赖版本审计

服务器实际环境与仓库 `conda_env.yml` 一致：

- Python 3.10.14；
- PyTorch 2.3.0；
- torchvision 0.18.0；
- Flower 1.8.0；
- Ray 2.6.3；
- NumPy 1.26.4；
- pandas 2.2.2；
- Hydra 1.3.2；
- PyTorch CUDA runtime 12.1。

GPU 驱动提供更新的 CUDA 兼容能力，但 PyTorch 使用其随包安装的 CUDA 12.1
运行库；未发现影响本次复现的依赖偏差。

## 正式实验设置

论文与官方配置共同确认的设置如下：

- CMNIST，`split_mode: spawrious2`，24 个客户端；
- 每次选择 9 个客户端；
- MobileNet v2，ImageNet 预训练，GroupNorm；
- 客户端学习率 0.001，batch size 28，本地训练 1 epoch；
- 客户端等权聚合；
- FedAvgM，服务器学习率 0.1，`beta_1: 0.95`；
- 总计 200 次模型更新；
- DHT 估计前先进行一次全局 FedAvg 预训练更新。

正式 DHT 参数采用发布源码最终默认值：

- biased trainer steps：50；
- left-right trainer steps：10；
- GCE q：0.3；
- biased optimizer：`ReSample`。

历史的 `100/100/q=0.7/ERM` 参数会得到更差的 AI 估计，未用于正式结果。

## seed 与 RNG 的准确含义

正式结果中的 `seed=42/43/44` 主要是数据划分 seed，而不是完整训练 seed。

- CMNIST 颜色属性在数据集实现中固定使用 42；
- 客户端样本索引由配置中的 seed 通过独立 NumPy RNG 生成；
- 因此相同数据 seed 会得到相同的客户端数据划分；
- 模型初始化、DataLoader、DHT 训练、客户端选择和 Ray 工作进程不再由
  `flower_train.py` 统一播种，符合论文时期官方代码行为。

固定数据划分下的重复应写作 `data_seed=42, trial=1/2/3`，不能表述为三次
完整确定性 seed 42。

### 固定 RNG 失败的代码原因

提交 `ef72933` 曾加入：

- `main` 中的 `utils.set_seed(conf["seed"])`；
- `client_fn` 中的 `utils.set_seed(conf["seed"] + client_id)`。

第二处可能在 Flower/Ray 重复创建客户端时把客户端随机状态反复重置到同一
起点，进而改变样本打乱、本地训练、DHT 和选择轨迹。FedDiverse 的选择与
模型更新形成反馈循环，该偏差被 FedAvgM 动量进一步累积。删除两处调用后，
Random 和 FedDiverse 都明显恢复，FedDiverse 的恢复幅度尤其大。

## 配置与代码行为

### `conf/cmnist_random.yaml`

Random 正式配置：200 个 Flower round，等于 200 次模型更新；每轮随机选择
9 个客户端，不进行 DHT 预训练或收集。

### `conf/cmnist_feddiverse.yaml`

发布源码的原始执行顺序：200 个 Flower round；第 1 轮在本地训练前估计
DHT，并参与 FedAvgM 聚合；第 2 至 200 轮根据 DHT 选择客户端。此行为与
论文“先预训练、再估计 DHT”的文字顺序不同，只保留为实现差异诊断。

### `conf/cmnist_feddiverse_paper.yaml`

正式论文式配置，共 201 个 Flower 事件、200 次模型更新：

1. 第 1 轮：24 个客户端参与一次标准 FedAvg 预训练更新；
2. 第 2 轮：24 个客户端仅收集 DHT 元数据，不更新模型或服务器动量；
3. 第 3 至 201 轮：每轮选择 9 个客户端，进行 199 次 FedDiverse/FedAvgM
   更新。

正式结果使用：

- `client_info: triplets_Npredicted`；
- `selection_method: triplets_stochasticmatrix`。

### `conf/cmnist_feddiverse_paper_oracle_official_rng_trial1.yaml`

仅用于隔离 RNG 影响：论文式调度、Oracle DHT、原始选择器和官方未播种训练
行为。它不是正式 FedDiverse 主结果。

### `conf/cmnist_feddiverse_paper_oracle_fixed.yaml`

仅用于 exact-3/3/3 负向消融：Oracle DHT 加
`triplets_stochasticmatrix_paper_fixed`。不得作为正式选择器。

## 正式复现结果

以下指标均为百分比。主表沿用本项目约定，报告总体标准差；同时保留样本
标准差用于统计审计。所有主结果均取最后一次模型更新，不按测试集挑选最佳
checkpoint。

### 最终论文对照

| 方法 | 本次最差群体准确率 | 论文最差群体准确率 | 本次减论文 |
|---|---:|---:|---:|
| Random | 91.49420785 ± 0.48410444 | 92.00 ± 1.61 | -0.50579215 |
| FedDiverse | 94.15049962 ± 0.36553036 | 94.01 | +0.14049962 |

FedDiverse 相对 Random 的平均提升为 `2.65629177` 个百分点；论文均值差为
`2.01` 个百分点。

### 各数据 seed 配对结果

| 数据 seed | Random 测试准确率 | Random 最差群体 | FedDiverse 测试准确率 | FedDiverse 最差群体 | 最差群体提升 |
|---:|---:|---:|---:|---:|---:|
| 42 | 95.54000000 | 91.81636727 | 95.56000000 | 93.69261477 | +1.87624750 |
| 43 | 95.66000000 | 91.85628743 | 95.78000000 | 94.58722741 | +2.73093998 |
| 44 | 95.15000000 | 90.80996885 | 95.92000000 | 94.17165669 | +3.36168784 |
| 均值 | 95.45000000 | 91.49420785 | 95.75333333 | 94.15049962 | +2.65629177 |
| 总体标准差 | 0.21771541 | 0.48410444 | 0.14817407 | 0.36553036 | 0.60872135 |
| 样本标准差 | 0.26664583 | 0.59290443 | 0.18147543 | 0.44768143 | 0.74552835 |

### 正式 Random checkpoint

| 数据 seed | checkpoint |
|---:|---|
| 42 | `checkpoints/20260909-104452/final` |
| 43 | `checkpoints/20260909-112853/final` |
| 44 | `checkpoints/20260909-121342/final` |

### 正式 FedDiverse checkpoint

| 数据 seed | checkpoint |
|---:|---|
| 42 | `checkpoints/20260908-113933/final` |
| 43 | `checkpoints/20260909-025420/final` |
| 44 | `checkpoints/20260909-034311/final` |

### 固定数据划分下的 FedDiverse 训练稳定性

三次均使用 `data_seed=42`，训练 RNG 独立且未显式固定：

| trial | 测试准确率 | 最差群体准确率 | checkpoint |
|---:|---:|---:|---|
| 1 | 95.56000000 | 93.69261477 | `checkpoints/20260908-113933/final` |
| 2 | 95.74000000 | 93.73253493 | `checkpoints/20260908-123052/final` |
| 3 | 95.73000000 | 93.92523364 | `checkpoints/20260908-131841/final` |

- 测试准确率：`95.67666667 ± 0.08259674`；
- 最差群体准确率：`93.78346112 ± 0.10156440`；
- 三次结果高度稳定，说明正式结果不是单次幸运轨迹。

## official-RNG Oracle 隔离诊断

Oracle DHT、原始选择器、相同 `data_seed=42` 的三次独立训练结果：

| trial | 测试准确率 | 最差群体准确率 | checkpoint |
|---:|---:|---:|---|
| 1 | 95.70000000 | 92.25548902 | `checkpoints/20260908-080026/final` |
| 2 | 95.64000000 | 91.69660679 | `checkpoints/20260908-090017/final` |
| 3 | 95.96000000 | 93.73052960 | `checkpoints/20260908-103144/final` |

- 测试准确率：`95.76666667 ± 0.13888444`；
- 最差群体准确率：`92.56087513 ± 0.85796504`。

该实验首先证明 RNG 是强影响因素。预测 DHT 的正式结果高于这组三次 Oracle
均值，不能解释为预测 DHT 优于真实 DHT，因为两组使用了不同的随机训练
轨迹；Oracle 仅用于隔离诊断。

## 固定 RNG 失败诊断（非正式结果）

以下实验均包含 `ef72933` 加入的全局和客户端播种调用。它们记录问题定位
过程，但不得再用于论文主结果。

### 固定 RNG Random

| seed | 测试准确率 | 最差群体准确率 | checkpoint |
|---:|---:|---:|---|
| 42 | 93.79000000 | 89.87538941 | `checkpoints/20260903-030721` |
| 43 | 93.67000000 | 88.30339321 | `checkpoints/20260903-065458` |
| 44 | 94.29000000 | 91.17764471 | `checkpoints/20260903-080628` |

- 测试准确率：`93.91666667 ± 0.26849374`；
- 最差群体准确率：`89.78547578 ± 1.17512943`；
- 样本标准差：`1.43923374`；
- 相比正式 official-RNG Random 低 `1.70873207` 个百分点。

### 固定 RNG、发布代码执行顺序的 FedDiverse

| seed | 测试准确率 | 最差群体准确率 | checkpoint |
|---:|---:|---:|---|
| 42 | 92.82000000 | 84.67065868 | `checkpoints/20260904-022502` |
| 43 | 93.35000000 | 89.26147705 | `checkpoints/20260904-031212` |
| 44 | 94.23000000 | 91.65668663 | `checkpoints/20260904-035900` |

最差群体准确率为 `88.52960745 ± 2.89860558`，明显低且波动大。

### 固定 RNG、论文式调度、预测 DHT

| seed | 测试准确率 | 最差群体准确率 | checkpoint |
|---:|---:|---:|---|
| 42 | 93.52000000 | 89.66067864 | `checkpoints/20260904-092456` |
| 43 | 92.62000000 | 87.46105919 | `checkpoints/20260904-120941` |
| 44 | 94.09000000 | 91.54984424 | `checkpoints/20260904-125651` |

- 测试准确率：`93.41000000 ± 0.60514461`；
- 最差群体准确率：`89.55719402 ± 1.67084262`；
- 样本标准差：`2.04635593`；
- 相比正式 official-RNG FedDiverse 低 `4.59330560` 个百分点；
- 曾在共同 evaluation 182 达到 `89.89845853 ± 0.56701855`，仍不能解释
  与论文的差距。

### 固定 RNG、论文式调度、Oracle DHT、原始选择器

| seed | 测试准确率 | 最差群体准确率 | checkpoint |
|---:|---:|---:|---|
| 42 | 94.08000000 | 89.98003992 | `checkpoints/20260907-080847` |
| 43 | 93.82000000 | 89.26147705 | `checkpoints/20260907-085447` |
| 44 | 94.17000000 | 91.05788423 | `checkpoints/20260907-093954` |

- 测试准确率：`94.02333333 ± 0.14839886`；
- 最差群体准确率：`90.09980040 ± 0.73825317`；
- 样本标准差：`0.90417179`。

Oracle 仅恢复少量性能，说明在固定 RNG 失败轨迹下，DHT 预测误差不是主因。

### exact-3/3/3 Oracle 负向消融

仅运行 `data_seed=42`，不得补跑 43/44：

- checkpoint：`checkpoints/20260908-022539`；
- 201 行权重记录，199 个选择轮均严格选择 3 SC + 3 AI + 3 CI；
- 最终测试准确率：`93.69000000`；
- 最终最差群体准确率：`88.54291417`；
- 最佳最差群体准确率：evaluation 191 的 `91.04361371`；
- 从最佳到最终下降 `2.50069954` 个百分点；
- 总结：`logs/cmnist_feddiverse_paper_oracle_fixed_seed42_summary.txt`。

该修改在机械上正确、群体暴露更均衡，但性能更差，只能视为负向消融。

## 数据集与评估审计

正式环境的数据审计结果：

- 原始训练集：60,000；
- validation：10,000；
- test：10,000；
- validation 是 test 的深拷贝，符合当前官方代码行为；
- 联邦客户端数：24；
- 每客户端样本数：200；
- 联邦训练实际使用样本总数：4,800；
- 客户端真实类型：16 SC、4 AI、4 CI；
- 测试集群体 `[y0g0,y0g1,y1g0,y1g1]`：
  `[2571,2568,2505,2356]`；
- 四个测试群体接近均衡，所有方法使用同一测试集和同一评估函数。

正式 FedDiverse 虽有 201 个 Flower 事件，但第 2 个事件只收集 DHT，不更新
模型或 FedAvgM 动量，因此与 Random 一样均为 200 次模型更新。

## DHT 与选择器历史诊断

- 日志处理器重复注册会使部分日志出现两次：48 条 True 记录代表 24 个真实
  DHT，400 条 evaluation 代表 200 次真实 evaluation；
- SC 相关系数约 0.93–0.96，MAE 约 0.064–0.078；
- 观察到的 CI 估计正确；
- AI 相关性接近 0 或为负，预测有向 SC 收缩的趋势；
- 离线选择器重放与实际计数的相关系数约 0.998，旧怀疑的客户端 ID 映射
  问题在当前实验中没有实际触发；
- 原始 Oracle 选择器覆盖 24/24 客户端，有效客户端数约 19.2–19.7；
- 原始选择器平均组成约为 4 SC、2.5 AI、2.5 CI；
- DHT 两两 cosine 约 0.286–0.290，Random 约 0.48；
- 对 one-hot Oracle DHT，原始选择器严格 3/3/3 的概率仅 14.21%。

这些现象有诊断价值，但 official-RNG 正式预测 DHT 已复现论文数值，故不再
将 AI 估计误差或选择器结构视为阻塞复现的问题。

### `data_seed=42` 的群体暴露负向消融

| 方法 | 最小群体占比 | 到均匀分布的 L1 距离 | reverse-SC 比率 |
|---|---:|---:|---:|
| Random | 0.106000 | 0.476222 | 0.395000 |
| 原始 Oracle | 0.146929 | 0.314461 | 0.276382 |
| exact-3/3/3 | 0.171273 | 0.237856 | 0.216080 |

聚合群体占比：

- Random：`[0.366444,0.128667,0.133333,0.371556]`；
- 原始 Oracle：`[0.327610,0.169933,0.175516,0.326940]`；
- exact-3/3/3：`[0.308850,0.193160,0.192714,0.305276]`。

exact-3/3/3 的群体暴露最均衡，但性能最差，说明 DHT 多样性、聚合数据的
群体均衡度与最终鲁棒性并非单调关系。

## Spawrious GSC 正式复现

### 实验协议

- 数据集：Spawrious GSC，`split_mode: spawrious2`；
- 数据 seed：42、43、44，仅控制客户端数据划分；
- 训练 RNG：遵循发布代码未统一播种的行为；
- 模型：ImageNet 预训练 MobileNet v2，GroupNorm；
- 客户端：24 个，每轮选择 9 个；
- 客户端训练：学习率 0.001，batch size 28，本地训练 1 epoch；
- 服务端：FedAvgM，学习率 0.1，`beta_1: 0.95`；
- Random：200 个 Flower round，即 200 次模型更新；
- FedDiverse：201 个 Flower round，包括 1 次全客户端 FedAvg 预训练、
  1 次全客户端 DHT-only 收集和 199 次 FedDiverse/FedAvgM 更新，共 200 次
  模型更新；
- DHT：`triplets_Npredicted`；
- 选择器：`triplets_stochasticmatrix`。

正式配置：

- `conf/spawrious_gsc_random.yaml`；
- `conf/spawrious_gsc_feddiverse_paper.yaml`。

### 三数据 seed 结果

| 数据 seed | Random 测试准确率 | Random 最差群体 | FedDiverse 测试准确率 | FedDiverse 最差群体 | 最差群体差值 |
|---:|---:|---:|---:|---:|---:|
| 42 | 92.86277603 | 87.85488959 | 92.86277603 | 89.27444795 | +1.41955836 |
| 43 | 92.78391167 | 87.69716088 | 92.58675079 | 88.01261830 | +0.31545742 |
| 44 | 92.98107256 | 87.69716088 | 92.38958991 | 85.96214511 | -1.73501577 |
| 均值 | 92.87592009 | 87.74973712 | 92.61303891 | 87.74973712 | 0.00000000 |
| 总体标准差 | 0.08102542 | 0.07435403 | 0.19406970 | 1.36495846 | 1.30702449 |
| 样本标准差 | 0.09923547 | 0.09106471 | 0.23768587 | 1.67172588 | 1.60077154 |

论文表 I 中 Spawrious GSC 的最差群体准确率为 Random
`86.27 ± 1.12`、FedDiverse `88.01 ± 0.96`。本次 Random 比论文高
`1.47973712` 个百分点，FedDiverse 比论文低 `0.26026288` 个百分点。
论文中的相对提升为 `1.74` 个百分点，本次相对提升为 `0`。

该结果的准确表述是：实验流程复现通过，FedDiverse 绝对性能接近论文，
但其相对 Random 的平均优势未复现。当前只有三个数据 seed，而且训练 RNG
未固定，因此不能据此断言两种方法总体等价或 FedDiverse 无效。

### 正式 checkpoint 与日志

Random：

| 数据 seed | checkpoint |
|---:|---|
| 42 | `checkpoints/20260911-025949/final` |
| 43 | `checkpoints/20260911-035247/final` |
| 44 | `checkpoints/20260911-044304/final` |

FedDiverse：

| 数据 seed | checkpoint |
|---:|---|
| 42 | `checkpoints/20260911-084525/final` |
| 43 | `checkpoints/20260911-093656/final` |
| 44 | `checkpoints/20260911-102916/final` |

日志与总结：

- Random：`logs/spawrious_gsc/random_official_rng/`；
- FedDiverse：`logs/spawrious_gsc/predicted_official_rng/`；
- Random 总结：
  `spawrious_gsc_random_official_rng_summary.txt`；
- FedDiverse 总结：
  `spawrious_gsc_feddiverse_predicted_official_rng_summary.txt`；
- 选择频率诊断：`client_selection_frequency.txt`。

三次 FedDiverse 运行均满足：`EXIT_CODE=0`，第 2 轮为 DHT-only 且不更新
模型，第 201 轮保存 `final/torchmodel.pt`。三个最终模型文件均存在且 SHA-256
不同，确认不是重复或缺失的 checkpoint。

### predicted DHT 与客户端选择诊断

`client_weights.csv` 均有 201 行。每个 seed 的客户端选择总计均为
`1839 = 2×24 + 199×9`：前两轮全部 24 个客户端参与，之后 199 轮每轮选择
9 个客户端，执行数量完全符合配置。

扣除前两轮固定的全客户端参与后，各真实客户端类型的选择次数为：

| seed | CI（0-3） | AI（4-7） | 常规 SC（8-22） | 反相关 SC（23） |
|---:|---:|---:|---:|---:|
| 42 | 687（38.36%） | 347（19.37%） | 730（40.76%） | 27（1.51%） |
| 43 | 690（38.53%） | 59（3.29%） | 919（51.31%） | 123（6.87%） |
| 44 | 720（40.20%） | 97（5.42%） | 963（53.77%） | 11（0.61%） |

三个 seed 的总体选择集中度接近，seed 44 并未出现明显更严重的少数客户端
垄断；主要差异来自所选客户端类型的构成。

按 `spawrious2` 的真实交互矩阵计算，CI、AI、SC 客户端的主导 DHT 分量
理论上均约为 `0.531`。预测结果中，真实 AI 客户端 4-7 的 AI 值明显偏低：

- seed 42：约 `0.016-0.181`；
- seed 43：约 `0.0003-0.053`；
- seed 44：约 `0.014-0.062`。

同时，部分真实 SC 客户端被估计出较大的 AI 分量。例如客户端 12 的预测
AI 在三个 seed 中分别约为 `0.000072`、`0.137825`、`0.180893`，对应的
FedDiverse 选择次数差异也很大。所有客户端的 `last_round=2`，符合本次静态
DHT 只在第 2 轮收集、后续不更新的配置。

这些结果说明 predicted DHT 估计和后续客户端类型构成具有明显的运行间波动，
可能是 FedDiverse 方差较大的来源之一。目前数据 seed 与未固定的训练 RNG
同时变化，尚不能建立单一因果解释。另因 SC 使用归一化互信息且不区分相关
方向，客户端 23 的低选择频率本身不能作为算法错误的证据。

## Ray 启动事件

曾出现 Ray 在第 1 轮前等待 `plasma_store` socket 超时。磁盘、`/dev/shm`
和内存均充足；日志显示 raylet 实际启动略慢于 Ray 2.6.3 的默认等待时间，
驱动超时后主动关闭会话。

处理方式：

- `ray stop --force`；
- 将旧 `/tmp/ray` 移到带时间戳的备份目录；
- 关闭无用 Dashboard；
- 将 Ray object store 显式设为 4 GiB。

这些失败均发生在第 1 轮前，没有模型更新，不算有效实验，也不构成结果
重跑或 cherry-pick。Ray 基础设施参数不改变数据、模型、算法或训练超参数。

## 最终结论

1. CMNIST GSC 的 Random 与 FedDiverse 核心实验已经基本复现成功。
2. 正式 FedDiverse 使用预测 DHT、原始选择器和论文式 200 模型更新调度。
3. 三个数据 seed 上 FedDiverse 均优于 Random，平均提升 2.6563 个百分点。
4. 正式 FedDiverse 与论文值仅相差 0.1405 个百分点。
5. 正式 Random 落在论文 `92.00 ± 1.61` 的波动范围内。
6. 先 FedAvg 预训练、再收集 DHT 的论文式顺序与发布代码顺序确有差异。
7. 额外的全局和客户端确定性播种是此前复现失败的主要原因。
8. 固定 RNG 旧结果仅作为失败诊断保留，不得用于正式主表。
9. Oracle、exact-3/3/3、最佳 checkpoint 和超参数修改均不是恢复正式结果的
   必要条件。
10. 当前 CMNIST 复现阶段结束，不应继续为追求更高数值而增加试验。
11. Spawrious GSC 的 Random 与 FedDiverse 三数据 seed 实验均完整结束，
    checkpoint、配置、DHT-only 轮次和客户端选择数量均已验证。
12. Spawrious GSC FedDiverse 为 `87.74973712 ± 1.36495846`，接近论文
    `88.01 ± 0.96`；Random 为 `87.74973712 ± 0.07435403`，高于论文
    `86.27 ± 1.12`。
13. Spawrious GSC 上论文报告的 FedDiverse 相对 Random 优势未复现；该结果
    原样保留，不通过选择 seed、最佳 checkpoint 或调参改变正式结论。
14. predicted DHT 对真实 AI 客户端存在明显估计偏差，选择构成在运行间变化
    较大，可作为后续受控诊断或自研模块的研究线索，但不阻塞继续复现。

## 后续工作

当前不再运行新的 CMNIST 或 Spawrious GSC 正式实验：

1. 保存 Spawrious GSC 正式 Random 与 FedDiverse 的日志、总结、resolved
   config、checkpoint、Git 提交号和环境信息；
2. 提交本状态文档并为 Spawrious GSC 正式结果建立冻结标签；
3. 下一数据集复现 `Spawrious GCI`，先建立 Random 与 FedDiverse 的论文式
   独立配置并完成 smoke test，再按数据 seed 42、43、44 正式运行；
4. Spawrious GCI 继续使用当前已下载的 Spawrious 数据集，配置使用
   `split_mode: spawrious_GCI`、24 个客户端、每轮选择 9 个客户端；
5. 论文表 I 的 Spawrious GCI 目标值为 Random `87.59 ± 2.00`、FedDiverse
   `89.91 ± 1.91`，相对提升 `2.32` 个百分点；
6. 若 Spawrious GCI 能恢复相对提升，则继续复现 Spawrious GAI；若再次没有
   提升，再进入固定完整 RNG、共享初始化与预训练状态的受控诊断；
7. 如进入自研方法阶段，应另行建立严格受控的完整播种协议，并在相同协议下
   重新运行所有基线，不得与本次官方未播种结果混用。

## 新会话提示词

请完整阅读 `REPRO_STATUS.md` 并检查 Git 状态。CMNIST GSC 已基本复现成功：
official-RNG Random 为 `91.49420785 ± 0.48410444`，正式预测 DHT
FedDiverse 为 `94.15049962 ± 0.36553036`。Spawrious GSC 三数据 seed 也已
完成：Random 与 FedDiverse 均值同为 `87.74973712`，FedDiverse 绝对性能
接近论文 `88.01`，但相对优势未复现。不要继续运行 CMNIST 或 Spawrious GSC
正式实验，也不要选择性补跑 seed。下一步在新分支复现 Spawrious GCI，使用
已有 Spawrious 数据、`split_mode: spawrious_GCI`、24 个客户端、每轮选择
9 个客户端，并先运行 Random 和 FedDiverse smoke test。
