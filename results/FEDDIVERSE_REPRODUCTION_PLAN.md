# FedDiverse 完整复现规划

> 审计日期：2026-09-28
>
> 论文：*FedDiverse: Tackling Data Heterogeneity in Federated Learning with Diversity-Driven Client Selection*，arXiv:2504.11216v2
>
> 官方仓库：<https://github.com/ellisalicante/SpuriousFL>
>
> 官方基线提交：`e0fc5ed5200339ff72c1428bb08115edb57b375d`
>
> 当前复现提交：`d5730d4864a723b6f8734a9eda4e8664280cdfdc`，分支 `reproduce-spawrious-gai`
>
> 原则：本文只把论文明确内容标为“论文给出”；源码行为单独标注；无法从论文或源码唯一确定的内容全部登记为风险。

## 0. 结论先行

当前工程已经足以复现论文的核心链路，但还不能称为“无歧义的精确复现”。阻断最终结论的不是模型结构，而是四个实现规范问题：

1. 论文要求 `FedAvg 预训练 → DHT 估计 → FedDiverse 选择`；官方发布源码没有把 DHT 估计拆成独立的预训练后阶段。当前分支增加了 metadata-only DHT 轮，以匹配论文文字。
2. 论文的 pivot class 是“预测多数组与少数组样本数差最小的类别”；源码先把每类重采样权重归一化，再取 `max(weights)`，该值通常每类都为 1，实际常退化为第一个类别。
3. 官方 selector 返回的是 `available_cids` 中的列位置，但 manager 将位置直接转成字符串 client ID。只有可用客户端顺序恰好为 `0..K-1` 时才正确。
4. 论文没有给出全部训练细节和正式配置；若仅运行仓库默认配置，并不能唯一恢复表 I。

**不需要额外加入残差连接。**论文指定的 MobileNetV2 本身已有 inverted residual/skip connection；FedDiverse 是客户端选择方法，不是要插入网络的特征融合模块。额外改残差会改变基线模型，反而不再是论文复现。

推荐采用“双轨复现”并同时报告：

- `source-faithful`：严格保持官方提交行为，用于回答“发布代码能否复现”。
- `paper-faithful`：预训练后单独估计 DHT，并修正已确认的 pivot/client-ID 问题，用于回答“论文算法能否复现”。

两条轨道不可混为一个结果；任何修正都应做单因素消融。

## 1. 证据等级与张量约定

| 标记 | 含义 |
|---|---|
| P | 论文正文、图、表或脚注明确给出 |
| S | 官方仓库源码或默认配置明确给出，但论文未明确 |
| L | 当前复现分支新增或修改 |
| X | 论文与源码冲突，或信息缺失；进入风险登记表 |

统一张量符号：

| 符号 | shape | 含义 |
|---|---:|---|
| `X` | `[B,3,H,W]` | 一批图像 |
| `F` | `[B,D]` | MobileNetV2 特征，当前 `D=1280` |
| `W_y`, `b_y` | `[C,D]`, `[C]` | 类别分类头 |
| `Y` | `[B,C]` | 类别 one-hot |
| `A` | `[B,2]` | 属性 one-hot；论文假定二元属性 |
| `N_client` | `[K,C,2]` | K 个客户端的真实/估计 interaction matrix |
| `Delta` | `[3,K]` | 论文方向的 DHT，行依次为 CI、AI、SC |
| `Theta` | `[S,P]` | 一轮 S 个客户端展平后的模型参数 |
| `eps3` | `[3,3,3]` | 三维 Levi-Civita 张量，用于 einsum 表达叉积 |

所有收缩公式均使用 `einsum`。标量除法、`log`、`softmax`、`argmin/argmax`、归一化和索引不是张量收缩，保留为逐元素操作。

基础维度校验：

```python
logits = einsum('bd,cd->bc', F, W_y) + b_y                 # [B,C]
N_batch = einsum('bc,ba->ca', Y, A)                       # [C,2]
N_global = einsum('kca->ca', N_client)                    # [C,2]
class_count = einsum('ca->c', N_global)                   # [C]
attr_count = einsum('ca->a', N_global)                    # [2]
assert logits.shape == (B, C)
assert N_batch.shape == (C, 2)
assert N_global.shape == (C, 2)
assert Delta.shape == (3, K)
```

## 2. 模块级复现表

| 名称 | 来源公式/源码 | 输入 → 输出 shape | 可学习参数 | 子操作顺序 | 主要复现风险 |
|---|---|---|---|---|---|
| 数据集与客户端划分 | P: Sec. V-A；S: `data_splits.py` | 全局样本 `[N,...]` → K 个索引集及 `[K,C,2]` | 无 | 读取标签/属性 → 留出测试集 → 按命名 split 构造目标计数 → 无放回分配样本 | 论文只描述宏观分布；精确矩阵来自源码；README 中 CMNIST split 拼写有误 |
| 图像预处理 | S: `cmnist.py`, `spawrious.py` | `[B,H,W,3]` → `[B,3,H,W]` | 无 | resize（Spawrious）→ tensor → normalization | 论文未给增强；源码正式配置关闭 crop/flip |
| 全局模型 | P: MobileNetV2、BN→GN、ImageNet 预训练；S: `mobilenet.py` | `[B,3,H,W]` → `F:[B,1280]` → `[B,C]` | backbone、分类头 | MobileNetV2 inverted residual blocks → pooling → linear head | checkpoint 来源和 hash 未在论文给出；GN 为 8 组、`eps=1e-6` 仅源码给出；不要额外加残差 |
| ERM 本地训练 | P: categorical CE；S: `subpopbench.ERM` | `[B,C]`,`[B,C]` → scalar loss → 本地参数 | 全局模型复制出的全部参数 | forward → CE → zero grad → backward → SGD step | 本地 epoch、SGD momentum 仅源码明确；论文未给 weight decay/scheduler |
| FedAvg 预训练 | P: Sec. IV-A，`T0` 小轮数；Table IV 列出 `T0=1/20` | `Theta:[K,P]` → `[P]` | 无服务器可学习参数 | 全客户端本地训练 → 等权参数平均 → 广播 | 表 I 使用哪个 `T0` 未明确；官方源码时序与论文图 2/文字不一致 |
| 真实 interaction matrix（只作审计） | P: `N^k` 定义；S: `ground_truth_matrix` | 标签 `[n_k]`、属性 `[n_k]` → `[C,2]` | 无 | one-hot → 联合计数 | 训练算法不应把真实属性/矩阵泄露给服务器 |
| biased model | P: Sec. IV-A.2；S: `matrix_inference.training` | 本地图像 → 预测正确/错误 `[n_k]` 与特征 `[n_k,1280]` | 临时 MobileNetV2 全参数 | 从预训练全局模型复制 → weighted infinite loader → GCE 训练 → 全本地集推理 | `range(steps+1)` 导致配置 50 实际 51 次更新；多类为 one-vs-rest |
| GCE | P: 引用 [43]；S: `q=0.3` | logits `[B,C]`、one-hot `[B,C]` → scalar | 无 | softmax → 取真类概率 → GCE → batch mean | 论文正文未给 `q`；数值稳定与 reduction 需锁定；公式在下节 |
| predicted majority/minority split | P: 正确预测近似 majority、错误预测近似 minority；S: `biased_prediction` | `F:[n_k,1280]`、error `[n_k]`、class `[n_k]` → 每类 split | 无 | 推理 → error label 0/1 → 按类拆分 | “预测正确=多数属性”是假设，不保证在弱 SC/GAI 情形成立 |
| pivot class | P: Sec. IV-A.3；S: `split_by_class` + `min(...)` | 每类 error counts `[C,2]` → scalar class ID | 无 | 计算每类两组数差 → 取差最小类别 | 已确认论文/源码不一致；源码统计量归一化后常全为 1 |
| attribute classifier | P: Sec. IV-A.3；S: `train_left_right` | pivot features `[n_p,1280]` → `[n_p,2]` | linear `[2,1280]` + bias `[2]`，共 2562 | class-balanced重采样 → linear → CE → SGD | 配置 10 实际 11 次更新；非 pivot 类依赖跨类泛化 |
| 估计 interaction matrix | P: `tilde N^k`；S: `estimate_interaction_matrix` | 每类 features/error → `[C,2]` | 无 | pivot 行直接用 error 0/1 计数；其余类用 attribute classifier 预测计数 | 两个属性标签允许整体翻转，但不同类预测质量可能很差 |
| CI/AI/SC 与 DHT | P: Eq. 2–4；S: `corr.py` | `[C,2]` → `[3]`；K 客户端 → `[3,K]` | 无 | 转联合概率 → marginals/entropy/MI → 拼接 | 零计数、零熵导致除零；源码仅熵 log 加 `1e-9` |
| 第一客户端：SC 概率 | P: Sec. IV-B.1；S: `select_3_clients` | `Delta:[3,K]` → client index | 无 | 按当前首行跨客户端归一化 → 无放回概率抽样 | 全零行会除零；源码先随机打乱维度，首行不一定是 SC |
| 第二客户端：互补 | P: Sec. IV-B.2；S: `select_3_clients` | 归一化 DHT `[3,K]` → client index | 无 | 论文：每客户端 triplet 做 L1 归一化；源码：每个维度跨客户端 L1，再对每客户端做 L2 → 与首客户端点积 → 取最小 | 已确认论文/源码归一化不同，可能改变方向和排序 |
| 第三客户端：正交 | P: Sec. IV-B.3；S: `select_3_clients` | 两个 `[3]` DHT → 第三个 client | 无 | 叉积 → 单位化 → 对候选点积 → 取最大 | 两向量共线时范数为 0；使用有向叉积而非绝对投影，客户端顺序会改变结果 |
| triplet 循环选择 | P: 每 3 个客户端旋转优先维度；S: `select_noreplacement` | `[3,K]`,`S` → S 个 client ID | 无 | 初始随机维度排列 → 每组三个执行上述选择 → `[2,0,1]` 旋转 | S 非 3 的倍数会超选；两次随机行置换含冗余；client index/ID 映射脆弱 |
| FedAvgM 聚合 | P: Sec. V，`beta=0.95`；S: `flower_strategy.py` | `Theta:[S,P]`,`current:[P]`,`m:[P]` → `[P]` | momentum state `[P]`（非可学习参数） | 等权平均 → pseudo-gradient → momentum → server LR 更新 | 符号约定与 Flower 实现需逐项验证；预训练轮应不使用 momentum |
| 全局评测 | P: worst-group accuracy；S: `optim_utils.evaluate` | 预测/标签/属性 `[N_test]` → global acc、`[C,2]` group acc、WGA | 无 | 全测试集推理 → 联合组累计正确/总数 → 取最小 | 验证集是测试集深拷贝；loss 累计的归一化方式不标准，但 accuracy 不受影响 |

## 3. 全部核心公式的 einsum 版本

### 3.1 interaction matrix 与异质性指标

```python
# Y:[n,C], A:[n,2]
N = einsum('nc,na->ca', Y, A)                              # [C,2]
mass = einsum('ca->', N)                                   # scalar
p_ca = N / mass                                            # [C,2]
p_y = einsum('ca->c', p_ca)                                # [C]
p_a = einsum('ca->a', p_ca)                                # [2]

H_y = -einsum('c,c->', p_y, log(p_y + eps))                # scalar
H_a = -einsum('a,a->', p_a, log(p_a + eps))                # scalar
p_ind = einsum('c,a->ca', p_y, p_a)                        # [C,2]
MI = einsum('ca,ca->', p_ca, log((p_ca + eps)/(p_ind+eps)))# scalar

CI = 1 - H_y / log(C)
AI = 1 - H_a / log(2)
SC = 2 * MI / (H_y + H_a)
Delta_k = stack([CI, AI, SC], axis=0)                       # [3]
Delta = stack(Delta_clients, axis=1)                        # [3,K]
```

维度断言：`p_ca:[C,2]`，`p_y:[C]`，`p_a:[2]`，`p_ind:[C,2]`，三个指标均为 scalar。推荐测试实现使用与 `corr.py` 相同的 `sklearn.metrics.mutual_info_score(contingency=N)` 做数值对照。

### 3.2 CE 与 GCE

```python
logits = einsum('bd,cd->bc', F, W_y) + b_y                 # [B,C]
log_p = log_softmax(logits, axis=1)                        # [B,C]
CE = -einsum('bc,bc->', Y, log_p) / B                      # scalar

p = softmax(logits, axis=1)                               # [B,C]
p_true = einsum('bc,bc->b', Y, p)                         # [B]
GCE = einsum('b->', (1 - p_true**q) / q) / B              # scalar
```

### 3.3 pivot、属性分类与估计矩阵

```python
# E:[n,2] 是 predicted-correct/error one-hot；Y:[n,C]
error_count = einsum('nc,ne->ce', Y, E)                   # [C,2]
gap = abs(error_count[:,0] - error_count[:,1])            # [C]
pivot = argmin(gap)                                       # scalar index

attr_logits = einsum('nd,ad->na', F_pivot, W_a) + b_a    # [n_p,2]
attr_pred = one_hot(argmax(attr_logits, axis=1), 2)        # [n_p,2]
N_hat_pivot = einsum('nc,na->ca', Y_pivot, E_pivot)       # [C,2]，仅 pivot 行非零
```

对非 pivot 类，用属性分类器输出替换 `E_pivot`，再以 `einsum('nc,na->ca', ...)` 累计。源码将 pivot 行的 error label 直接视为两个伪属性标签。

### 3.4 三客户端选择

```python
# Delta:[3,K]；以下严格表达官方源码；进入函数时行顺序可被随机置换
row_mass = einsum('hk->h', Delta)                          # [3]
R = Delta / row_mass[:,None]                               # [3,K]，每维跨客户端 L1
p_first = R[0,:]                                           # [K]
k1 = categorical_without_replacement(p_first)

col_norm = sqrt(einsum('hk,hk->k', R, R))                  # [K]
U = R / col_norm[None,:]                                   # [3,K]，每客户端 L2
similarity = einsum('hk,h->k', U, U[:,k1])                 # [K]
k2 = argmin(mask_selected(similarity))

cross12 = einsum('hij,i,j->h', eps3, U[:,k1], U[:,k2])     # [3]
cross12 = cross12 / sqrt(einsum('h,h->', cross12, cross12))
alignment = einsum('hk,h->k', U, cross12)                  # [K]
k3 = argmax(mask_selected(alignment))
```

论文 Sec. IV-B.2 对第二客户端写出的归一化则是：

```python
client_mass = einsum('hk->k', Delta)                       # [K]
V = Delta / client_mass[None,:]                            # [3,K]，每客户端 L1
paper_similarity = einsum('hk,h->k', V, V[:,k1])           # [K]
```

`V` 与源码的 `U` 一般不相同。这不是转置记号即可消除的差别，必须作为独立消融项。

### 3.5 等权 FedAvg 与当前 FedAvgM

```python
alpha = ones(S) / S                                        # [S]
theta_avg = einsum('sp,s->p', Theta, alpha)                # [P]
delta = theta_avg - theta_current                           # [P]
m_new = beta * m_old + delta                               # [P]，与当前源码一致
theta_new = theta_current + eta_server * m_new             # [P]
```

注意：当前源码的 FedAvgM momentum 没有乘 `(1-beta)`；这与其所引用的 Flower FedAvgM 形式一致，但与某些优化器文献写法不同。报告时必须写清具体更新式，不能只写“使用 FedAvgM”。

### 3.6 评测指标

```python
# pred_ok:[N] 为 0/1；Y:[N,C]，A:[N,2]
global_acc = einsum('n->', pred_ok) / N
group_total = einsum('nc,na->ca', Y, A)                    # [C,2]
group_correct = einsum('n,nc,na->ca', pred_ok, Y, A)       # [C,2]
group_acc = group_correct / group_total                    # [C,2]
worst_group_acc = min(group_acc[group_total > 0])
```

## 4. 完整超参数表

### 4.1 论文明确给出（P）

| 类别 | 参数 | 值 | 位置/备注 |
|---|---|---:|---|
| 模型 | backbone | MobileNetV2 | Sec. V-B |
| 模型 | normalization | 用 GroupNorm 替换 BatchNorm | Sec. V-B；组数未给 |
| 初始化 | backbone 初始化 | ImageNet pretrained | Sec. V-B；checkpoint 未给 |
| 本地训练 | loss | categorical cross-entropy | Sec. V-B |
| 本地训练 | client learning rate | `0.001` | Sec. V-B |
| 本地训练 | batch size | `28` | Sec. V-B |
| 联邦训练 | 模型更新轮数 | `T=200` | Sec. V-B |
| 聚合 | client aggregation weight | equal | Sec. V-B |
| 聚合 | server optimizer | FedAvgM | Table I 实验 |
| 聚合 | FedAvgM momentum | `beta=0.95` | Sec. V-E.1 |
| 选择 | active clients | `9` | 除 GCI-100 外 |
| 选择 | GCI-100 active clients | `12` | 100 客户端情形 |
| DHT | attribute cardinality | `2` | Sec. IV-A |
| 评测 | repetitions | `3` | Sec. V-B |
| 评测 | primary metric | worst-group accuracy，均值与标准差 | Sec. V-B |
| 评测 | test distribution | balanced global test | Sec. V-B；“balanced”的精确定义未给 |
| 数据 | Spawrious test ratio | `10%`，按 `(class,attribute)` 留出 | Sec. V-A |
| 数据 | 客户端数 | GSC/GCI/CMNIST 24；GAI/Spawrious4 25；WaterBirds 30；GCI-100 100 | 页 6 脚注 |

### 4.2 需推断或由源码固定（S/L，不可冒充论文值）

| 参数 | 当前采用值 | 证据 | 推荐处理 |
|---|---:|---|---|
| local epochs / round | `1` | `conf/federated_training.yaml`；正式配置 | 固定为 1，并在报告中标“源码值” |
| local optimizer | SGD | `client_opt/all_defaults.yaml` | 固定；记录 torch 版本 |
| client SGD momentum | `0.9` | 同上 | 固定；论文未给 |
| weight decay | 未传入，等价默认 `0` | `get_base_optimizer` | 固定为 0 |
| LR scheduler | 无 | `ERM._init_model` | 禁用 |
| GN groups / eps | `8` / `1e-6` | `mobilenet.py` | 固定；跑模型结构断言 |
| model drop rate / drop-path rate | `0.2` / `0.2` | `mobilenet.py` | 固定；注意 train/eval mode |
| classifier input dim | `1280` | `mobilenet.py` | 固定 |
| temporary pretrained head dim | `2028` | `mobilenet.py` checkpoint 兼容技巧 | 必须验证 checkpoint load 后再替换为 C 类 |
| server learning rate | `0.1` | `server_opt/all_defaults.yaml` | 固定；论文未给 |
| FedAvgM initial momentum | 全零 | `flower_strategy.py` 首次聚合时创建 | 固定 |
| `beta_2` / `tau` | `0.99` / `1e-9`，但 FedAvgM 路径不用 | server defaults | 记录为 inactive，不纳入调参 |
| biased optimizer | SGD + ReSample | defaults + DHT loader | 固定，并记录真实采样权重 |
| DHT configured biased steps | `50` | `client_opt/all_defaults.yaml` | 源码值，不是论文正文值；实际执行 51 |
| DHT configured attribute steps | `10` | 同上 | 源码值，不是论文正文值；实际执行 11 |
| GCE q | `0.3` | 同上 | 源码值；论文只说明采用 GCE，未给 q |
| DHT refresh | `0`，即只估计一次 | `update_static_info_rounds` | 固定 |
| Spawrious input | `224×224` | dataset config | 固定 |
| Spawrious normalization | ImageNet mean/std | `spawrious.py` | 固定 |
| Spawrious augmentation | crop 0、flip false | 正式配置 | 禁用 |
| CMNIST input | `28×28` RGB | `cmnist.py` | 固定 |
| CMNIST normalization | 三通道 MNIST mean/std | `cmnist.py` | 固定 |
| CMNIST color RNG | hard-coded `RandomState(42)` | `cmnist.py` | 明确与实验 seed 独立 |
| Spawrious split seed | dataset loader 默认 `0` | `spawrious.get_dataset` 未从 conf 传 seed | 固定为 0，另登记风险 |
| 客户端 split seed | experiment seed | `get_envs/data_splits.py` | 当前 42/43/44 |
| 重复 seeds | `42,43,44` | 当前复现实验选择 | 不是论文明确 seeds |
| Python | conda `3.10.14`；README 写 `3.10.11` | 环境文件/README 冲突 | 优先冻结可运行 lock，报告差异 |
| flwr / timm / torchvision | `1.8.0 / 1.0.3 / 0.18.0` | `conda_env.yml` | 固定 |
| PyTorch/CUDA build | 环境只明确 CUDA 12.1，PyTorch 版本未在筛选结果中确认 | conda env | 运行时导出完整版本 |
| Flower `fraction_fit` | `1.0`，选择激活后被 `num_active_clients` 覆盖 | `flower_train.py` | 预训练全客户端；选择阶段 9/12 个 |
| Flower `fraction_evaluate` | `1e-6`，但 `min_evaluate_clients=1` | `flower_train.py` | 每轮抽 1 个 client proxy；其本地 `test_data` 实为同一 global validation/test 副本 |
| Flower min fit/eval | `1 / 1` | `flower_train.py` | 记录；`sample_size` 在预训练由 fraction 决定 |
| Ray resources | server 8 CPU/1 GPU；每 client 2 CPU/1 GPU | `conf/machine/env.yaml` | 属运行参数，不影响算法；按硬件调整时记录 |
| logging | 当前正式配置 `wandb=false` | 四个正式配置 | 使用本地完整日志替代 |
| Flower 事件数（当前 paper-faithful） | `201` | L：1 FedAvg update + 1 DHT-only + 199 FedDiverse updates | 以“200 model updates”计数，DHT-only 不计更新 |
| 主实验 `T0` | 当前取 `1` | 论文 Table IV 比较 `1` 与 `20`，但未明确声明表 I 统一采用哪个；当前配置选择 1 | 标为推断值；向作者确认 |

### 4.3 必须通过实验决定（E）

这些参数没有唯一的论文/源码答案，不应在主结果前偷偷优化：

| 决策项 | 最小候选集 | 推荐默认 | 决策指标 |
|---|---|---|---|
| 论文式 vs 源码式 DHT 时序 | source / paper | 双轨都跑；主文 paper-faithful | DHT 估计误差、WGA |
| pivot 实现 | official / paper-gap / 最大 error-balance 的等价实现 | paper-gap | `N_hat` vs `N_true` 的 MAE、DHT Spearman |
| DHT step 计数 | 51/11（现源码） vs 精确 50/10 | source 轨 51/11；paper 轨精确 50/10 | DHT 质量，不先看最终测试挑参 |
| selector 零向量策略 | epsilon / uniform fallback / 重抽 | epsilon + uniform fallback | 无 NaN、选择频率稳定 |
| 叉积方向 | signed / absolute | source 轨 signed；修正版先 signed | 合成几何单测与作者回复 |
| 正式 seed 方案 | 数据 seed 与训练 seed耦合/解耦 | 解耦并记录两者 | 可重复性与方差 |
| checkpoint 选择 | final / best validation | final | 论文未说 best；避免测试集选模 |
| 标准差定义 | population (`ddof=0`) / sample (`ddof=1`) | 同时报出，表格主值 `ddof=0` 以贴源码 `np.std` | 与论文表格舍入最接近只能作旁证 |

## 5. 数据管线

```text
原始数据
  ├─ CMNIST: MNIST → y=(digit>=5) → hard-coded RNG 生成 red/green 属性 → RGB mask
  └─ Spawrious: path/location/breed 元数据 → 过滤前2背景与前2犬种
        ↓
全局 train/test
  ├─ CMNIST: 官方 MNIST train/test；test 的 confounding_factor=0.5
  └─ Spawrious: 每个 (breed,location) 内 random_state=0 留出10%
        ↓
客户端训练划分
  ├─ spawrious2       → 24 clients，GSC/CMNIST GSC
  ├─ spawrious_GCI    → 24 clients
  └─ spawrious_GAI_2  → 25 clients
        ↓
每客户端 SubsetDataset → shuffled DataLoader(batch=28)
        ↓
MobileNetV2-GN 本地训练 / 一次性 DHT 估计
        ↓
全局 test DataLoader → global accuracy + 每个 (y,a) accuracy + WGA
```

四个已完成数据划分的审计锚点：

| 数据集 | split | K | pooled `[y0a0,y0a1,y1a0,y1a1]` | client size |
|---|---|---:|---:|---|
| CMNIST GSC | `spawrious2` | 24 | `[1760,640,640,1760]` | 每个 200 |
| Spawrious GSC | `spawrious2` | 24 | `[1760,640,640,1760]` | 每个 200 |
| Spawrious GCI | `spawrious_GCI` | 24 | `[1760,1760,640,640]` | 每个 200 |
| Spawrious GAI | `spawrious_GAI_2` | 25 | `[2000,500,2000,100]` | 155–208，总计 4600 |

每次运行必须导出：原始文件清单 hash、train/test 联合计数、每客户端 `[C,2]` 真值计数、客户端样本 ID hash。只比较同一数据清单和同一划分 hash 的方法。

## 6. 训练循环伪代码

### 6.1 推荐的 paper-faithful 主轨

```python
set_and_log_all_rngs(data_seed, train_seed, worker_seed, selector_seed)
train_clients, global_test = build_data_pipeline(config)
theta = load_mobilenetv2_gn_imagenet(checkpoint_hash)

# T0=1 个“模型更新”预训练轮；所有客户端、等权、标准 FedAvg
for t in range(T0):
    local = []
    for k in all_clients:
        theta_k = local_erm(theta, train_clients[k], epochs=1,
                            batch=28, lr=1e-3, momentum=0.9)
        local.append(flatten(theta_k))
    theta = einsum('kp,k->p', stack(local), ones(K)/K)

# 单独的 metadata-only 阶段，不改变 theta，不改变 FedAvgM momentum
for k in all_clients:
    biased = deepcopy(theta)
    biased = train_gce(biased, train_clients[k], q=0.3, exact_steps=50)
    features, error, y = infer_error_and_features(biased, train_clients[k])
    count = einsum('nc,ne->ce', one_hot(y,C), one_hot(error,2))
    pivot = argmin(abs(count[:,0] - count[:,1]))
    psi = train_linear_attribute_classifier(features[y==pivot], error[y==pivot], exact_steps=10)
    N_hat_k = estimate_all_class_rows(features, y, error, pivot, psi)  # [C,2]
    Delta[:,k] = dht_einsum(N_hat_k)                                  # [3]

m = zeros(P)
for t in range(T0, 200):
    selected = feddiverse_select(Delta, S=9, selector_rng=round_rng(t))
    local = []
    for k in selected:
        theta_k = local_erm(theta, train_clients[k], epochs=1,
                            batch=28, lr=1e-3, momentum=0.9)
        local.append(flatten(theta_k))
    theta_avg = einsum('sp,s->p', stack(local), ones(S)/S)
    delta = theta_avg - theta
    m = 0.95*m + delta
    theta = theta + 0.1*m
    log_round(t, selected, Delta[:,selected], global_test_metrics(theta))

save_final(theta, config, hashes, logs)
```

### 6.2 source-faithful 对照轨

不修源码逻辑，仅锁定官方提交、环境与显式 override。必须保留：官方 DHT 时序、pivot 行为、51/11 实际步数、selector 原始 client-index 映射和随机维度顺序。该轨结果只能解释为“released code reproduction”。

## 7. 评测协议

1. **主指标**：最终第 200 次模型更新后的 global accuracy 与 worst-group accuracy。禁止在测试集上选择最佳轮。
2. **组定义**：`(class y, attribute a)` 的笛卡尔积；只在测试集中实际存在的组上计算；正式数据应断言全部 `C×2` 组非空。
3. **重复**：至少 3 次。记录 data split seed、model/RNG seed、Ray worker seed、selector seed；不能只记录一个笼统 `seed`。
4. **汇总**：逐 seed 保存 group-level 分子/分母；再计算均值、population std 和 sample std。论文主表只声明 mean/std，未声明 `ddof`。
5. **配对对照**：FedDiverse 与 Random 共用数据划分、初始 checkpoint、模型初始化和可配对的训练 seed；selector RNG 独立。
6. **诊断指标**：
   - `N_hat` 对 `N_true`：cell MAE、总变差距离、每个 DHT 维度 Pearson/Spearman。
   - selector：每轮 client IDs、DHT、每轮 pooled group counts、pairwise TV、最小组暴露、长期选择频率 CV。
   - 性能：global accuracy、每组 accuracy、WGA；训练 loss 仅作健康检查。
7. **公平性**：Random 也必须有 200 次模型更新；paper-faithful 的 DHT-only Flower 事件不计入模型更新次数。
8. **显著性边界**：3 次运行只适合复现论文汇总，不足以支持强因果结论。报告每个 seed，不只报告均值。

## 8. 风险登记表（按严重度排序）

| ID | 严重度 | 风险/证据 | 可能解释 | 推荐默认值与验证 |
|---|---|---|---|---|
| R1 | 阻断 | 论文先 FedAvg 预训练再估计 DHT；官方源码未分离 | 发布代码遗漏；论文描述简化；内部实验分支未发布 | 双轨；paper-faithful 使用 1 update + 1 metadata-only + 199 updates；向作者确认 |
| R2 | 阻断 | pivot 论文取 error-count gap 最小；源码 `max(normalized_weights)` 通常全为 1 | 实现 bug；本意可能存未归一化 gap；变量第三项写错 | 主修正版用 `argmin(abs(count[:,0]-count[:,1]))`；先做 DHT-only 配对实验，再决定是否重跑完整 GSC/GAI |
| R3 | 高 | `range(steps+1)` 令 50/10 变 51/11 | 作者把参数当末端 index；普通 off-by-one | source 轨保留；paper 轨精确执行 50/10；记录 optimizer-step counter |
| R4 | 高 | selector 返回可用列表位置，manager 直接当 client ID | 假设 Flower 注册顺序恒为数字顺序；潜在映射 bug | 改为 `sampled_cids=[available_cids[i] for i in selected]`；用乱序注册集成测试 |
| R5 | 高 | selector 初始对 DHT 三行随机 permutation 后又 `shuffle`，首维不总是 SC | 用随机初始维度实现“增强 variability”；与论文第一步固定 SC 有冲突 | source 轨保留；paper 轨固定首组三维顺序 `[SC,CI,AI]`，之后每组三元循环旋转；向作者确认 |
| R6 | 高 | 叉积对共线/零 DHT 无保护 | 数据恰好不触发；作者依赖 `nan_to_num` 但未覆盖叉积 | `norm<1e-12` 时从未选客户端中按最大到两点最小距离选；记录 fallback 次数 |
| R16 | 高 | 论文对每客户端 triplet 做 L1 归一化；源码先对每个维度跨客户端 L1，再对客户端列 L2 归一化 | 实现变更未写入论文；作者可能意图使用方向余弦；内部版本不同 | source 轨保持源码；paper 轨实现论文 `V`；以固定 DHT 比较第二/第三客户端排序，并向作者确认 |
| R7 | 高 | 官方仓库无正式表 I 配置和 seeds | 配置保存在未发布脚本/集群系统；README 只给拼装参数 | 所有源码派生值单列；保存 resolved Hydra YAML、commit、环境、数据 hash |
| R8 | 高 | RNG 未在官方主流程统一固定；selector 使用全局 NumPy/Python RNG，Ray 会改变状态 | 论文只关注跨运行方差；发布代码未追求 bitwise repeatability | 使用显式 RNG 对象；数据、训练、worker、selector 分种子；先验证同 seed 两次选择轨迹一致 |
| R9 | 高 | ImageNet GN checkpoint 的来源/hash 未写入论文 | 自训练 timm checkpoint；内部 artifact | 保存当前文件 SHA-256、加载 key、参数统计；缺文件时不得以 timm BN 权重静默代替 |
| R10 | 高 | GAI 客户端大小不等，但论文说等权客户端 | equal 表示 client-level equal 而非 sample-weighted；与 FedAvg 标准定义不同 | 保持 `weight_clients=same`；额外报告 sample-weighted 只作消融，不替换主结果 |
| R11 | 中 | “balanced global test”与 CMNIST 随机生成的属性数不一定逐格精确相等 | balanced 指生成概率 0.5；或指期望均衡 | 报告实际 `[C,2]` test counts；不通过再采样偷偷改测试集 |
| R12 | 中 | Spawrious test split seed 固定 0，未接实验 seed | 作者希望所有运行共用测试集；配置传参遗漏 | 固定为 0 并写入 manifest；训练划分 seed 单独记录 |
| R13 | 中 | validation 是 test 的深拷贝 | 只为 Flower 每轮监控；并未 best-checkpoint 选模 | 仅使用 final checkpoint；如需调参，另建 train-derived validation，最终测试只评一次 |
| R14 | 中 | `optim_utils.evaluate` 将 batch mean loss 累加后除样本数 | loss 仅诊断；实现归一化 bug | 不用该 loss 跨实验定量比较；另算样本加权 CE；accuracy/WGA不受影响 |
| R15 | 中 | S 非 3 倍数时 selector 每次加 3 会超选 | 论文实验 9/12 都是 3 的倍数 | assert `S % 3 == 0`；扩展实验需定义截断规则 |
| R17 | 中 | 多类 one-vs-rest 和二类 biased model 路径不同 | 设计如此；论文只简述 | Spawrious4 单独写测试；当前四个数据集均为二类，不用多类结果外推 |
| R18 | 低 | Python 3.10.11/3.10.14、依赖 build 差异 | README 与导出环境时间不同 | 输出 `pip freeze`、GPU/driver/cuDNN 和 determinism flags |
| R19 | 低 | MobileNetV2 内含残差，用户可能误加额外 residual | 将其他“缝合模块”建议误套到客户端选择论文 | 不新增；模型 state_dict 与官方结构逐 key 对比 |

## 9. 官方代码与当前分支逐模块对比

对比范围：官方 `e0fc5ed...` → 当前 `d5730d4...`。

| 模块 | 是否相同 | 差异与影响 |
|---|---:|---|
| `src/corr.py` | 是 | CI/AI/SC、global/local 指标未改 |
| `src/optimizers/matrix_inference.py` | 是 | biased model、pivot、attribute classifier、51/11 步行为均保持官方；因此 pivot 风险仍存在 |
| `src/optimizers/weighting_strategy.py` | 是 | 三客户端选择几何与随机行排列保持官方 |
| `src/flower_manager.py` | 是 | selector index/client-ID 映射风险保持官方 |
| `src/optimizers/dataloaders.py` | 是 | weighted infinite sampling 未改 |
| `src/optimizers/subpopbench.py` | 是 | ERM/GCE/SGD 路径未改 |
| `src/datasets/cmnist.py` | 是 | CMNIST hard-coded color RNG 未改 |
| `src/datasets/spawrious.py` | 是 | 10% 分层测试与 transform 未改 |
| `src/datasets/data_splits.py` | 是 | 命名 client matrices 未改 |
| `src/datasets/data_preparation.py` | 是 | val=test copy 行为未改 |
| `src/models/mobilenet.py` | 是 | MobileNetV2-GN、内置 inverted residual、checkpoint load 未改 |
| `src/models/model_utils.py` | 是 | 参数序列化未改 |
| `src/flower_client.py` | 否 | 新增 `metadata_only`：从预训练全局参数估计并上报 DHT，不做本地模型更新 |
| `src/flower_strategy.py` | 否 | 新增 paper-faithful 时序；DHT-only 不聚合、不推进 momentum；预训练使用标准 FedAvg；选择从下一轮开始 |
| `conf/server_opt/all_defaults.yaml` | 否 | 新增 `paper_faithful_pretrain` 和 `dht_collection_rounds`，默认关闭，保持兼容 |
| 正式四数据集配置 | 当前新增 | 明确 1 epoch、batch 28、200 model updates、FedAvgM、9 clients、split 等 |
| `src/utils.py` | 否 | `set_seed` 增加 CUDA/cudnn deterministic 设置；但当前主流程没有统一调用，不能据此宣称完全可复现 |
| `flower_train.py` | 否 | `int(cid)` 抽成局部变量，语义不变 |

当前分支对论文时序的修正是有意的 paper-faithful 改动；它不是官方源码原样复现。核心 DHT 与 selector 没改，所以不能自动消除 R2–R6。

## 10. 实施步骤清单与可验证中间产出

| 步骤 | 操作 | 必须产出/通过条件 |
|---:|---|---|
| 1 | 冻结证据 | `manifest.json`：paper version、official/local commit、remote URL、所有 config SHA-256 |
| 2 | 冻结环境 | `environment.txt`：Python/PyTorch/CUDA/cuDNN/GPU/flwr/timm；能从空环境 import |
| 3 | 冻结预训练权重 | checkpoint SHA-256、state_dict key 数、加载 strict 成功、缺失/多余 key 均为 0 |
| 4 | 审计数据 | 每数据集 raw file hash、train/test `[C,2]`、客户端 `[K,C,2]`、样本 ID 不重叠断言 |
| 5 | 审计模型 | 输入 28/224 的 forward shape；GN 层数/组数；总参数约与论文 Table II 的 `2.23e6` 一致；无额外残差改动 |
| 6 | 单测 einsum 指标 | 随机矩阵上 einsum CI/AI/SC 与 `corr.py` 误差 `<1e-8`；shape JSON 全通过 |
| 7 | 合成 pivot 单测 | 构造 `[[90,10],[50,50]]`，论文 pivot 必须为 class 1；官方实现会暴露反例 |
| 8 | DHT-only 对照 | official-pivot vs paper-pivot，同一模型/数据/RNG；输出每客户端 `N_true/N_hat`、MAE/TV、CI/AI/SC 相关性 |
| 9 | selector 单测 | 手工 6/9 个 DHT，验证首选概率、最小点积、Levi-Civita 叉积、无重复、乱序 `available_cids` 映射 |
| 10 | 时序集成测试 | 日志证明：round 1 改 theta；DHT-only 不改 theta/momentum；之后总计 200 次 theta 更新 |
| 11 | 确定主规范 | 根据步骤 8，不看最终测试性能，冻结 pivot/step/fallback 选择；写 `SPEC_DECISIONS.md` |
| 12 | 单 seed smoke | 每数据集 3–5 轮：无 NaN、每轮 9 个唯一客户端、group metrics 完整、checkpoint 可重载 |
| 13 | 随机基线 | 42/43/44；与 FedDiverse 配对的数据/初始化 hash 一致；输出每 seed final metrics |
| 14 | FedDiverse 正式运行 | 保存 round-level selections、DHT、group exposure、global/group accuracy、final checkpoint |
| 15 | 三 seed 汇总 | CSV 同时含每 seed、mean、`std(ddof=0)`、`std(ddof=1)`；自动生成论文表格式结果 |
| 16 | 因果诊断 | 完成“数据异质性 → 客户端互补性 → 选择结果 → diversity → global/WGA”链路；只作关联结论 |
| 17 | 差异报告 | source-faithful、paper-faithful、论文表 I 三列；逐项解释配置/数据/随机性差异 |
| 18 | 归档 | 命令、日志、resolved config、环境、hash、checkpoints、分析脚本均可从 manifest 定位 |

停止条件：任何一项出现数据交叉污染、模型更新次数不等、DHT NaN、选择重复、checkpoint 无法重载或同 seed 轨迹不一致时，不进入正式三 seed 运行。

## 11. GitHub tracker 核验与候选 Issue 清单

### 11.1 当前官方 tracker 事实

2026-09-28 通过 GitHub REST API 读取 `state=all`：

- 独立 Issues：0。
- PR #1：`Reproduce paper`，创建于 2026-09-03，已关闭、未合并、无评论。链接：<https://github.com/ellisalicante/SpuriousFL/pull/1>。

因此下面都是**建议向作者提交/确认的候选问题**，不是官方已承认 bug。

### 11.2 建议提交顺序

| 优先级 | 候选标题 | 最小证据/问题 | 希望作者确认 |
|---:|---|---|---|
| P0 | `Pivot-class selection appears degenerate after weight normalization` | `split_by_class` 保存 `torch.max(weights)`，归一化后通常为 1；反例 `[[90,10],[50,50]]` 中源码选 0、论文定义选 1 | 第三 tuple 项本应是什么；正式实验使用了哪个版本 |
| P0 | `Clarify pre-training and one-time DHT estimation order` | 论文图 2/正文先得到 `theta_T0`，再估 DHT；发布源码未分离 metadata-only 阶段 | 表 I 使用的真实时序与轮数计法 |
| P0 | `FedDiverse selector indices may not map back to available client IDs` | selector 返回列位置，manager 使用 `str(index)`，未通过 `available_cids[index]` | 是否假设注册顺序恒定；接受修复 PR |
| P1 | `Configured DHT steps execute steps + 1 optimizer updates` | 两个训练函数使用 `range(steps+1)`；50/10 实际为 51/11 | 论文数字指配置上界还是实际更新次数 |
| P1 | `Does client selection start from SC or a random DHT dimension?` | 论文第一步固定 SC；源码进入每轮三选前随机排列 DHT 行 | 表 I 实验的维度顺序策略 |
| P1 | `Request exact Table I configs, seeds, and pretrained checkpoint hash` | 官方仓库没有完整正式 sweep 配置；checkpoint provenance 不明确 | Hydra overrides、seeds、artifact URL/hash、依赖 lock |
| P1 | `Define balanced global test set precisely` | CMNIST 的 0.5 属性由二项抽样生成，通常不是逐格精确相等 | balanced 指期望/近似，还是精确重采样 |
| P2 | `Selector behavior for zero or collinear DHT vectors` | SC 行全零会概率归一化失败；叉积范数 0 会 NaN | 官方 fallback 规则 |
| P2 | `Clarify standard-deviation convention and checkpoint selection` | 论文只说三次实验的 mean/std；未说 `ddof` 或 final/best | 表 I 的统计脚本规则 |
| P2 | `README typo for CMNIST split mode` | README 写 `sparwious2`，源码/实际配置是 `spawrious2` | 修正文档 |

提交 Issue 时每条只讨论一个问题，并附：官方 commit、最小可执行脚本、实际/期望输出、对表 I 的潜在影响。不要把当前复现结果下降直接表述为某个 bug 的因果证据。

## 12. 针对当前四个已完成实验的处置建议

当前四组运行已经适合做机制分析，但在解决 R1–R4 前，应标为“当前 paper-timing + official-core implementation”，而非最终精确复现。

| 数据集 | 当前 FedDiverse WGA | Random WGA | 下一步 |
|---|---:|---:|---|
| CMNIST GSC | `94.1505±0.3655` | `91.4942±0.4841` | 保留；作为正向 sanity anchor |
| Spawrious GSC | `87.7497±1.3650` | 同均值 | 先做 pivot DHT-only 诊断，不立即全量重跑 |
| Spawrious GCI | `91.0620±1.4475` | `89.6425±1.5364` | 保留；核验 seed 42 负增益与 DHT/选择轨迹 |
| Spawrious GAI | `84.7003±0.8445` | `86.2250±1.9798` | 最高优先级做 pivot 与 DHT 精度诊断；只有 DHT 明显改善才补跑完整对照 |

推荐的最省算力决策门：先对 GSC、GAI 的三个现有 seed 仅重算 DHT，不做 200 轮训练。若 paper-pivot 相比 official-pivot 同时改善 `N_hat` TV/MAE 和目标 DHT 维度排序，再对这两个数据集补跑完整 `official-pivot vs paper-pivot`；否则将偏差归因范围保留为“未确定”，不靠继续扫超参数解释。

## 13. 验收标准

复现可以宣称“完成”需同时满足：

- 数据、模型、更新次数、DHT 与选择轨迹都有机器可读证据；
- source-faithful 与 paper-faithful 分开报告；
- 每个论文缺失项都有风险编号，没有隐式默认值；
- 所有 contraction 公式和实际张量 shape 已通过单测；
- 三个 seed 的逐 seed 结果可见；
- 任何与论文表 I 的差异都能定位到数据、配置、代码规范或随机性，而不是仅给“结果接近/不接近”的结论。
