# CMNIST：真实联合矩阵与方向感知选择

原九次运行结果及后续步骤见 `CMNIST_NEXT_STEPS.md`。先分析选择频率，勿重复运行本文件中的首批正式命令。

分支：`research/cmnist-direction-aware-oracle`。基于已有 `baseline-validation-v1`
的受控协议实现；不修改已冻结的 official-unseeded 实验。

## 研究问题与边界

本阶段检验：当真实标签—属性联合分布已知时，利用方向和群体互补性进行
客户端选择，是否改善最终 worst-group accuracy。本阶段不估计交互矩阵，
也不证明未知属性条件下的方向估计有效。

原 DHT 的 SC/AI/CI 均不保留全部方向。例如 `[[90,10],[10,90]]` 与
`[[10,90],[90,10]]` 的 DHT 相同，但两者等权平均后四个群体均衡。
新选择器保留矩阵的统一列语义：同一个 g 在所有客户端必须表示同一颜色。
本阶段真实 g 满足该要求；局部预测的正确/错误伪标签不能直接替换它。

## 数据路径

1. 复用 `FlowerClient.share_client_params_once(before_train=True)`：它已有
   `count_groups`，统计本客户端训练分片的真实 `(y,g)` 计数。
2. 已有 `interaction_matrix_0..3` 按行展开，即 `[y0g0,y0g1,y1g0,y1g1]`。
   不修改客户端、不额外训练偏置模型、不读取测试集计算目标。
3. 第 2 个事件沿用 metadata-only 收集；服务端已有的存储逻辑保存这些字段。
4. `MyManager` 的新 `joint_direction_oracle` 分支按当前候选 ID 读取计数，
   重建 `(候选数,2,2)` 数组，并调用新选择器。
5. 返回位置显式映射回候选客户端 ID，不能把位置直接解释成 ID。

`client_info: joint_oracle` 用来标识真实矩阵路径；原来的客户端逻辑本就总是
上报真实计数。新分支要求 research 模式和 `weight_clients: same`，并拒绝
带 `Npredicted` 标记的配置，避免误标为估计实验。

## 选择算法

每个客户端按自身总样本数归一化：`P_i=N_i/sum(N_i)`。
目标预先固定为 `T=[[.25,.25],[.25,.25]]`，不是按测试集频数拟合。
选择 9 个客户端，最小化：

`J(S) = sum((mean(P_i for i in S) - T)^2)`。

这里使用等权客户端平均，与当前等权模型聚合一致，不使用将所有原始计数
直接相加的样本量加权目标。它描述数据暴露互补性，不等价于模型更新或性能保证。

实现位于 `src/direction_selection.py`：

- 从空集合开始，每步加入使当前集合平均分布最接近 T 的未选客户端。
- 达到预算后，枚举“移出一名已选、移入一名未选”的交换，选择最好的严格改进。
- 无严格改进或达到 100 次交换上限即停止；不保证全局最优。
- 距离最小值相差不超过 `1e-12` 的候选随机打破平局，使用现有逐轮服务端
  RNG scope。相同训练 seed/轮次/输入可复现，不永久偏向最小 ID。
- 验证计数有限、非负、各客户端非空、目标为合法概率分布、选择预算合法。
- 所有客户端都参与的预训练/收集事件继续走已有全选逻辑。

该方法同时改变了表示和选择目标，不应把性能变化全部归因于“加了方向”。
后续若要隔离表示贡献，可增加同一新选择器下丢弃方向的消融。

## 三组受控配置

| 配置 | 选择依据 |
|---|---|
| `research_cmnist_random` | 随机；收集事件保留用于对齐调度 |
| `research_cmnist_oracle_dht` | 真实矩阵转 SC/AI/CI，现有 guarded 三维选择器 |
| `research_cmnist_direction_oracle` | 真实联合矩阵，新方向感知选择器 |

共同继承 `research_cmnist_base`：数据 seed 42、训练 seed 142、验证 seed 1042，
独立验证集比例 10%，24 客户端，每轮选择 9 个，等权 FedAvgM。
201 个事件包括 1 次全客户端预训练更新、1 次只收集元数据、199 次选择更新。
模型、优化器和学习率继承已复现的 CMNIST 配置。

真实 DHT 对照使用已有 guarded 选择器和 candidate ID 映射；不是原始 legacy
代码的逐行复制。各配置均启用确定性协议，独立训练/验证划分、初始模型 hash、
聚合排序等复用 `baseline_validation.py`。每轮所谓 test 指标实际来自验证集，
最终模型另行输出 `final_test_metrics.json`。不要与旧复现表直接混合。

## 检查与运行

服务器在 `/root/SpuriousFL` 的 `spurious` 环境中执行。

```bash
python tools/check_direction_selection.py
python tools/check_baseline_validation.py

python flower_train.py --config-name=research_cmnist_direction_oracle --cfg job
python flower_train.py --config-name=research_cmnist_oracle_dht --cfg job
python flower_train.py --config-name=research_cmnist_random --cfg job
```

先做短测试，每组顺序执行，避免争用 GPU。`num_steps=2` 控制真实局部训练步数：

```bash
python flower_train.py --config-name=research_cmnist_direction_oracle server_opt.rounds=3 client_opt.num_steps=2 research.run_tag=direction-smoke
python flower_train.py --config-name=research_cmnist_oracle_dht server_opt.rounds=3 client_opt.num_steps=2 research.run_tag=oracle-dht-smoke
python flower_train.py --config-name=research_cmnist_random server_opt.rounds=3 client_opt.num_steps=2 research.run_tag=random-smoke
```

检查退出成功、第 2 事件没有模型更新、第 3 事件选择 9 个不同 ID、最终模型存在。
三组 `data_manifest.sha256` 与 `initial_weights.sha256` 必须相同。
方向组新增 `joint_selection_trace.jsonl`，检查 selected_ids、selected_joint、
objective、greedy_objective，交换后的 objective 不应变大。
短测试仅验证流程，不解释精度。所有产物位于各自新的 `checkpoints/<exp_id>/`。

短测试通过后，正式运行去掉短测试覆盖项：

```bash
python flower_train.py --config-name=research_cmnist_direction_oracle
python flower_train.py --config-name=research_cmnist_oracle_dht
python flower_train.py --config-name=research_cmnist_random
```

重复时三组同时覆盖相同 `seed`、`research.training_seed`、`research.validation_seed`。
建议先固定数据/验证 seed，预先指定多个训练 seed，以隔离训练随机性；另行研究
数据划分变化。完整报告最终 checkpoint，不能根据测试分数选中间模型或选择性补跑。
现有 `run_baseline_validation.py` 专用于 GAI，本阶段不使用该 runner。

## 当前验证状态

本地新增 4 项 CPU 测试通过：相反关联互补；等权归一化和复现；预算/非法输入；
真实 manager 分支的真实计数读取和非连续候选 ID 映射。交换结果还检查了单交换
局部最优性。已有检查 9 项通过，真实 Torch 检查因本地无 PyTorch 跳过。
三组结构化 Hydra 配置均已实际 compose 并 resolve 成功。
尚未进行 Flower/GPU 短测试或正式训练，不能声称运行链路或性能已经验证。
