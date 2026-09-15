# Spawrious GAI 复现计划

本计划只推进 Spawrious GAI，不补跑已经冻结的 CMNIST GSC、Spawrious GSC
或 Spawrious GCI，也不按中间 checkpoint 选择结果。

## 目标与判定口径

- 方法：Uniform Random 与 predicted-DHT FedDiverse；
- 数据 seed：42、43、44，仅控制客户端数据划分；
- 训练 RNG：保留发布代码未统一播种的行为；
- 主指标：最后一次模型更新后的 worst-group accuracy；
- 汇总：三次运行的均值和总体标准差；
- 论文目标：Random `85.86 ± 2.56`，FedDiverse `87.28 ± 1.61`，平均提升
  `1.42` 个百分点。

## 1. 已完成的静态配置审计

发布代码中存在两个近似命名的划分：

- `spawrious_GAI`：24 个客户端，每端 200 个样本，共 4800；
- `spawrious_GAI_2`：25 个客户端，共 4600；客户端大小不完全相同。

README 明确把论文的 25-client GAI 实验映射到 `spawrious_GAI_2`。因此正式配置
使用 `split_mode: spawrious_GAI_2` 和 `num_clients: 25`。该矩阵的客户端大小应为：

```text
[155, 175, 185, 185, 185, 185, 170, 170, 182, 182,
 187, 187, 187, 187, 187, 187, 187, 187, 187, 187,
 187, 187, 187, 187, 208]
```

按 `[y0g0, y0g1, y1g0, y1g1]` 展平后的联邦训练群体总量应为
`[2000, 500, 2000, 100]`。不要把 GSC/GCI 的“每端 200、总计 4800”检查误用到 GAI。

## 2. 服务器配置解析检查

```bash
cd /root/SpuriousFL
conda activate spurious

python - <<'PY'
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from src.config_params import Config

ConfigStore.instance().store(
    group="job", name="federated_training", node=Config
)

expected = {
    "spawrious_gai_random": (200, "random", ""),
    "spawrious_gai_feddiverse_paper": (
        201, "triplets_stochasticmatrix", "triplets_Npredicted"
    ),
}
for name, want in expected.items():
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name=name)
    conf = OmegaConf.to_container(cfg, resolve=True)
    got = (
        conf["server_opt"]["rounds"],
        conf["server_opt"]["selection_method"],
        conf["server_opt"]["client_info"],
    )
    assert got == want, (name, got, want)
    assert conf["dataset_options"]["split_mode"] == "spawrious_GAI_2"
    assert conf["dataset_options"]["num_clients"] == 25
    assert conf["server_opt"]["num_active_clients"] == 9
    print(name, "OK", got)
PY
```

## 3. 服务器数据审计

本地工作区没有 Spawrious 图像，以下检查必须在已有完整数据的服务器执行：

```bash
python - <<'PY'
from collections import Counter
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from src.config_params import Config
from src.datasets import data_preparation

ConfigStore.instance().store(
    group="job", name="federated_training", node=Config
)
with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="spawrious_gai_random")
conf = OmegaConf.to_container(cfg, resolve=True)

train_ds, val_ds, test_ds = data_preparation.load_data(conf=conf)
splits = data_preparation.split_data(train_ds, conf)
sizes = [len(x) for x in splits]
indices = [int(i) for split in splits for i in split.indices]
groups = Counter()
for split in splits:
    for i in range(len(split)):
        _, _, (y, g) = split[i]
        groups[(int(y), int(g))] += 1

expected_sizes = [
    155, 175, 185, 185, 185, 185, 170, 170, 182, 182,
    187, 187, 187, 187, 187, 187, 187, 187, 187, 187,
    187, 187, 187, 187, 208,
]
assert (len(train_ds), len(val_ds), len(test_ds)) == (22808, 2536, 2536)
assert len(splits) == 25
assert sizes == expected_sizes, sizes
assert sum(sizes) == 4600
assert len(indices) == len(set(indices)), "client splits overlap"
assert [groups[(0, 0)], groups[(0, 1)], groups[(1, 0)], groups[(1, 1)]] == [
    2000, 500, 2000, 100
]

print("dataset_sizes=", len(train_ds), len(val_ds), len(test_ds))
print("client_sizes=", sizes)
print("federated_train_total=", sum(sizes))
print("federated_group_counts=", dict(sorted(groups.items())))
print("unique_client_indices=", len(set(indices)))
print("GAI_DATA_AUDIT=PASS")
PY
```

只有看到 `GAI_DATA_AUDIT=PASS` 才进入 smoke test。若数据集总量不符，先检查压缩包
完整性、解压目录以及 `domain_adaptation_ds -> 0` 兼容链接；不要修改划分矩阵来迁就错误数据。

## 4. 审计通过后的下一步

1. Random：`seed=42 server_opt.rounds=2` smoke；
2. FedDiverse：`seed=42 server_opt.rounds=3` smoke，并确认日志出现
   `DHT collection round 2 completed without a model update`；
3. 按 42、43、44 顺序完成 Random 正式运行并汇总；
4. 再按相同 seed 顺序完成 FedDiverse 正式运行并汇总；
5. 检查退出码、最终 checkpoint、配置、`client_info.json`、
   `client_weights.csv` 的 201 行，以及三个 seed 的最终模型 SHA-256 不同；
6. 记录最终模型 worst-group accuracy，完成备份、提交、标签和冻结。
