# Spawrious GCI 完整复现清单

## 目标与判定口径

- 正式方法：Uniform Random 与 predicted-DHT FedDiverse；
- 数据 seed：42、43、44，仅控制客户端数据划分；
- 训练 RNG：保留发布代码未统一播种的行为；
- 主指标：最后一次模型更新后的 worst-group accuracy；
- 汇总：报告三次运行的均值、总体标准差和样本标准差；
- 论文表 I 目标：Random `87.59 ± 2.00`，FedDiverse `89.91 ± 1.91`；
- 论文中的 FedDiverse 相对提升：`2.32` 个百分点。

不要依据中间最佳 checkpoint 选结果，也不要因为某个 seed 较低而删除或补跑。

## 1. 同步代码并核对版本

本地完成提交并推送后，在服务器执行：

```bash
cd /root/SpuriousFL
git fetch origin --tags
git switch reproduce-spawrious-gci
git pull --ff-only origin reproduce-spawrious-gci
git status --short --branch
git log -3 --oneline --decorate
```

工作区应干净，当前分支应为 `reproduce-spawrious-gci`。

## 2. 配置解析检查

```bash
cd /root/SpuriousFL
conda activate spurious

python - <<'PY'
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from src.config_params import Config

# flower_train.py registers this structured config before Hydra starts.
# A standalone compose script must perform the same registration explicitly.
ConfigStore.instance().store(
    group="job", name="federated_training", node=Config
)

for name in (
    "spawrious_gci_random",
    "spawrious_gci_feddiverse_paper",
):
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name=name)
    conf = OmegaConf.to_container(cfg, resolve=False)
    print(f"===== {name} =====")
    print("seed=", conf["seed"])
    print("split_mode=", conf["dataset_options"]["split_mode"])
    print("num_clients=", conf["dataset_options"]["num_clients"])
    print("rounds=", conf["server_opt"]["rounds"])
    print("selection_method=", conf["server_opt"]["selection_method"])
    print("num_active_clients=", conf["server_opt"]["num_active_clients"])
    print("client_info=", conf["server_opt"]["client_info"])
PY
```

Random 应为 200 rounds、random；FedDiverse 应为 201 rounds、
`triplets_stochasticmatrix` 和 `triplets_Npredicted`。

## 3. 数据审计

```bash
python - <<'PY'
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from src.config_params import Config
from src.datasets import data_preparation

# Match the ConfigStore registration performed by flower_train.py.
ConfigStore.instance().store(
    group="job", name="federated_training", node=Config
)

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="spawrious_gci_random")

conf = OmegaConf.to_container(cfg, resolve=True)
train_ds, val_ds, test_ds = data_preparation.load_data(conf=conf)
splits = data_preparation.split_data(train_ds, conf)

print("train=", len(train_ds))
print("validation=", len(val_ds))
print("test=", len(test_ds))
print("num_clients=", len(splits))
print("client_sizes=", [len(x) for x in splits])
print("federated_train_total=", sum(len(x) for x in splits))
PY
```

必须得到 24 个客户端、每个客户端 200 个样本、联邦训练总样本数 4800。

## 4. Random smoke test

```bash
cd /root/SpuriousFL
mkdir -p logs/spawrious_gci/smoke
set -o pipefail

python -u flower_train.py \
  --config-name spawrious_gci_random \
  seed=42 server_opt.rounds=2 \
  2>&1 | tee logs/spawrious_gci/smoke/random_seed42_smoke.log

run_status=${PIPESTATUS[0]}
echo "EXIT_CODE=$run_status" | tee -a \
  logs/spawrious_gci/smoke/random_seed42_smoke.log
```

检查：

```bash
grep '^EXIT_CODE=' logs/spawrious_gci/smoke/random_seed42_smoke.log
grep -iE 'traceback|exception|out of memory|cuda error|\bnan\b' \
  logs/spawrious_gci/smoke/random_seed42_smoke.log
grep 'aggregated eval results' \
  logs/spawrious_gci/smoke/random_seed42_smoke.log | tail -n 1
```

必须为 `EXIT_CODE=0`；两轮 smoke 的性能只用于验证执行，不用于论文比较。

## 5. FedDiverse smoke test

三轮分别对应一次预训练更新、一次 DHT-only 收集和一次选择更新：

```bash
set -o pipefail

python -u flower_train.py \
  --config-name spawrious_gci_feddiverse_paper \
  seed=42 server_opt.rounds=3 \
  2>&1 | tee logs/spawrious_gci/smoke/feddiverse_seed42_smoke.log

run_status=${PIPESTATUS[0]}
echo "EXIT_CODE=$run_status" | tee -a \
  logs/spawrious_gci/smoke/feddiverse_seed42_smoke.log
```

检查：

```bash
grep '^EXIT_CODE=' logs/spawrious_gci/smoke/feddiverse_seed42_smoke.log
grep 'DHT collection round 2 completed without a model update' \
  logs/spawrious_gci/smoke/feddiverse_seed42_smoke.log
grep -iE 'traceback|exception|out of memory|cuda error|\bnan\b' \
  logs/spawrious_gci/smoke/feddiverse_seed42_smoke.log
grep 'aggregated eval results' \
  logs/spawrious_gci/smoke/feddiverse_seed42_smoke.log | tail -n 1
```

必须为 `EXIT_CODE=0`，并明确出现第 2 轮 DHT-only 日志。

## 6. Random 三种子正式运行

建议在 tmux 会话中顺序运行，使每个 seed 使用独立 Python 进程：

```bash
tmux new -s spawrious-gci-random
```

在 tmux 中执行：

```bash
cd /root/SpuriousFL
conda activate spurious
mkdir -p logs/spawrious_gci/random_official_rng
set -o pipefail

for data_seed in 42 43 44; do
  log="logs/spawrious_gci/random_official_rng/random_seed${data_seed}.log"
  echo "START_TIME=$(date --iso-8601=seconds)" | tee "$log"
  echo "GIT_COMMIT=$(git rev-parse HEAD)" | tee -a "$log"

  python -u flower_train.py \
    --config-name spawrious_gci_random \
    seed="$data_seed" 2>&1 | tee -a "$log"

  run_status=${PIPESTATUS[0]}
  exp_id=$(grep -oE 'checkpoints/[0-9]{8}-[0-9]{6}/final' "$log" \
    | tail -n 1 | cut -d/ -f2)
  echo "DATA_SEED=$data_seed" | tee -a "$log"
  echo "EXP_ID=$exp_id" | tee -a "$log"
  echo "END_TIME=$(date --iso-8601=seconds)" | tee -a "$log"
  echo "EXIT_CODE=$run_status" | tee -a "$log"

  test "$run_status" -eq 0 || break
  test -s "checkpoints/$exp_id/final/torchmodel.pt" || break
done
```

按 `Ctrl-b`、再按 `d` 可退出 tmux 而不中止实验。

## 7. 汇总 Random

```bash
python - <<'PY'
import ast
import re
import statistics
from pathlib import Path

log_dir = Path("logs/spawrious_gci/random_official_rng")
records = []

for seed in (42, 43, 44):
    text = (log_dir / f"random_seed{seed}.log").read_text(errors="replace")
    if not re.search(r"^EXIT_CODE=0$", text, re.MULTILINE):
        raise RuntimeError(f"seed {seed} did not finish successfully")
    matches = re.findall(r"aggregated eval results (\{.*?\})", text)
    if not matches:
        raise RuntimeError(f"seed {seed} has no evaluation result")
    result = ast.literal_eval(matches[-1])
    exp_id = re.findall(r"^EXP_ID=(.+)$", text, re.MULTILINE)[-1].strip()
    records.append((seed, result["test_accuracy"], result["worst_group"], exp_id))

acc = [x[1] for x in records]
wg = [x[2] for x in records]
lines = [
    "experiment=spawrious_gci_random_official_rng",
    "dataset=Spawrious_GCI",
    "method=Random",
    "selector=random",
    "optimizer=FedAvgM",
    "data_seeds=42,43,44",
    "seed_role=data_split_only",
    "training_rng=official_unseeded_behavior",
    "flower_rounds=200",
    "model_updates=200",
    "",
]
for seed, accuracy, worst_group, exp_id in records:
    lines.append(
        f"seed={seed}, test_accuracy={accuracy:.8f}, "
        f"worst_group={worst_group:.8f}, flower_rounds=200, "
        f"model_updates=200, checkpoint=checkpoints/{exp_id}/final"
    )
lines += [
    "",
    f"test_accuracy_mean={statistics.mean(acc):.8f}",
    f"test_accuracy_std_population={statistics.pstdev(acc):.8f}",
    f"test_accuracy_std_sample={statistics.stdev(acc):.8f}",
    f"worst_group_mean={statistics.mean(wg):.8f}",
    f"worst_group_std_population={statistics.pstdev(wg):.8f}",
    f"worst_group_std_sample={statistics.stdev(wg):.8f}",
    "paper_worst_group=87.59000000",
    f"worst_group_mean_minus_paper={statistics.mean(wg)-87.59:.8f}",
]
out = log_dir / "spawrious_gci_random_official_rng_summary.txt"
out.write_text("\n".join(lines) + "\n")
print(out.read_text())
PY
```

## 8. FedDiverse 三种子正式运行

Random 三次全部成功并完成汇总后，再运行 FedDiverse：

```bash
tmux new -s spawrious-gci-feddiverse
```

在 tmux 中执行：

```bash
cd /root/SpuriousFL
conda activate spurious
mkdir -p logs/spawrious_gci/predicted_official_rng
set -o pipefail

for data_seed in 42 43 44; do
  log="logs/spawrious_gci/predicted_official_rng/feddiverse_seed${data_seed}.log"
  echo "START_TIME=$(date --iso-8601=seconds)" | tee "$log"
  echo "GIT_COMMIT=$(git rev-parse HEAD)" | tee -a "$log"

  python -u flower_train.py \
    --config-name spawrious_gci_feddiverse_paper \
    seed="$data_seed" 2>&1 | tee -a "$log"

  run_status=${PIPESTATUS[0]}
  exp_id=$(grep -oE 'checkpoints/[0-9]{8}-[0-9]{6}/final' "$log" \
    | tail -n 1 | cut -d/ -f2)
  echo "DATA_SEED=$data_seed" | tee -a "$log"
  echo "EXP_ID=$exp_id" | tee -a "$log"
  echo "END_TIME=$(date --iso-8601=seconds)" | tee -a "$log"
  echo "EXIT_CODE=$run_status" | tee -a "$log"

  test "$run_status" -eq 0 || break
  test -s "checkpoints/$exp_id/final/torchmodel.pt" || break
  test -s "checkpoints/$exp_id/client_info.json" || break
  test -s "checkpoints/$exp_id/client_weights.csv" || break
done
```

## 9. 汇总 FedDiverse

```bash
python - <<'PY'
import ast
import re
import statistics
from pathlib import Path

random_summary = Path(
    "logs/spawrious_gci/random_official_rng/"
    "spawrious_gci_random_official_rng_summary.txt"
).read_text()
random_mean = float(re.search(
    r"^worst_group_mean=([0-9.]+)$", random_summary, re.MULTILINE
).group(1))

log_dir = Path("logs/spawrious_gci/predicted_official_rng")
records = []

for seed in (42, 43, 44):
    text = (log_dir / f"feddiverse_seed{seed}.log").read_text(errors="replace")
    if not re.search(r"^EXIT_CODE=0$", text, re.MULTILINE):
        raise RuntimeError(f"seed {seed} did not finish successfully")
    if "DHT collection round 2 completed without a model update" not in text:
        raise RuntimeError(f"seed {seed} has no verified DHT-only round")
    matches = re.findall(r"aggregated eval results (\{.*?\})", text)
    if not matches:
        raise RuntimeError(f"seed {seed} has no evaluation result")
    result = ast.literal_eval(matches[-1])
    exp_id = re.findall(r"^EXP_ID=(.+)$", text, re.MULTILINE)[-1].strip()
    records.append((seed, result["test_accuracy"], result["worst_group"], exp_id))

acc = [x[1] for x in records]
wg = [x[2] for x in records]
lines = [
    "experiment=spawrious_gci_feddiverse_paper_predicted_official_rng",
    "dataset=Spawrious_GCI",
    "method=FedDiverse",
    "dht=predicted",
    "selector=triplets_stochasticmatrix",
    "optimizer=FedAvgM",
    "data_seeds=42,43,44",
    "seed_role=data_split_only",
    "training_rng=official_unseeded_behavior",
    "flower_rounds=201",
    "model_updates=200",
    "",
]
for seed, accuracy, worst_group, exp_id in records:
    lines.append(
        f"seed={seed}, test_accuracy={accuracy:.8f}, "
        f"worst_group={worst_group:.8f}, flower_rounds=201, "
        f"model_updates=200, checkpoint=checkpoints/{exp_id}/final"
    )
lines += [
    "",
    f"test_accuracy_mean={statistics.mean(acc):.8f}",
    f"test_accuracy_std_population={statistics.pstdev(acc):.8f}",
    f"test_accuracy_std_sample={statistics.stdev(acc):.8f}",
    f"worst_group_mean={statistics.mean(wg):.8f}",
    f"worst_group_std_population={statistics.pstdev(wg):.8f}",
    f"worst_group_std_sample={statistics.stdev(wg):.8f}",
    "paper_worst_group=89.91000000",
    f"worst_group_mean_minus_paper={statistics.mean(wg)-89.91:.8f}",
    f"reproduced_random_worst_group_mean={random_mean:.8f}",
    f"worst_group_mean_minus_reproduced_random={statistics.mean(wg)-random_mean:.8f}",
]
out = log_dir / "spawrious_gci_feddiverse_predicted_official_rng_summary.txt"
out.write_text("\n".join(lines) + "\n")
print(out.read_text())
PY
```

## 10. 完整性检查与结论

```bash
grep -H '^EXIT_CODE=' \
  logs/spawrious_gci/random_official_rng/*.log \
  logs/spawrious_gci/predicted_official_rng/*.log

grep -inE 'traceback|exception|out of memory|cuda error|\bnan\b' \
  logs/spawrious_gci/random_official_rng/*.log \
  logs/spawrious_gci/predicted_official_rng/*.log

grep '^worst_group_mean=' \
  logs/spawrious_gci/random_official_rng/*summary.txt \
  logs/spawrious_gci/predicted_official_rng/*summary.txt

grep '^worst_group_mean_minus_reproduced_random=' \
  logs/spawrious_gci/predicted_official_rng/*summary.txt
```

判定顺序：

1. 六次实验必须全部成功，并存在最终模型；
2. 核对最终轮指标，不使用中间最佳轮；
3. 比较两种方法各自与论文值的差异；
4. 比较 FedDiverse 与本次 Random 的均值差异；
5. 如 GCI 恢复正向提升，下一步复现 Spawrious GAI；
6. 如 GCI 再次没有提升，暂停扩大正式实验，转入共享初始化和完整 RNG 的
   受控诊断。
