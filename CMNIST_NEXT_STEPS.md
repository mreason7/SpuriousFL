# CMNIST 自研后续：先诊断，再固定协议重复

日期：2026-10-30。以下数字来自用户在服务器执行 `726bc86` 汇总脚本的输出；
本地尚未取得服务器 checkpoint 和日志，不能声称已本地核验完整运行。

## 已完成的九次实验

| 数据seed | Random测试WGA | 真实DHT测试WGA | 方向感知测试WGA |
|---|---:|---:|---:|
|42|89.6208|92.6148|93.5748|
|43|94.0918|93.7725|93.5329|
|44|92.5349|93.9252|93.7725|

三组分别采用训练seed142/143/144、验证seed1042/1043/1044。
方向感知平均测试WGA为93.626719，真实DHT为93.437486，Random为92.082502。
方向感知与真实DHT差值仅+0.189233个百分点，逐seed两负一正。
目标误差显著改善，但性能优势未得到稳定验证。保留全部最终结果，不按测试分数调参。

## 第一步：完整性快照与选择频率

在服务器同一分支运行：

```bash
python tools/summarize_cmnist_direction.py
python tools/analyze_cmnist_selection.py
```

新增脚本只读checkpoint，输出 `results/cmnist_direction_formal/selection_analysis/`：

- `selection_summary.csv`：使用过多少客户端、有效客户端数、最常选9人的选择份额、
  不同集合数、相邻轮集合Jaccard。
- `client_frequency.csv`：每个客户端的选择次数及真实联合分布、类别/属性偏向。
- `snapshot_sha256.json` 和 `source_audit.json`：运行文件哈希及本次审计副本。

默认拒绝覆盖已有输出。需要重跑时传入新的 `--output` 路径。
哈希快照不是独立备份；另行复制九个checkpoint目录、训练日志和结果到备份位置。
脚本不证明进程退出码、第2轮无模型更新；这两项仍应检查原始日志。

解释时看：方向组是否只反复选择少数客户端？是否只覆盖某些属性方向？
同样的目标值下是否存在多个组合？Random与DHT是否覆盖更多样本？
集中选择只是机制描述，不能据此直接声称有害、数据质量差或解释性能因果。

## 第二步：固定数据的探索性重复计划

先读完第一步结果，再启动本批，不在同一GPU并行训练。保持算法、目标、预算不变。
固定数据seed42和验证seed1042，新训练seed预先定为242、243、244、245、246。
每个训练seed都运行三种方法，共15次。先前seed142保留为已观察的历史结果，
不混入新增五次的主汇总。选择固定42是沿用既有划分；已看到42收益较大，
因此本批只能评价该固定划分的训练随机性，不能视为无偏选择的新外部验证。

主要对照：方向感知减真实DHT的最终验证WGA差值，报告五次均值、标准差和每次差值；
Random为辅助对照。测试指标仅在预先固定的全部运行完成后统一报告，不据其挑参数。
相同training_seed只是匹配随机源控制，不意味着不同方法经历完全相同训练轨迹。

服务器在tmux中执行以下批次；任意进程失败即停止，不自动重试或跳过。

```bash
conda activate spurious
cd /root/SpuriousFL
bash <<'BASH'
set -euo pipefail
batch_dir="logs/cmnist_direction_fixed42/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$batch_dir"
for training_seed in 242 243 244 245 246; do
  for arm in direction_oracle oracle_dht random; do
    python -u flower_train.py \
      --config-name="research_cmnist_${arm}" \
      seed=42 research.training_seed="$training_seed" research.validation_seed=1042 \
      research.run_tag="cmnist-direction-fixed42-${arm}-train${training_seed}" \
      2>&1 | tee "$batch_dir/${arm}_train${training_seed}.log"
    echo "DONE arm=$arm training_seed=$training_seed"
  done
done
BASH
```

每个训练seed内部三组数据与初始化hash应一致；不同训练seed数据hash保持一致，
初始化hash允许且预期变化。旧summarize脚本只处理原九次formal标签，不会自动汇总本批；
本批须使用独立汇总，不修改旧标签强行混入。

## 第三步及以后

若性能收益仍很小，不继续只为追求更低分布误差调参。优先研究模型需求与覆盖的关系。
方向消融、估计矩阵和动态目标均另立实验；本批不实现、不混入。
局部列排序的消融会人为制造伪坐标，不能当作原DHT的严格单因素替代，需单独设计。
