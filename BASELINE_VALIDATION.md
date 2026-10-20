# 基线修正与单因素验证 v1

本批代码基于 `3674a55`，只修pivot、客户端ID映射和选择器边界；三项分别可切换。
旧配置默认保持legacy行为。步数多1、loss旧归一化、动态DHT刷新等尚未修改，避免把多项行为变化混在一起。

## 对照组

| arm | 配置 | 相对同协议legacy组的唯一方法改动 |
|---|---|---|
| legacy | research_gai_legacy | 无，作为新受控协议的基线 |
| pivot | research_gai_pivot | 按两组预测正确/错误数量差选择pivot |
| mapping | research_gai_mapping | 将返回位置映射到候选客户端ID |
| guarded | research_gai_guarded | 显式排除已选项、退化输入回退、精确选择预算 |
| corrected | research_gai_corrected | 三项合并，不能当单因素结果 |
| random | research_gai_random | 同样的全客户端预训练及200次模型更新，后续随机选择 |

Random没有DHT推断开销，额外开销需单独报告。只通过same update count不能宣称总计算/通信公平。

## 所有组共享的新协议

- Spawrious文件名稳定排序；从训练池按(y,g)留出10%独立验证数据，客户端划分只在剩余训练池中执行。
- 测试集在最终模型确定后评测一次；逐轮框架日志仍叫test_accuracy/worst_group，但其输入已是独立验证集，应按验证指标解释。
- GAI仍为25客户端、4600训练样本、每轮9客户端，1次FedAvg预训练+1次DHT-only+199次选择更新。
- training_seed独立于data_seed；当前runner用data_seed+100作为训练seed，data_seed+1000作为验证划分seed。每轮、每客户端及服务端选择的RNG分流，避免客户端重建重复常量种子。
- 强制确定性算法；如果PyTorch报告某算子不支持确定性，停止排查，不自动改为warn-only。
- 聚合前按客户端ID排序，避免Ray返回顺序改变浮点累加顺序；跨硬件/库版本仍不保证逐位一致。
- 所有组共同采用按数字ID排序的候选顺序，消除异步注册顺序这个混杂。连续ID下mapping-only应是无效应对照，而非预期获得更高准确率。乱序/非连续ID通过单元用例单独验证。
- 保存训练/验证/测试成员清单和digest、初始模型参数digest；不同组若不一致，runner会停止。
- 成员清单的digest验证样本身份/标签和分片，不是每张图片内容的哈希。原始压缩包完整性仍按迁移时SHA-256核验。
- 保存pivot伪属性计数、候选列表及实际选中ID。真实群体信息仍仅用于审计和评测，不用于修正pivot/guarded选择。
- 客户端失败/缺失不继续聚合。所有变化都在research.enabled=true下启用；历史结果不重写。

这是新的受控协议，数字不能直接与旧official-unseeded表格混合。
原GAI测试表现已被观察过，不能把这组结果包装成从未用于开发的独立外部验证。

## 服务器执行顺序

所有命令在 `/root/SpuriousFL`、`spurious` 环境执行。

```bash
conda activate spurious
cd /root/SpuriousFL
python tools/check_baseline_validation.py
```

服务器装有PyTorch，`test_torch_runtime_if_available`必须实际通过，不能跳过。

先验证六套Hydra配置都能加载，不启动训练：

```bash
for arm in legacy pivot mapping guarded corrected random; do
  python flower_train.py --config-name=research_gai_${arm} --cfg job > /tmp/research_gai_${arm}.yaml || break
done
```

再运行四组短测试。每组只有3个事件、局部2步、DHT阶段配置2步（旧语义实际3步）。

```bash
python tools/run_baseline_validation.py --stage smoke --arms legacy,pivot,mapping,guarded --seeds 42
```

runner自动检查退出码、3轮/25客户端参与形状、全客户端前两轮、第三轮9客户端、DHT-only日志、最终模型和元数据、各组相同数据与初始化hash。失败自动停止，不自动补跑挑结果。

短测试成功仅说明运行链路通过，不能据此解释精度或判断论文贡献。

## 正式单因素实验（短测试验证后）

先查看不执行的命令清单：

```bash
python tools/run_baseline_validation.py --stage plan --arms legacy,pivot,mapping,guarded --seeds 42,43,44
```

正式命令（共12次完整训练）为：

```bash
python tools/run_baseline_validation.py --stage formal --arms legacy,pivot,mapping,guarded --seeds 42,43,44
```

在tmux中运行。不要与其他训练并行占用同一GPU。选择器若在legacy组遇到退化输入会停止；这是需记录的失败证据，不应悄悄给legacy组加guarded后仍称同一对照。

这三个data/training seed组合用于第一轮诊断，不是充分隔离数据随机性与训练随机性的交叉实验；后续应固定数据seed跨多个training_seed做补充。runner暂不支持交叉seed网格。

## 产物与解释

每次运行保存在新的 `checkpoints/<时间>/` 下：

- `config.yaml`：实际生效配置与run_tag。
- `data_manifest.json` / `data_manifest.sha256`：共同数据清单。
- `initial_weights.sha256`：共同初始化校验。
- `selection_trace.jsonl`：FedDiverse选择轮的候选ID、返回位置和实际ID。
- `client_info.json` / `client_weights.csv` / `final/torchmodel.pt`。
- `final_validation_metrics.json`：最终模型在独立验证集上的指标。
- `final_test_metrics.json`：固定最终checkpoint测试指标；loss显式标记为旧归一化口径。

批次日志和summary写在 `logs/baseline_validation/<stage>-<时间>/`。
summary主表只列验证指标，避免在开发时依据测试分数不断选方案。

先看pivot是否改善AI/SC估计，再看选择轨迹、群体覆盖和验证WGA。mapping/guarded在正常输入和稳定候选顺序下应保持结果；若不同，先排查额外行为差异。

本批本地检查不含真实GPU训练。服务器必须完成配置加载、真实torch测试和四组短测试后，才进入完整对照。
