# RuleGrid RL-from-scratch smoke report

> 日期：2026-07-16  
> 状态：机制 smoke test，非预注册正式结果

## 问题

测试一个从随机初始化开始的 policy，是否能在不使用 reconstruction loss、规则标签、
version space、oracle EIG 或 diagnostic target 的条件下，仅通过稀疏的“行为已辨识”
成功奖励学会选择 RuleGrid probe。

## 方法

- 输入：公开 `8×8` state/next-state、公开动作、完整已观察历史、8 个候选 probe；
- 模型：共享 CNN grid encoder、action embedding、GRU history encoder、actor/critic heads；
- 算法：Monte-Carlo actor-critic；
- reward：辨识成功时 `1`，其他时刻 `0`；
- 动作无放回，budget `4`；
- 训练：100 updates、batch size 32，共 3,200 tasks；
- 验证：独立 `rl-validation` split 的 192 tasks。

完整 artifact：[summary.json](../runs/rl_scratch_smoke_ablation_seed20260716/summary.json)。

## 结果

| 方法 | success rate | mean actions |
|---|---:|---:|
| uniform without replacement | 82.81% | 2.3646 |
| learned greedy policy | 100.00% | 1.0000 |
| learned policy，删除两条 calibration history | 100.00% | 1.0000 |

## 解释

RL 很快找到了高回报策略，但 calibration ablation 完全不影响结果。这表明它不需要
利用 support history 推断 held-out rule；它可以仅从候选 probe 的可见结构识别出
`1+1+1+1` strong probe 模板。

因此该结果证明的是：

1. 稀疏奖励 actor-critic 链路可以工作；
2. 当前 active-bank 允许一个无需 belief update 的可见 probe-quality shortcut；
3. 高 RL 回报不能作为形成对象表示、世界模型、多假设 belief 或主动证伪能力的证据。

## 下一步门槛

在继续扩大 RL 训练前，必须重做一个 policy benchmark，使候选 probe 的信息价值不能
从单个候选画面独立判断。至少应满足：

- 同一个公开 probe 在不同 support histories 下可以是 informative 或 uninformative；
- 候选的单帧可见统计与 EIG 严格配平；
- 正确动作必须依赖历史产生的 belief；
- 增加 history permutation / calibration removal / cross-task history swap ablations；
- 测试留出新的规则组合，而不只是新的 palette 和 candidate permutation。

只有 learned policy 显著超过 candidate-only ablation，才能声称 RL 学到了历史条件化的
探索策略。
