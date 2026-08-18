# JEPA × GRAM × ARC-AGI-3：当前思路与结果

## 一句话思路

让 GRAM 式模块维护多个 latent 规则猜想，让 JEPA/world model 预测干预结果，再按
query 价值主动尝试并显式 Bayes 更新，即“猜规则 → 试验 → 排除 → 再试”，而不是直接
学习一个固定 policy。

## 当前实验

先在 RuleGrid 隔离 world-model likelihood 是否具有因果局部性。隐藏规则有
collision / trigger / relation 三个四值因子，共 64 个 codes。每次先强制执行一个
query-axis probe，再从同一 posterior 独立执行四个 cross-axis 和两个 neutral
probe；无 selector。

比较五条路径：`EE`（全 exact）、`RR`（全 raw learned）、`RP`（只投影 forced
likelihood）、`PR`（只投影 query likelihood）、`PP`（两阶段都做 oracle
axis projection）。projection 在 proper full-grid log-likelihood 上对无关因子做
`logmeanexp`。

主分析每条路径含 2,304 个唯一语义序列，其中 cross-axis 为 1,536、neutral 为
768。64 个 palettes 在 oracle canonicalization 后只是顺序重复，不作为独立样本。

## 结果

| 路径（cross-axis） | 高置信反转率 | P99 odds drop | 最大 drop |
|---|---:|---:|---:|
| EE | 0 | 0 | 0 |
| RR | **2.083%** | **35.712 nats** | 45.563 |
| RP | 0 | 3.578 | 4.413 |
| PR | **2.083%** | **34.634** | 43.215 |
| PP | 0 | 0.0000019 | 0.0000019 |

query positive control 为 100% top-1，`p(true)≥0.95` 为 91.67%。32 次 RR
高置信反转全部来自 `relation query → collision v1 probe`；投影 query 但保留 raw
forced likelihood（PR）几乎不能改善，说明主要问题是 forced likelihood 直接跨轴
泄漏，而非只由 query posterior 的相关性造成。cross-probe outcome-partition F1 为
0.778，最弱的是 collision v1（0.355）和 trigger v1（0.559）。

预注册 gate 判为 `support-factorized-jepa-executor`。batch 16/64 及第二
split/seed 的 summaries、gate 和去除候选索引后的 11,520 条路径记录逐字段完全一致；
exact controls、计数、canonical invariance 和 JSON 有限性均通过。

## 新进展：探索是否真的在获取有效信息

two-axis compositional query 已完成。当前 query-conditioned selector 在 exact
control 上得到严格的 `0.25 → 0.50 → 1.00`，前两步 100% 选择两个互补相关轴；
global-EIG 虽然先取得更多全局信息，却只有 `0.25 → 0.25 → 0.50 → 1.00`。

冻结 learned executor 后，B2 的 exact/exact、exact/learned、learned/exact 和
learned/learned accuracy 分别为 **100%、93.9%、83.7%、82.6%**。这把当前首要
探索瓶颈定位为 learned MAP outcome partition，而不是 query-value utility。
完整结果见[主动探索信息审计](exploration_information_audit.md)。

## 结论与下一步

这支持下一版 executor/JEPA 显式约束“一个 probe 只更新它作用的 rule factor”，例如
factorized likelihood heads、cross-axis invariance regularizer，再用 GRAM/TTT 在
episode 内快速校准。`PP` 是 oracle 因果干预，其零漂移在数学上被保证，因此它证明
了故障位置与可修方向，**不等于一个已学会的架构已经成功**。

当前仍使用已知三因子 codebook、oracle palette canonicalization、固定 geometry 和
预制 probe bank，不是 ARC-AGI-3 成绩。下一项应同时检验：

1. 用完整 predictive distribution / ensemble disagreement 代替 learned MAP
   partition 来估计 query information；
2. locality continuation 加 teacher distillation 或 parameter trust region，避免
   新 geometry 训练破坏旧 active-prefix gate；
3. 真正的 2+2 partial probe、early stop 和 geometry OOD；
4. 最终去掉预制 probe bank、显式 axes/codebook 与 oracle canonicalization。

详细协议见[10 小时实验计划](next_10h_experiment_plan.md)，正式结果见
[result.json](../runs/forced_cross_axis_likelihood_audit_g64_b16_seed2026072401/result.json)
和[分析报告](../runs/forced_cross_axis_likelihood_audit_g64_b16_seed2026072401/analysis.md)。
