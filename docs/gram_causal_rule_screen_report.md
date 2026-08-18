# GRAM causal-rule screen：随机递归假设是否优于固定 K=4

## 结论

在当前受控 RuleGrid version-space 任务上，**原始 GRAM 式随机递归假设器不适合直接替换固定四槽 K4**。

同一 context fold、同一 seed、同样 600 steps 下：

| support-only inference | held-out Coverage@4 | 四规则全部覆盖 | valid particle rate | 平均不同 tuple |
|---|---:|---:|---:|---:|
| 固定四槽 factorized K4 | 91.67% | 75.00% | 93.75% | 3.92 |
| GRAM，4 条随机 trajectory | 13.54% | 0.00% | 18.75% | 3.40 |

GRAM 确实生成了多个不同候选，但约 81% 的 trajectory 不满足公开 support。因而这里的瓶颈不是随机宽度或候选多样性，而是 **prior 没有学会把随机分支限制在证据兼容的 version space 内**。扩大 width=32 只把 compatible-rule recall 提高到 28.13%，仍没有一个 task 收集齐四条规则。

这是一项有价值的负结果。当时的直接 follow-up 猜想是把 GRAM 当作 proposal network，放进由公开 support verifier 驱动的筛选循环。后续审计发现，没有 proposal correction 和合法 MH kernel 时不能把该循环称为 SMC；实际实现改成了路径无关的 verifier-guided population search。实验已经完成，结论是外置 verifier 与 carry 有效，但 GRAM proposal 明显弱于 matched uniform control。详见 [顺序证据同化报告](gram_sequential_assimilation_report.md)。

## 实现了什么

本地实现依据 [GRAM 论文](https://arxiv.org/abs/2605.19376) 的方法结构，而不是官方代码；截至本次实验时，[项目页](https://ahn-ml.github.io/gram-website/) 仍将代码标为 coming soon。

[`prp_wm/gram_causal_rules.py`](../prp_wm/gram_causal_rules.py) 实现了：

- deterministic low-level refinement；
- deterministic high-level proposal；
- Gaussian stochastic guidance residual；
- 训练时读取无序 behavior set 的 posterior `q`，以及推理时只读取 public support 的 prior `p`；
- diagonal-Gaussian analytic KL、`0.8` KL balance 与 warmup；
- 每层 recursive step 的 deep supervision，以及 step 间 truncated gradient；
- 共享递归参数，并允许 inference depth / width 扩展；
- 不给 trajectory 固定 slot identity，四条训练 trajectory 对应同一个无序行为集合。

[`scripts/run_gram_causal_screen.py`](../scripts/run_gram_causal_screen.py) 复用已审计的 support-calibrated executor、Latin context split、exact 64-code detached costs 与 Coverage@4 evaluator。推理 width `W` 表示一次从 prior 独立采样的总 trajectory 数；候选去重后，只按六条公开 support transition 的 frozen-executor cost 排序。

这个 screen 只测试“能否从 support 摊销出四个兼容 latent 规则”。它仍然给定了三条机制轴、每轴四个值、oracle palette-role canonicalization、privileged executor pretraining，并在训练中使用 simulator 提供的无序 behavior set；所以它不等于从 ARC 原始像素自主发现规则。

## 配对实验

正式 run 使用：

- context fold `0`，model seed `20260728`；
- 600 steps × batch 8 = 4,800 task draws；
- recursive depth `4`，guidance dimension `8`；
- KL weight `0.01`、balance `0.8`、warmup `100` steps；
- tuple-diversity weight `0.1`；
- 最后 100 steps 的 learning rate 为 `5e-4`；
- frozen executor：`runs/support_calibrated_executor_seed20260724/checkpoint_last.pt`。

GRAM checkpoint SHA256 为 `f2504860fe060ed7cc4700cde5f571a6df6f9d453fe391c1a037aec325fabc20`。模型与 runner 的 source SHA256 分别为 `d97a29d5259186f1294e4546abad55496831de0a7c7836d835dfdd98cf923d59` 和 `e47c6d6d0bca2cc631944f01a2489c3b6032a18110f51805d95f562473ff1f70`。

完整 artifact：[`runs/gram_causal_screen600_fold0_seed20260728/result.json`](../runs/gram_causal_screen600_fold0_seed20260728/result.json)。配对固定 K4 artifact：[`runs/factorization_latin_4fold_3seed_v1/runs/factorized-3x4/fold_0/seed_20260728/attempts/001/output/result.json`](../runs/factorization_latin_4fold_3seed_v1/runs/factorized-3x4/fold_0/seed_20260728/attempts/001/output/result.json)。

### 训练信号

target-conditioned posterior `q` 不是完全失效：到 step 600，其平均不同 tuple 达到 `3.375`，set cost 从早期的 `2.633` 降到 `0.246`。但固定 K4 在同一训练点达到 `3.875` 个 tuple 与 `0.0437` set cost。更重要的是，公开推理使用的 support-only prior `p` 仍把绝大部分概率放在无效 tuple 上。这显示了明显的 posterior-to-prior distillation gap。

### width scaling

| 总 trajectory width | 平均不同候选 | compatible-rule recall | raw trajectory valid rate | 四规则全部覆盖 |
|---:|---:|---:|---:|---:|
| 1 | 1.00 | 4.17% | 16.67% | 0.00% |
| 4 | 3.40 | 13.54% | 18.23% | 0.00% |
| 8 | 5.83 | 19.27% | 18.49% | 0.00% |
| 16 | 8.35 | 24.48% | 16.54% | 0.00% |
| 32 | 11.71 | 28.13% | 16.99% | 0.00% |

作为参照，如果 generator 在四条真实兼容规则上均匀独立采样，width=4 的期望 recall 已是 68.36%，收齐四条的概率为 9.375%；width=16 收齐四条的概率约为 96.0%。实际 GRAM 到 width=32 仍为 0%，因此失败不能用普通 coupon-collector 效应解释。

额外只推理、不重新训练的 width `64/128/256` 检查中，recall 分别约为 `36.46%/41.15%/42.19%`，仍然没有 task 收齐四条，说明继续堆 width 已经开始饱和。

### controls 与 gate

- held-out Coverage@4：`13.54%`；seen-context Coverage@4：`17.01%`；
- shuffled-support control：`6.25%`，说明模型不是完全忽略 support，但依赖很弱；
- held-out all-particles-support-exact：`6.25%`；
- 四项 `>=90%` 静态 gate 全部失败。

## 为什么固定 K4 在这里更合适

当前任务的 version space 是一个非常规整的 Cartesian 结构：三个机制轴中两个由 support 确定，剩下一轴有四个可能值。固定四槽 K4 的 inductive bias 恰好对应“系统地覆盖四个离散分支”；它不需要从连续高斯噪声中学习这一组合结构。

GRAM 的优势更可能出现在以下情形：

- version space 大小随 task 改变；
- 候选规则之间存在相关、非 Cartesian 的分支；
- 单次固定 K 无法覆盖所有高概率机制；
- 递归计算可以逐步修正一个假设，而不是一次输出四个已知形态的离散槽。

因此，这个结果没有否定 GRAM 对 ARC-AGI-3 的潜力；它否定的是“仅靠无约束随机 guidance + 最后一次 support 排序，就能替代显式多假设维护”这一具体方案。

## 后续实验：verifier-guided population search 已完成

后续版将抽象过程写成了显式的猜想—证伪循环：

1. `p(z | support)` 产生一批随机 rule proposals；
2. 每个 recursive step 后，用 frozen executor 重放公开 support，得到能量 `E(z; support)`；
3. 合并 fresh proposals 与上一阶段保留的离散 codes，并在评分前去重；
4. 每阶段只按当前完整历史做一次 compatibility-first 排序，不继承 path weight；
5. 保留 exact-compatible tuple；没有 exact 时保留 minimum-error stratum；
6. 用 paired partial→neutral→strong observations 测试更新、稳定性与恢复。

这把因果理解落实为可检验的机制假设：latent 不需要有人工可读坐标，但每个假设必须能生成干预后果，并能被历史 transition 证伪。实现明确不声称 Bayesian posterior 或 SMC。

同一 fold 的三个 inference seeds 显示，W32 learned-verifier search 的 t3 target coverage：GRAM 为 factual `38.19%`、counterfactual `45.83%`，uniform control 为 `79.17%`、`84.72%`。GRAM 对 t2-missing target 的 strong-evidence recovery 仅为 `2.20%`、`0%`。因此当前停止扩大旧 GRAM width；先加入 full-version-space coverage、hard replay 与 stratified/uniform exploration。完整数字、active-prefix verifier gate、proposal blind spots 和下一 gate 见 [新报告](gram_sequential_assimilation_report.md)。

## 复现

```bash
/Users/yangzhenbang/anaconda3/bin/python3 scripts/run_gram_causal_screen.py \
  --output runs/gram_causal_screen600_fold0_seed20260728 \
  --executor-checkpoint runs/support_calibrated_executor_seed20260724/checkpoint_last.pt \
  --context-fold 0 --seed 20260728 \
  --steps 600 --batch-size 8 \
  --recursive-steps 4 --guidance-dim 8 --guidance-mode stochastic \
  --kl-weight 0.01 --kl-balance 0.8 --kl-warmup-steps 100 \
  --diversity-weight 0.1 \
  --tail-steps 100 --tail-learning-rate 5e-4 \
  --inference-widths 1 4 8 16 32 --device cpu
```

新增单元测试覆盖 prior 的 support-only 边界、target-conditioned posterior、解析 KL、排列不变性、随机种子复现、递归参数共享、width scaling、候选去重/排序、frozen executor 与 gate。全仓测试为 `138 passed`；冻结 Stage 0-A manifest 也通过验证。
