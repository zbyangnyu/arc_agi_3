# Latent-rule ceiling：从正确规则执行到多猜想抽象

> 日期：2026-07-21 / 2026-07-22  
> 结论：executor ceiling 已通过；连续 Persistent-K4 相对 tied-K1 有明显收益，但静态 Coverage@4 只有 `43.75%`，未通过 `90%` gate。当前不进入主动探索或完整 GRAM/LoopWM 实验。

## 问题拆分

端到端失败至少可能来自三个不同位置：

1. 颜色在一个任务中的语义角色无法辨识；
2. 已知正确规则时，transition executor 仍不会执行；
3. executor 可用，但 support encoder 无法形成多个相干规则猜想。

本轮按这个顺序逐层隔离。所有结果都是诊断 ceiling，不是 public-input ARC agent 成绩。

## 1. Palette 可辨识性

RuleGrid 每个 task 随机打乱颜色角色。默认 trigger query 中不展示 `payload_p1` 和 `payload_p2`，但 `TOGGLE` 与 `RECOLOR` 分别需要输出它们。若模型只收到 query state/action 与正确 rule factor，它仍无法知道这两个任务特定颜色。

因此 raw-palette executor 的失败不能解释为架构失败。500-step raw run 的 held-out triple 仅有：

| 指标 | raw palette |
|---|---:|
| exact grid / task | 4.6875% |
| changed-cell accuracy | 55.56% |
| NLL / cell | 0.14414 |

artifact：[`runs/executor_ceiling_smoke_seed20260721/result.json`](../runs/executor_ceiling_smoke_seed20260721/result.json)。

后续 ceiling 使用显式标记的 oracle role canonicalization：读取与 program 独立的 palette，把 actor / blocker / payload 等角色映射到固定 `1..12`。这移除了颜色绑定问题，但违反当前 public-input contract。

## 2. Rule-executor ceiling

模型输入三个独立 factor ID（collision / trigger / relation），每个 factor 单独 embedding 后相加并映射到一个连续 rule latent；没有 64-way program lookup。训练只读取 single 与 pair target（diagnostic `0..20`），triple `21..23` 只在不同 split 上评估。

比较两个 action encoder：

- pooled：沿用原模型，对 composite action atoms 做 mean pooling；
- spatial：把每个 atom 的 kind/direction embedding scatter 到对应 `(row, col)`，再与 state feature 融合。

500 step、4,000 tasks 的结果：

| executor | single exact | pair exact | held-out triple exact | triple changed-cell | triple NLL/cell |
|---|---:|---:|---:|---:|---:|
| pooled | 100% | 100% | 100% | 100% | 0.0002233 |
| spatial | 100% | 100% | 100% | 100% | 0.0001749 |

artifacts：

- [`runs/executor_ceiling_pooled_canonical_seed20260721/result.json`](../runs/executor_ceiling_pooled_canonical_seed20260721/result.json)
- [`runs/executor_ceiling_spatial_canonical_seed20260721/result.json`](../runs/executor_ceiling_spatial_canonical_seed20260721/result.json)

所以在这个受控 panel 上，decoder capacity 不是剩余瓶颈；spatial binding 的 NLL 略好，但 pooled 已足够达到 exact ceiling。

## 3. 无标签 latent hypothesis set

接着冻结通过验证的 pooled executor，只训练从六条 public support transition 产生 latent rules 的部分。训练输入不包含 program、factor label、task/probe ID 或某个真实 program 的 query target。

监督来自 support 推导的无序行为集合：

1. 用 observed support 得到四个 compatible programs；
2. 在训练 query panel 上模拟四个完整 behavior signatures；
3. 模型产生四个 query-independent continuous latents；
4. 在完整 panel 上枚举 `4! = 24` 个双射，禁止逐 query 换 mode，也禁止多个 slot 匹配同一类；
5. changed / unchanged cell 使用平衡辅助 NLL，正式 coverage 仍要求 proper NLL 与 coherent MAP exact。

对照是 `tied-k1`：参数与 K4 基本相同，但四个 tensor-interface slot 从同一个 latent 开始，并在共享 deterministic updater 下始终完全相同。它是一个 effective-K=1 control。

1000 step、8,000 tasks 后，在新 split 的 192 tasks、768 个 held-out triple behavior classes 上：

| 指标 | Persistent-K4 | tied-K1 |
|---|---:|---:|
| mass-weighted Coverage | **43.75%** | 3.125% |
| covered classes | **336 / 768** | 24 / 768 |
| all-four-covered task rate | **10.42%** | 0% |
| mean covered classes / task | **1.75** | 0.125 |
| mean unique MAP signatures | **3.83** | 1.00 |
| mean pairwise latent distance | **3.77** | 0.00 |

artifacts：

- [`runs/canonical_latent_k4_seed20260722/result.json`](../runs/canonical_latent_k4_seed20260722/result.json)
- [`runs/canonical_latent_tied_k1_seed20260722/result.json`](../runs/canonical_latent_tied_k1_seed20260722/result.json)

这证明多 latent slots 不只是参数冗余：它们确实分化为接近四个不同行为预测，并将 coverage 提高约 14 倍。但 `43.75%` 仍远低于静态 gate 的 `90%`；它不是“规则抽象已经解决”的证据。

## 与 JEPA、AdaJEPA、GRAM、LoopWM 的关系

- JEPA 的价值不是自动带来泛化，而是允许 predictor 只预测任务相关 latent，主动丢弃像素级不可预测细节。能否泛化取决于 latent target 是否保留因果规则、是否去掉 nuisance，以及 predictor 是否被 action 条件约束。本轮 palette 结果正说明：错误的不变性或缺失 binding 会让 latent prediction 也无解。
- [AdaJEPA](https://arxiv.org/abs/2606.32026) 的核心映射是执行试探、观察真实 transition、用自监督误差在闭环中适配、再规划。这里应对应 active prefix `7..10` 后的 belief/model update，但静态 K4 尚未过 gate，暂时不能把 test-time adaptation 与 hypothesis quality 混在一起。
- [GRAM](https://arxiv.org/abs/2605.19376) 把 recursive reasoning 变成随机多轨迹 latent computation。当前 K4 是它的确定性、固定四轨迹最小近似；结果支持“并行猜想有用”，但还未测试随机 branching、variational training 或 inference-time sampling。
- [LoopWM](https://arxiv.org/abs/2606.18208) 提供共享参数的迭代深度轴。合理的下一对照是固定 executor 后比较 tied refinement `J=1/2/4`，而不是立刻扩大模型宽度。LoopWM 并非 Mengye Ren 的论文；GRAM 与 AdaJEPA 才包含 Mengye Ren。

## 尚未解决的 benchmark 问题

这些问题使当前结果只能称为 representation/optimization gate：

- oracle palette canonicalization 不是公开输入；public-compliant 版本需要 palette legend 或可见的输出色 reference；
- `D12..14`、`D15..17`、`D18..20` 各自是重复输入，`D21..23` 也重复，因此当前 three-frame triple 并非三个独立几何 probe；
- `task_id` 中含 `Pxx/Hx`，active `probe_id` 与 strong/partial/neutral 类型固定对应；当前 tensor pipeline 未编码字符串，但未来 controller 必须继续排除；
- 两个 partial candidates 产生同一分区，而 strong candidate 一步即可完全辨识，现有 active bank 不足以检验多轮猜想—证伪循环；
- geometry nuisance 当前变化有限，canonicalization 后的 train/test 语义多样性不高。

## 决策

1. 不把 `43.75%` 当作静态 gate 通过，也不进入 learned active policy。
2. 先做两个独立修订：公开可辨识 palette/legend 与真正不同的 pair/triple fixtures。
3. 在固定 executor 下比较连续 K4 与受限 factor/codebook K4；这会检验失败来自 latent 搜索几何还是 recurrent inference。
4. 若连续 K4 仍不足，再比较 LoopWM 式共享 updater 深度 `J=1/2/4`；只有静态 coverage 接近 90% 后，才接 AdaJEPA 式 trial adaptation。
5. 真正的主动 benchmark 要移除 universal strong probe，加入 history-dependent complementary partitions，再测试 GRAM 式多轨迹的淘汰、保留与重采样。

## 复现与回归

新增入口：

```bash
/Users/yangzhenbang/anaconda3/bin/python3 scripts/run_rulegrid_executor_ceiling.py \
  --output runs/executor_ceiling_pooled_canonical_seed20260721 \
  --steps 500 --executor pooled --palette-input oracle-canonical

/Users/yangzhenbang/anaconda3/bin/python3 scripts/run_canonical_latent_coverage.py \
  --output runs/canonical_latent_k4_seed20260722 \
  --executor-checkpoint runs/executor_ceiling_pooled_canonical_seed20260721/checkpoint_last.pt \
  --model persistent-k4 --steps 1000
```

完整测试为 `73 / 73` 通过；Stage 0-A reference 的 result SHA256 仍为 `77dc5eaded9b3d98b03dddd79cf731e00e4d67076a1a42e12b1e155304fc7c7d`，source manifest 验证通过。
