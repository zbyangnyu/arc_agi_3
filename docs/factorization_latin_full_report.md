# Factorization Latin 全量实验报告

## 结论先行

这轮结果支持一个有限但清楚的结论：**在当前给定三条因果轴、离散 codebook、canonical palette role 与冻结 simulator/executor 的受控问题里，`3×4` factorized hypothesis head 比近参数量的 unstructured `64`-code head 更容易跨未见 factor-pair context 恢复四个兼容规则。**

主实验的 12 个 fold/seed 配对中，factorized 模型在 held-out `Coverage@4` 上 **12/12 胜出**；均值为 `75.00%`，rank-5 unstructured 对照为 `6.42%`。但 factorized 模型只有 **2/12** 次通过预注册的四指标 `≥ 90%` 静态 gate，因此结果不能写成“已经稳定学会规则抽象”，更不能外推为 ARC-AGI-3 上的自主规则发现。

增加 unstructured head 容量没有改变这个现象：rank-9 和 direct-linear 共 12 个补充 run 的平均 held-out coverage 仍只有 `7.55%` 和 `7.29%`。另一方面，修正后的 symbolic-code interchange v2 在固定 canonical 几何上能把给定 code 执行到 `100%` exact，却在随机合法几何上只剩 `28.32%` exact。这把目前最值得优先消除的混淆定位到了 **geometry generalization / mechanism composition**，而不是继续扩大旧固定模板上的训练量或立即加入 TTT。

> **统计口径：**以下 fold 与 seed 共享同一个 48-context universe 和数据生成器，是相关的重复观测，不是独立样本。所有均值、标准差、胜负数都只作描述；本报告不提供置信区间，也不作显著性检验或总体概率推断。

## 1. 问题与实验设计

主实验比较两种 support-only amortized K4 hypothesis generator：

| 模型 | 输出结构 | trainable parameters | 相对 factorized 差额 |
|---|---:|---:|---:|
| `factorized-3x4` | collision / trigger / relation 三个四值 head | 35,054 | — |
| `unstructured-64` | 64-code low-rank head，rank = 5 | 35,079 | +25（`0.0713%`） |

参数差远低于预注册的 `1%` 容差，12 个配对全部通过参数匹配审计。模型看到 public support，不读取 query target；两者共享训练预算、数据、冻结 executor 和评估逻辑。

| 设计项 | 冻结值 |
|---|---|
| 总 run | `2 models × 4 folds × 3 seeds = 24`；24/24 valid |
| folds | `0, 1, 2, 3` |
| model seeds | `20260727, 20260728, 20260729` |
| Latin holdout | `(observed_value_1 + observed_value_2) % 4 == fold` |
| 每 fold context | train 36 个，held-out 12 个 |
| data master seed | `2026071601` |
| 训练 | 600 steps，batch 8；前 500 步 LR `1e-3`，后 100 步 LR `5e-4` |
| 任务池 | train 144 tasks，eval 48 tasks |
| 执行 | CPU，最多 2 个并行 worker |
| gate | 每个 run 的四项 held-out 指标都必须 `≥ 0.90` |
| suite orchestrator | `../scripts/run_factorization_latin_suite.py`；SHA `6e4735a6271c77d3d22bf42f92aa12e7ec40874b196288484282eb45e9ae3592` |
| training runner | `../scripts/run_expected_discrete_causal_coverage.py`；SHA `a8d1b811fdb9e58f1512e48bb07332c4686a78fee0d61592212af158c12eb63f` |
| frozen executor | `../runs/support_calibrated_executor_seed20260724/checkpoint_last.pt`；SHA `2b2d0e5f3380f6b45fef0b2c0892186fd53dc51a3eb6c4ebc6273ef2895f5254` |

冻结配置见 [`../runs/factorization_latin_4fold_3seed_v1/suite_config.json`](../runs/factorization_latin_4fold_3seed_v1/suite_config.json)，其 SHA256 为 `3d96a6168179208b7b6472b2689bb9116a08ec0a3b6d59173b926c1bc52f1685`。主汇总见 [`../runs/factorization_latin_4fold_3seed_v1/summary.json`](../runs/factorization_latin_4fold_3seed_v1/summary.json)，SHA256 为：

```text
c8c5b9630500039330a0117d3306cb00859ae11618cbe96640471b64a4b80c9d
```

汇总 integrity gate 通过：24/24 run valid、12/12 配对齐全、所有配对参数差小于 1%，且每个 run 的 runner、source 与冻结 executor identity 均通过验证。该 integrity gate 只证明实验完整，**不是**事后定义的性能胜利条件。

## 2. 主实验结果

下表给出跨 12 个相关 fold/seed run 的描述性 `mean ± sample std`：

| held-out 指标 | factorized `3×4` | unstructured rank-5 |
|---|---:|---:|
| factor-tuple `Coverage@4` | **`75.00% ± 18.97%`** | `6.42% ± 3.49%` |
| all-classes-covered task rate | **`49.31% ± 33.80%`** | `0.00% ± 0.00%` |
| support-exact particle rate | **`80.38% ± 15.11%`** | `15.63% ± 8.40%` |
| all-particles-support-exact task rate | **`65.97% ± 24.22%`** | `6.94% ± 8.58%` |
| mean unique factor tuples（上限 4） | **`3.785 ± 0.190`** | `2.188 ± 0.287` |
| 通过四指标静态 gate | **`2/12`** | `0/12` |

`Coverage@4` 的逐配对差值定义为 `factorized − unstructured`：

| 配对统计 | 结果 |
|---|---:|
| 完整配对 | `12/12` |
| factorized 胜 / 平 / 负 | **`12 / 0 / 0`** |
| 平均差值 | **`+68.5764` percentage points** |
| 差值 sample std | `18.6224` percentage points |
| 最小 / 最大差值 | `+45.8333 / +95.8333` percentage points |

因此，当前数据确实排除了“factorized 只因参数更多而赢”这一简单解释，也显示其输出分解能明显抑制 unstructured head 的 tuple collapse。然而，`12/12` 配对胜利回答的是**相对优势**；`2/12` gate pass 回答的是**绝对可靠性**。两者必须同时保留：结构偏置有用，但尚不足以形成稳定的抽象器。

## 3. unstructured 容量敏感性

为检查 rank-5 是否把 unstructured baseline 人为卡得太小，补充实验运行了 8 个 rank-9 run（4 folds × 2 seeds）和 4 个 direct-linear run（4 folds × 1 seed），共 12/12 valid。汇总见 [`../runs/unstructured_capacity_sensitivity_v1/summary.json`](../runs/unstructured_capacity_sensitivity_v1/summary.json)，SHA256 为：

```text
29891a7d6b0c99a1009085afeb1d007c7e63dab6f84cdb9287ee9c7aea2abc8c
```

| head | parameters | 相对 factorized | runs | held-out Coverage@4 | all-classes rate | gate | 同 fold/seed 的 factorized 胜负 |
|---|---:|---:|---:|---:|---:|---:|---:|
| unstructured rank-5（主实验） | 35,079 | `+0.0713%` | 12 | `6.4236%` | `0%` | `0/12` | `12/12` 胜 |
| unstructured rank-9 | 35,467 | `+1.1782%` | 8 | `7.5521%` | `0%` | `0/8` | `8/8` 胜 |
| unstructured direct-linear | 36,642 | `+4.5302%` | 4 | `7.2917%` | `0%` | `0/4` | `4/4` 胜 |

对应的 paired factorized-minus-control coverage 差值，rank-9 为平均 `+69.2708` points，direct-linear 为 `+52.0833` points。容量增加至 direct-linear 并未接近 factorized 表现，所以“rank-5 容量不足”不是这组数据的主要解释。但这些容量 run 仍复用了同一 context universe、模板和 simulator，不能被当成新的独立证据或外部泛化验证。

## 4. 修正后的 symbolic-code interchange v2

[`../runs/factorized_symbolic_interchange_v2/result.json`](../runs/factorized_symbolic_interchange_v2/result.json) 的 SHA256 为：

```text
03d49100c2a5dd960cb8b9853c0c18759a796c937a7409b6a0a1c45aaed9fa03
```

这项实验做的是 **post-argmax 显式整数 factor code 的单轴替换**，不是 learned hidden activation interchange。三条轴、每个 value 的语义、palette canonicalization、codebook 与 ground-truth simulator 都是给定的；patched code 的 target 也由同一 deterministic simulator 构造。因此它测试的是“support 推断出的 privileged code 是否可用，以及冻结 executor 能否执行/替换这个 code”，不测试系统是否自主发现了因果变量。

| 层次 | 指标 | v2 结果 | gate 判断 |
|---|---|---:|---|
| support inference | exact version-space rate | **`49.3056%`** | 未达 `≥90%` |
| support inference | compatible-code recall | **`75.0000%`** | 未达 `≥90%` |
| code 与原 support 的一致性 | source / donor support-compatible | `81.9444% / 80.7870%` | 未达 `100%` |
| canonical code execution | MAP grid exact | **`100.0000%`**（64 个去重 execution cases） | exact 通过 |
| canonical interchange | all diagnostics MAP exact | **`100.0000%`**（1,728 interchanges） | exact 通过 |
| canonical NLL | mean / worst diagnostic mean-cell NLL | `0.0000973 / 0.0007262` | 低于 `0.05` |
| randomized geometry | MAP grid exact | **`28.3203%`**（512 cases） | 未达 `100%` |
| randomized geometry NLL | mean / max-case mean-cell NLL | `0.0823141 / 0.4855730` | 未达 `0.05` |

固定 canonical 输入上的 `100%` 与约 `82%` 的 support compatibility 并不矛盾：前者是在明确给定 patched symbolic code 后测 executor；后者问该 source/donor code 是否真的解释了产生它的 support。模型可以正确执行一个 code，同时从 support 中选错 code。

随机几何审计为每个 executor 使用 8 个非 canonical 合法几何，覆盖全部 64 个 program code，共 512 个唯一 execution cases。exact 从 `100%` 降到 `28.32%`、mean NLL 超过阈值，说明 canonical execution 更像固定模板上的 code-conditioned lookup/插值证据，尚不是几何变化下稳定的局部机制执行。

v2 的 overall gate 因此为 `false`。它给出的正面证据是“显式 factor code 具有可执行、可单轴替换的接口”；它没有证明 amortizer 稳定恢复了正确 version space，也没有证明 learned latent 已形成自主因果表征。

## 5. 当前证据的主要限制

1. **固定几何。**旧 executor 与 hypothesis 实验主要围绕 canonical RuleGrid 布局；随机几何结果已经表明这一分布外变化不是小扰动。
2. **事实上的重复 fixtures。**旧 held-out triple indices `21..23` 的 canonical public input 冗余，v2 去重后只有 1 个 unique public diagnostic input。重复测量不能冒充新的几何或新的机制组合。
3. **data/model seed 不会自动产生新语义。**改变 seed 能改变 minibatch、初始化和任务排列，却仍从同一 48-context universe、同一模板族和同一 simulator 取样；它增加优化重复，不增加规则轴、机制语义或真正 nuisance support。
4. **privileged axes 与 codebook。**collision / trigger / relation 三轴、四值语义、64-code 映射和 palette roles 已提供。当前结果不能说明这些变量能从 raw pixels 自主涌现。
5. **privileged simulator。**训练 teacher、symbolic target 与 interchange ground truth 使用给定 simulator，且不是独立实现。结果证明内部受控一致性，不是对未知环境规律的外部验证。
6. **相关统计。**四个 Latin folds 共享 universe；多个 seeds 共享数据生成过程。`12/12` 是这套冻结重复上的描述，不是 12 个独立环境的置信保证。
7. **尚未测试在线适应。**这轮不含 TTT，也不含 learned hidden-state intervention。现在加入 TTT 会把固定模板记忆、support inference 和几何适应混在一起，难以定位收益来源。

## 6. 已冻结的随机几何 protocol

为隔离 geometry composition，仓库已生成并审计 [`../runs/random_geometry_protocol_v1/manifest.json`](../runs/random_geometry_protocol_v1/manifest.json)，SHA256 为：

```text
43bea3d1242271e9d4ae10161afbbdc43da02f7fc03da2fa9ce5e8439bb466a0
```

详细协议见 [`random_geometry_executor_protocol.md`](random_geometry_executor_protocol.md)。manifest 的 overall protocol gate 已通过，但 `training_started=false`、`checkpoint_written=false`；它是下一轮训练的数据契约，不是训练结果。

| split | geometry seed domain | panels | examples | target scope |
|---|---:|---:|---:|---|
| train | `100000..100063` | 384（192 singleton + 192 pair） | 24,576 | singleton 与 pair，全部 64 codes |
| eval | `200000..200031` | 32 triple | 2,048 | triple-only，全部 64 codes |

协议保证：

- train 384 个、eval 32 个 geometry hash 各自唯一，且交集为空；
- collision/relation 在两个 split 都覆盖 `N/S/E/W` 与 single-cell / perpendicular-domino 两种形状；trigger anchor 同样随机化；
- 每个 panel 的局部机制 write envelope 两两不相交，action atom 顺序随机；
- 加入 0–4 个不会改变规则的 inert distractors；
- 每个训练 panel 的 selected mechanism values 都可区分，且每个 panel 物化全部 64 codes；
- 模型输入只有 `state`、public `action` 和刻意 privileged 的 `factor_code`；split、seed、hash、axis name、task/probe ID 均不暴露。

它仍给出 factor code 和 canonical palette role，所以首先只回答：**给定正确机制变量，结构化 executor 能否从随机 singleton/pair 学到可组合的局部动力学，并在从未见过的随机 triple 几何上执行？**

## 7. 下一步训练实验

首要实验应直接消费已经审计的 `RandomGeometryDataset.iter_examples`，而不是在旧固定 fixtures 上再加训练步数：

1. **冻结数据与 provenance。**优化前把 manifest SHA、train/eval seed ranges、runner/source SHA 写入新 checkpoint；训练脚本只打开 train iterator，eval iterator 在 checkpoint 冻结后才可访问。
2. **做参数匹配的二模型比较。**主模型使用三轴 factor code 的可组合/local mechanism executor；对照使用 64-code unstructured embedding/head，并把总参数差控制在 1% 内。建议先用 3 个 model seeds，所有 seed 共用同一冻结 protocol。
3. **只在 singleton + pair 上训练。**不混入 canonical triple，也不把 eval triple 用于 early stopping、超参数选择或 TTT。
4. **在 triple-only 随机几何上一次性评估。**主指标为 2,048 cases 的 MAP grid exact；同时报告 proper mean-cell NLL、worst-case NLL、按 geometry/axis/value 的分层结果和逐 seed gate，而不只报告 pooled average。
5. **沿用严格 executor ceiling。**目标仍为 `100%` MAP exact 且 max-case mean-cell NLL `≤0.05`。若 factorized 只在平均值上赢、却不能跨 seed 稳定过 gate，结论仍应是“有 inductive-bias advantage，但 executor ceiling 未建立”。
6. **预注册 shuffled/code controls。**至少加入 shuffled factor-code、去掉一个 axis 的 pair coverage、以及 distractor 数量/形状分层，确认模型使用机制变量而不是 geometry-frequency shortcut。

只有该随机几何 executor ceiling 稳定通过后，才进入下一层：不给真实 factor code，由 support-conditioned K4/energy model 恢复 version space，再在随机 triple 上执行。JEPA/AdaJEPA 风格的 latent prediction、GRAM/LoopWM 风格的循环世界模型可以在这一层作为 hypothesis scorer/executor；TTT 最适合最后作为**仅用 public transition likelihood 更新 latent hypothesis/belief**的消融，并与“更新模型权重”分开。这样每次失败都能定位到 geometry execution、rule inference 或 online adaptation 中的一个环节。

## 8. 复现与验证状态

| artifact / check | 状态 | SHA / 结果 |
|---|---|---|
| 24-run 主汇总 | complete，24/24 valid | `c8c5b9630500039330a0117d3306cb00859ae11618cbe96640471b64a4b80c9d` |
| 12-run 容量敏感性汇总 | complete，12/12 valid | `29891a7d6b0c99a1009085afeb1d007c7e63dab6f84cdb9287ee9c7aea2abc8c` |
| symbolic interchange v2 | complete，overall gate false | `03d49100c2a5dd960cb8b9853c0c18759a796c937a7409b6a0a1c45aaed9fa03` |
| random geometry manifest | protocol gate passed，未训练 | `43bea3d1242271e9d4ae10161afbbdc43da02f7fc03da2fa9ce5e8439bb466a0` |
| unit tests | **129 tests，全部通过** | `Ran 129 tests ... OK` |
| Stage0 frozen verification | Python `3.12.3`，source manifest verified | config `d23e21701de8e97b12dca843c1d33014ac37b8b8ea4bbea33eea9fc5a8e4db0a`；result `77dc5eaded9b3d98b03dddd79cf731e00e4d67076a1a42e12b1e155304fc7c7d` |

本轮复核命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/yangzhenbang/anaconda3/bin/python3 \
  -m unittest discover -s tests -p 'test_*.py'

PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3 scripts/verify_stage0a.py
```

## 最终判断

当前最合理的判断不是“factorized JEPA 已经解决规则抽象”，也不是“只要一个大 policy 就够了”。证据更接近：**显式多猜想与 factorized latent code 是有用的归纳偏置，但旧任务的几何冗余让抽象和记忆仍无法区分。**下一项能最大幅度减少不确定性的实验，是已经准备好的 random-geometry singleton/pair → triple executor 训练。它若失败，应先修 mechanism-local execution；它若稳定通过，再把 support inference、JEPA-style latent scoring 与 TTT 逐层接回，而不是一次性堆入同一系统。
