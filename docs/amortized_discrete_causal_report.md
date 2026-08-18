# Expected-discrete K4：把显式规则筛选蒸馏成快速抽象器

> 日期：2026-07-26  
> 结论：用 64 个整数机制的 detached counterfactual cost 训练 support encoder 后，amortized K4 在 12 个从未训练过的 observed-factor-pair contexts 上达到 **Coverage@4 = 100%**，并与显式 64-rule filter 一样恢复完整 version space。当前 CPU 实现约快 **23.7×**。这证明显式规则筛选可以被摊销，但仍是给定机制坐标系和 privileged teacher 的 ceiling。

## 为什么不能继续用 straight-through decoder gradient

旧 causal K4 令三个四值 categorical code 做 straight-through hard sampling，再通过冻结 executor 的 embedding、factor mixer 与 decoder 反向传播。forward 位于真实整数 code 顶点，但 backward 使用 frozen decoder 从未训练过的连续插值切向；最终四个 slots 塌缩，Coverage@4 只有 `10.94%`。

显式 64-rule filter 则达到 100%，说明规则空间、executor 与 observation evidence 本身足够。剩余问题是如何把 expensive search 蒸馏为一次 support-conditioned forward。

## Expected-discrete objective

support encoder 输出：

\[
z\in\mathbb{R}^{B\times4\times3\times4}.
\]

三轴 categorical posterior 构成 64 个整数规则上的 factorized joint：

\[
q_{bk}(h)=\prod_{a=1}^{3}
\operatorname{softmax}(z_{bka})_{h_a}.
\]

冻结、support-calibrated executor 在 `no_grad` 下运行全部整数 code，得到完整 behavior-panel cost：

\[
C^{query}_{bhm}\in\mathbb{R}^{B\times64\times4}.
\]

encoder 接收的梯度只来自离散期望：

\[
E_{bkm}=\sum_h q_{bk}(h)C^{query}_{bhm}.
\]

随后在完整 query panel 上一次性枚举 `4!` 个双射：

\[
L_{set}=\min_{\pi\in S_4}\frac14\sum_k E_{bk,\pi(k)}.
\]

另外使用两项受限正则：

- `0.1 ×` calibrated support expected cost，保证每个粒子解释 public evidence；
- `0.1 × -log(1-<q_k,q_l>)`，只排斥完整 rule tuple 重合，不强迫每个 axis 都不同。

assignment temperature 从第一步就是 `0`；factor temperature 固定 `1`；不使用 entropy sharpening。decoder cost 完全 detached，executor 始终 frozen + eval。

## 先移除 48-context lookup 捷径

审计发现 oracle palette canonicalization 后，任意原 train/eval split 都只有：

\[
3\text{ heldout axes}\times4\times4=48
\]

种 permutation-invariant support contexts，而且 train/eval 的 48 个集合完全相同。一个 48-entry hash lookup 即可在旧 composition split 上达到 100%，所以旧 split 不能证明组合抽象。

本轮按每个 heldout axis 的两个 observed factor values 划分：

- training：排除 `(0,0), (1,1), (2,2), (3,3)`，共 36 个 contexts；
- evaluation：只使用这 12 个未见 contexts；
- 每个单独 axis value 在训练中仍出现；
- 训练与评估 canonical support-context overlap 为 0。

evaluation 的 48 tasks 是 12 个 contexts × 4 个真实 heldout values。普通 lookup 在这里没有 key hit。

## 正式结果

先训练 500 steps / 4,000 tasks，再从该 checkpoint 以 `5e-4` 学习率续训 100 steps / 800 tasks；累计 600 steps / 4,800 tasks。

| 指标 | 未见 12 contexts | 已见 36 contexts | shuffled support |
|---|---:|---:|---:|
| Coverage@4 | **100%** | **100%** | 13.02% |
| all-four-covered task | **100%** | **100%** | 10.42% |
| factor-tuple Coverage@4 | **100%** | **100%** | 13.02% |
| all particles support exact | **100%** | **100%** | 12.5% |
| mean unique tuples / signatures | **4.0 / 4.0** | **4.0 / 4.0** | 2.98 / 2.98 |

三个 heldout axes 各有 16 个 unseen-context tasks，全部达到 100%。训练末 joint entropy 为 `0.0495` nats、top-rule probability 为 `0.9933`，没有使用额外 entropy sharpening。

正式 artifact：[`runs/expected_discrete_context_holdout_ft100_seed20260726/result.json`](../runs/expected_discrete_context_holdout_ft100_seed20260726/result.json)，SHA256 `f8c86a1712aa84dbeb7e9a3c576b63038ddaeb0d4746a39480b6400928cf7a8b`。checkpoint SHA256 为 `96900700b046be4b11221dd7e1a4c2436f1aadc97936297c399ea403792093a4`。

## 推断速度

在同一批 48 个 unseen-context tasks 上，batch size 16、CPU 4 threads、20 次重复：

| 方法 | exact version-space task | ms / task |
|---|---:|---:|
| amortized support encoder | 100% | **0.237** |
| cached exhaustive 64-code scoring | 100% | 5.621 |

当前实现的 amortized speedup 为 **23.68×**。这是当前 CPU 实现的 wall-clock 对比，不是硬件无关复杂度结论。artifact：[`inference_speed.json`](../runs/expected_discrete_context_holdout_ft100_seed20260726/inference_speed.json)。

## 可以得出的结论

在给定 mechanism axes、codebook 和可靠 executor 后，快速规则抽象可以实现为：

\[
\text{support history}
\xrightarrow{\text{amortized encoder}}
q(h)
\xrightarrow{\text{top-K}}
\text{persistent version-space particles}.
\]

相比“大概 policy”，这个 posterior 保留了四个仍兼容的机制，因此可以复用于新 query、counterfactual rollout 和之后的信息增益动作选择。相比显式穷举，它把 search 的结果蒸馏进一次 forward，同时仍以离散 causal code 的真实代价训练。

## 仍然不能声称什么

这不是 autonomous causal discovery，也不是 ARC-AGI-3 结果：

- collision / trigger / relation 三个 axes 与每轴四值是人工给定的；
- executor 的 code semantics 使用 factor labels 和 symbolic support version space 预训练；
- training behavior panels 由 symbolic version space 与 simulator 构造，属于强 privileged teacher；
- palette role 仍是 oracle canonicalization；
- 只有 36 train + 12 test canonical contexts，且当前 geometry seed 没有真正改变布局；
- pair panel 的 9 项只有 3 个不同 input，triple 的 3 项只有 1 个不同 input；
- 只有一个正式 model seed；
- public `task_id` 可解析 program / heldout axis，active `probe_id` 可解析 strong / partial / neutral 类型。当前 tensor adapter 不编码这些字符串，所以本轮模型没有使用泄漏，但 active benchmark API 必须先修。

因此准确表述是：**在预结构化的离散因果空间中，expected-discrete credit assignment 能把 privileged exhaustive filter 摊销为快速、组合 context OOD 的 K4 推断器。**

## 下一步 gate

进入主动 EIG / GRAM / LoopWM 前，应先：

1. 将 public task/probe IDs 改成 opaque、无类型语义的 nonce；
2. 让 geometry RNG 真正生成 canonical 后仍不同的布局，并做 shape / D4 / spatial OOD；
3. 把 pair / triple fixtures 改成真正不同的 state-action probes；
4. 比较 factorized `3×4` proposer 与参数匹配的 unstructured 64-way proposer；
5. 跑至少 3 个 seeds，并加入 states+actions-only、semantic target derangement 和 tied-slot controls；
6. 再让当前 posterior 通过 EIG 选择互补 partial probes，验证 active 阶段是否比随机策略节省交互。

## 复现

```bash
/Users/yangzhenbang/anaconda3/bin/python3 scripts/run_expected_discrete_causal_coverage.py \
  --output runs/expected_discrete_context_holdout_seed20260726 \
  --executor-checkpoint runs/support_calibrated_executor_seed20260724/checkpoint_last.pt \
  --steps 500 --batch-size 8 --train-pool-tasks 144 --eval-tasks 48

/Users/yangzhenbang/anaconda3/bin/python3 scripts/run_expected_discrete_causal_coverage.py \
  --output runs/expected_discrete_context_holdout_ft100_seed20260726 \
  --executor-checkpoint runs/support_calibrated_executor_seed20260724/checkpoint_last.pt \
  --initial-checkpoint runs/expected_discrete_context_holdout_seed20260726/checkpoint_last.pt \
  --steps 100 --batch-size 8 --learning-rate 5e-4
```

完整回归为 `87 / 87` tests 通过；冻结 Stage 0-A manifest 仍验证通过。
