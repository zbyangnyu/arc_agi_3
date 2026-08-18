# Causal hypothesis filter：规则猜想、证据筛选与 latent version space

> 日期：2026-07-23 至 2026-07-25  
> 结论：在显式给定三轴机制空间、oracle palette role 和经过 support-domain 校准的世界模型后，枚举 64 个 latent 规则并用 public support 筛选 top-4，可在 192 个 held-out task 上达到 **Coverage@4 = 100%**。打乱 support target 后降到 **7.68%**。这证明“多个显式规则猜想 + 世界模型证伪”在该受控空间中可行，但不证明能从原始像素自主发现规则坐标系。

## 要回答的问题

此前连续 Persistent-K4 相对 tied-K1 有明显收益，但 Coverage@4 只有 `43.75%`。本轮进一步区分三个问题：

1. 一个网络能否直接把 support 摊销成四个离散 causal slots；
2. 冻结世界模型能否在 support 上正确比较规则；
3. 如果完整规则假说空间可枚举，“提出多个猜想，再用观察验证”本身能否恢复正确的 latent version space。

这里的“规则”不是自然语言公式，而是一个 persistent latent mechanism tuple：

\[
h=(h_{collision},h_{trigger},h_{relation})\in\{0,1,2,3\}^3.
\]

世界模型给出 \(p(s'\mid s,a,h)\)，filter 用 support likelihood 与 MAP consistency 排序 64 个 \(h\)，保留四个不同候选。query target 不参与推断。

## 1. 直接 causal-slot 摊销失败

第一个实现让 support encoder 输出四个无标签 slot，每个 slot 含三个四值 categorical code；straight-through hard code 经过冻结 factor executor，在完整行为 panel 上做一次全局 `4!` matching。

1000 step、8,000 tasks 的正式结果：

| 指标 | causal ST slots | continuous K4 |
|---|---:|---:|
| Coverage@4 | 10.94% | 43.75% |
| covered classes | 84 / 768 | 336 / 768 |
| all-four-covered task | 0% | 10.42% |
| mean unique rules | 1.02 | 3.83 MAP signatures |

causal slots 从 step 200 左右开始塌缩为约一个 hard tuple。打乱 support 后 Coverage 进一步降到 `1.82%`，说明 encoder 并非完全忽略 evidence；它读取了 evidence，但没有维持一个完整的假说集合。

artifact：[`runs/causal_mechanism_k4_seed20260723/result.json`](../runs/causal_mechanism_k4_seed20260723/result.json)，SHA256 `b6051346c6880e92f33354a234ae0c42f2942e36e65287483041b0ee021a76bc`。

## 2. 真正的首要 bug：world model 没有在 evidence domain 校准

原 executor 在 diagnostic `0..20` 训练，并在 held-out triple `21..23` 达到 100% exact；但它没有在 episode 的六条 support transition 上训练或验证。

64-code audit 显示：

- 四个真实 compatible tuple 在 diagnostic behavior panels 上仍为 100% exact；
- 同四个 tuple 在 support 上的 MAP exact particle rate 为 0%；
- 真实 tuple 的 support cost 平均 rank 约为 `32 / 64`，接近随机；
- 因而旧目标中的 `0.1 × support_nll` 实际会系统性惩罚真规则。

这说明“query executor 已通过”不等于“可用作因果后验的 likelihood model”。若 observation domain 与 counterfactual-query domain 不同，两者必须分别校准和审计。

## 3. Support-calibrated factor executor

新 executor 仍使用 pooled architecture、oracle palette canonicalization 和显式三轴 factor input。区别是每个 task 同时训练三个 domain：

- diagnostic `0..20`：权重 `0.50`；
- 前两条 calibration support：权重 `0.25`；
- 后四条 neutral support：权重 `0.25`。

每条 support 不只配 true program，而是复制给 support version space 中全部四个 compatible tuples。这样未被观察区分的 axis value 必须产生相同 evidence likelihood，不能获得虚假的偏好。

500 step、4,000 tasks 后，在独立 split 的 192 tasks 上：

| 审计 | 结果 |
|---|---:|
| single / pair / held-out triple exact task | 100% / 100% / 100% |
| compatible support exact frame | 100% |
| all-four compatible six-frame exact task | 100% |
| neural MAP-exact bank = symbolic version space | 100% |
| proper-NLL top-4 exact version space | 100% |
| compatible likelihood spread | `1.21e-5` nats/cell |

artifact：[`runs/support_calibrated_executor_seed20260724/result.json`](../runs/support_calibrated_executor_seed20260724/result.json)，SHA256 `dccf7aff017965c8e0352c93b63605afc706f21a85a454f83e7d21f6a60e7506`。checkpoint SHA256 为 `2b2d0e5f3380f6b45fef0b2c0892186fd53dc51a3eb6c4ebc6273ef2895f5254`。

## 4. 显式 latent hypothesis filter

推断阶段固定执行以下步骤：

1. 枚举 \(4^3=64\) 个不同 latent mechanism tuples；
2. 对每个 tuple，用冻结 executor 预测六条 public support outcome；
3. 首先按完整 support 的 MAP error 排序，再用 changed/unchanged-balanced NLL 打破平局；
4. 保留四个 persistent hypotheses；
5. 用同一组四个 hypotheses 预测全部 held-out triple queries。

推断不读取 true program、factor label 或 query target。正式 evaluation 仍同时要求 coherent full-panel MAP exact 和 `NLL ≤ 0.05` nats/cell。

192 tasks、768 behavior classes 的结果：

| 指标 | causal hypothesis filter | shuffled-support control | continuous K4 | causal ST slots |
|---|---:|---:|---:|---:|
| Coverage@4 | **100%** | 7.68% | 43.75% | 10.94% |
| all-four-covered task | **100%** | 2.08% | 10.42% | 0% |
| factor-tuple Coverage@4 | **100%** | 7.68% | — | 10.94% |
| all particles fit support | **100%** | 12.5% | — | 0% |
| mean unique rules / signatures | **4.0 / 4.0** | 4.0 / 4.0 | — / 3.83 | 1.02 / 1.02 |

三个 heldout axes 各有 64 tasks，coverage 均为 100%。uniform prior 下，symbolic version space 获得的平均 posterior mass 为 `99.9971%`；在四个 compatible tuples 内，normalized entropy 为 `0.999997`，最大 conditional mass 仅 `25.04%`。模型不仅选对集合，也保留了尚未被 evidence 区分的不确定性。

artifact：[`runs/causal_hypothesis_filter_seed20260725/result.json`](../runs/causal_hypothesis_filter_seed20260725/result.json)，SHA256 `2b82e1fbf22f60854c4dd7d8a48f8ee1b27767542ebd089e9909346205167d56`。

## 5. 快速抽象发生在哪里

逐条加入 support 的结果为：

| prefix | symbolic version-space size | top-4 compatible recall | true rule in top-4 | neural exact bank = symbolic bank |
|---:|---:|---:|---:|---:|
| 1 | 16 | 25% | 25% | 100% |
| 2 | 4 | 100% | 100% | 100% |
| 3–6 | 4 | 100% | 100% | 100% |

第一条 calibration evidence 排除一个 axis 的三种错误取值，留下 16 个规则；第二条再识别另一个 axis，收缩到 4 个。后四条 neutral evidence 不应也没有继续缩小集合。这里的“抽象”可以精确表述为：把 observation history 映射为一个持久 latent version space，而不是过早压成单一 policy state。

## 这支持什么结论

支持：

- 一个规则可以用 latent mechanism tuple 表示，不必先给出自然语言或手写公式；
- 对需要反事实组合泛化的任务，维护一个规则后验比只输出大概 policy 更稳妥；
- 世界模型若能在 evidence 和 query 两个 domain 上正确执行，显式 hypothesize-and-test 可以快速恢复多个兼容规则；
- 旧连续 K4 的主要问题至少部分来自 hypothesis search / credit assignment，而不是规则不可表示；
- support likelihood 的校准是 causal filtering 的硬前提。

不支持：

- 从 raw pixels 自主发现 collision / trigger / relation 三个 axis；
- public-compliant palette binding；
- 对开放式 ARC-AGI-3 规则空间穷举；
- 当前 benchmark 上的主动实验能力，因为前两条固定 calibration support 已经完成被动收缩；
- causal factors 的独立性已经被发现。这里的 factorization 与 support version-space supervision 都是 privileged 的。

## 对 JEPA、GRAM、LoopWM 与 TTT 的架构启示

合理的组合不是让一个 JEPA embedding 同时承担所有职责，而是分层：

\[
\text{objects/events}
\rightarrow \text{mechanism hypotheses}
\rightarrow \text{world-model consistency}
\rightarrow \text{posterior/version space}
\rightarrow \text{information-gain action}.
\]

- JEPA 可负责学习只保留可预测对象、事件与 action-conditioned change 的 representation；但 latent target 必须保留机制差异，不能把因果信息当 nuisance 丢掉。
- GRAM 式多轨迹适合在 64 枚举不可行时生成一组 stochastic latent programs；每条轨迹仍需由 world-model evidence likelihood 重加权和淘汰。
- LoopWM 式共享循环适合迭代执行“提出局部修改 → rollout → 比较 observation → 更新 posterior”，但循环深度不能代替多峰 belief。
- TTT / AdaJEPA 式适配应优先更新 perception/binding 或 likelihood calibration；直接在单一 latent 上最小化 prediction error，容易把互相冲突的规则平均掉。
- policy 可以在规则后验之后蒸馏：\(\pi(a\mid s,q(h))\)。在需要 OOD counterfactual transfer 时，不应让 policy 取代 \(q(h)\)。

## 下一步

1. 把 oracle palette canonicalization 替换成 public legend 或从 support 学到的 object-role binding。
2. 不再给定 factor value semantics；以 64-code detached discrete cost 或 contrastive interchange loss训练 amortized \(q(h\mid support)\)，避免 straight-through decoder 的错误切向梯度。
3. 做 held-out factor-combination 的 single-axis swap / causal interchange：换一个 axis latent 后，rollout 必须与独立 simulator 的对应机制一致。
4. 修改 benchmark，使不同 action 只产生互补的 partial partitions，移除 universal strong probe；然后直接在当前 version space 上使用 EIG 选 action。
5. 当显式 bank 太大时，再比较 beam search、GRAM-style particles 与 LoopWM refinement 的 compute–coverage 曲线。

其中第 2 项的 expected-discrete amortization 已完成：在 36/12 个互斥 observed-factor contexts 上训练/评估后，K4 达到 100% Coverage，并比当前 cached exhaustive filter 快约 23.7×。详见 [Expected-discrete K4 report](amortized_discrete_causal_report.md)。这仍未解除 palette、geometry、fixture 与 public-ID 限制。

## 复现与回归

```bash
/Users/yangzhenbang/anaconda3/bin/python3 scripts/run_support_calibrated_executor.py \
  --output runs/support_calibrated_executor_seed20260724 \
  --steps 500 --batch-size 8 --eval-tasks 192

/Users/yangzhenbang/anaconda3/bin/python3 scripts/run_causal_hypothesis_filter.py \
  --output runs/causal_hypothesis_filter_seed20260725 \
  --executor-checkpoint runs/support_calibrated_executor_seed20260724/checkpoint_last.pt \
  --continuous-result runs/canonical_latent_k4_seed20260722/result.json \
  --slot-result runs/causal_mechanism_k4_seed20260723/result.json \
  --eval-tasks 192
```

完整测试为 `81 / 81` 通过。Stage 0-A reference 的 result SHA256 仍为 `77dc5eaded9b3d98b03dddd79cf731e00e4d67076a1a42e12b1e155304fc7c7d`，source manifest 验证通过。
