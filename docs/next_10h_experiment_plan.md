# 接下来 10 小时：Cross-axis likelihood locality audit

> 状态：已完成。正式结果通过全部 gate 与敏感性检查，见
> [结果 JSON](../runs/forced_cross_axis_likelihood_audit_g64_b16_seed2026072401/result.json)
> 和[分析报告](../runs/forced_cross_axis_likelihood_audit_g64_b16_seed2026072401/analysis.md)。

## 唯一主目标

冻结并完成一个**无主动选择器**的因果局部性审计，回答：

> 当前 learned executor 是否会让与 query 无关的观测产生错误规则证据；如果把
> likelihood 投影到 probe 真正作用的 factor，这些 posterior 反转是否消失？

这比立即加训练步数或跑 harder game 更优先。现有 2×2 已发现约 37 nats 的伪
Bayes factor，但失败数受 B1 后近似并列的动作选择影响；forced audit 可以去掉这个
混杂。

## 冻结协议

- 新 held-out split；
- 3 个 query axes × 全部 64 个 true programs × 2 个固定 atomic
  query variants × 6 个 forced probes；
- 主分析共 **2,304 条唯一语义序列**。64 个 public palettes 在 oracle
  canonicalization 后只剩候选顺序变化，因此仅作为 palette/order invariance
  重复，不能当作独立环境或扩大主分母；
- 先强制执行一个 query-axis atomic probe；
- 再从同一个 query posterior，分别独立执行 4 个 cross-axis probes 和 2 个
  neutral probes，不顺序累积；
- 不运行 selector，不读取 candidate kind 进行决策；
- 对比五条路径：
  1. `EE`：exact query → exact forced；
  2. `RR`：raw learned query → raw learned forced；
  3. `RP`：raw learned query → projected learned forced；
  4. `PR`：projected learned query → raw learned forced；
  5. `PP`：projected learned query → projected learned forced。

对 axis `a` 的 probe，projection 将 learned likelihood 对另外两个 factor 做
`logmeanexp`，然后只按 `h_a` 广播；neutral probe 使用全 code 常数。它不是可部署
模型，而是判断“factor locality 是否足以救回 posterior”的 oracle 因果干预。
projection 必须作用于 `OutcomePrediction.log_prob(feedback)` 给出的 proper
full-grid likelihood，不能先平均 logits 或逐 cell probability。

这里保留固定 geometry 是有意选择：现有 executor 对随机 geometry 的 query
positive control 只有约 70% top-1，会把 OOD 泛化失败混入 locality 审计。固定
geometry 的 exhaustive query control 为 100% top-1；其局限在报告中明确记录，
下一轮再单独扩展 geometry 泛化。

## 主指标与预注册 gate

报告：

- query probe 后及 forced probe 后的 `p(true query)`；
- true-query log-odds 变化和 P99 drop；
- high-confidence catastrophic reversal rate；
- cross-axis marginal KL / TV；
- 各 probe 类别的 outcome-partition F1；
- worst 20 counterexamples。

有效性检查：

- exact query probe 必须留下 16 个 hypotheses，且 query value 唯一；
- exact cross-axis forced probe 后必须留下 4 个 hypotheses，neutral 后仍为 16；
- exact forced control 的 query marginal 必须逐元素不变，KL/TV 为 0；exact
  log-odds 是 `+∞`，不计算 `∞−∞`；
- uniform prior 下 raw/projected query evidence 与 query marginal 必须相等；
- learned likelihood/evidence 全部有限；
- true-code exact map 与 canonical feedback 完全一致；
- 64 个 palette/order repeats 的 canonical semantic summary 必须完全一致；
- batch size 16/64 的核心汇总一致。

决策 gate：

- **支持 factorized JEPA/executor**：`RR` cross-axis reversal `≥1%`，或 P99
  odds drop `≥5` nats；`PP` 对达到阈值的失败指标改善 `≥90%`、不产生新的
  reversal，且 true-code conditional full-grid query NLL 恶化 `<0.05` nat。
- **转向 within-axis calibration**：raw 明显失败，但 projection 无法救回。
- **转向 harder compositional benchmark**：raw 未达到上述失败 gate，说明旧反例
  太稀，不能作为首要架构依据。

`RP/PR` 只作定位诊断：区分错误主要来自 forced-likelihood 的跨轴污染，还是
query-stage 已经诱发的轴间相关。

## 10 小时时间盒

| 时间 | 交付物 |
|---|---|
| 0–1h | 冻结 schema、公式、split 和 gate |
| 1–3.5h | runner 与协议单测 |
| 3.5–4.5h | smoke、exact control、泄漏边界检查 |
| 4.5–6h | full run |
| 6–7h | batch-size、split 和 palette/order invariance 复核 |
| 7–8.5h | 按 canonical semantic key 去重分析与反例导出 |
| 8.5–10h | 一页结果报告和 go/no-go 决策 |

本时间盒**不重训 GRAM/JEPA、不改 policy、不报告 ARC-AGI-3 成绩**，避免同时改变
belief、world model 和 benchmark。

## 紧接着的下一项

若 locality audit 完成，下一实验使用 two-axis compositional door query。两个相关
axis 的 atomic probe 各只揭示一位，理论 exact/exact 曲线应严格为
`B0=0.25 → B1=0.50 → B2=1.00`；它比现有一击全解的 single-axis menu 更适合检验
多步主动规则发现。
