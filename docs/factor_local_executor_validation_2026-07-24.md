# Factor-local executor：同参数量实验验证

日期：2026-07-24

## 结论

这轮实验支持以下判断：

> 当前双轴闭环失败的主要近因不是模型参数量太小，而是全局 decoder
> 没有满足任务本身的条件独立性；无关 factor 的微小 per-cell 偏差经过
> full-grid likelihood 累加后，会成为很大的错误 posterior evidence。

证据来自两个相互补充的干预：

1. 不训练新模型，只对原 global executor 的 learned likelihood 做 oracle
   轴投影。`learned-select/learned-update` 的 B2 从 89.06% 提升到
   97.57%，恰好等于同一 learned selector 配 exact update 的上限。
2. 把 factor locality 写进执行器计算图，保持参数量与 state-dict
   完全相同（两者均为 17,177 参数）。新模型的原始 learned update
   已与 oracle 投影数值等价，并在正式 B2 上达到 100%。

这不是最终 agent 架构的验证。当前 router 使用 oracle-canonical
角色颜色和动作类型，是用于定位因果瓶颈的结构消融；下一步仍需 learned
public router、matched wider-global control 和 3 个训练 seed。

## 被检验的假设

- H-scale：原模型主要因为参数/数据规模不足而失败。
- H-structure：原模型的 factorized input 最终进入同一个 global
  decoder，允许 nuisance factor 污染 acted-axis likelihood；这是
  两步组合推理失败的主要原因。

同参数量 routed 条件控制了最直接的 capacity 混杂。oracle projection
则提供了不改变模型预测、只改变 evidence geometry 的硬上限。

## 实验条件

Global 与 routed continuation 使用完全相同的：

- 初始 audited checkpoint；
- model seed `2026072402`；
- 100 个 updates；
- batch size 8、每任务 8 个 factor codes；
- AdamW learning rate `1e-4`、weight decay `1e-4`；
- locality weight `0.1`，50-step ramp；
- geometry weight `0.1`；
- frozen-teacher categorical KL weight `10000`；
- train/eval geometry seed streams。

两者参数量均为 17,177。routed executor 没有新增 head：对每个
canonical action atom，router 选择 collision、trigger、relation 或
neutral；未激活 axis 的 factor embedding 被该 axis 四个 embedding
的均值替代。因而 singleton probe 对两个 nuisance factors 严格不变，
composite action 可以同时激活多个 axes。

正式跨轴审计使用每个 query 64 个 palette/order groups：

- 1,536 条 cross-axis semantic sequences；
- 768 条 neutral sequences；
- 每个 raw/projected branch 共 2,304 条。

正式双轴闭环使用：

- 3 个 query-axis pairs；
- 每个 pair 16 groups；
- seeds `20260911, 20260912, 20260913`；
- 共 144 scenarios；
- 每个 condition 2,304 个 truth tasks。

## 结果

### 1. 能力保持与随机几何

| 指标 | Global | Factor-local routed | 差值 |
|---|---:|---:|---:|
| 参数量 | 17,177 | 17,177 | 0 |
| active-prefix 192-task hard gate | 100% | 100% | 0 pp |
| single / pair / heldout-triple exact task | 100% | 100% | 0 pp |
| heldout-triple NLL / cell | 0.000089 | 0.000613 | +0.000524 |
| heldout-singleton exact MAP grid | 24.93% | 18.75% | -6.18 pp |
| heldout-singleton NLL / cell | 0.33614 | 0.35744 | +0.02130 |
| singleton mean nuisance-fiber range | 16.595 nats | 0 | -16.595 |
| singleton max nuisance-fiber range | 62.971 nats | 0 | -62.971 |

结构约束消除了 nuisance dispersion，也保留了所有旧 hard gates；但
随机 singleton 的静态 MAP/NLL 略差。这说明 locality 不是免费的，
后续需要更好的 local event head 或 base/axis decomposition，而不是
简单地扩大同一个 global decoder。

### 2. Forced cross-axis likelihood audit

| 指标 | Global raw | Factor-local raw |
|---|---:|---:|
| RR catastrophic reversal | 0.5208% | 0% |
| RR cross P99 log-odds drop | 18.352 nats | 0.0000038 nats |
| RR cross maximum drop | 27.584 nats | 0.0000038 nats |
| learned outcome-partition F1 | 0.9098 | 0.9818 |
| exact outcome-partition rate | 62.5% | 87.5% |
| mean true query probability | 0.9679 | 0.9831 |

Factor-local raw 与 projected branch 的 query posterior 相同到浮点误差。
仍有一个局部弱点：`collision:v1` 的 exact partition 为 0、pair F1 为
0.7895。结构保证阻止了这个预测误差污染其他 factor，但没有让每个
局部 dynamics prediction 本身都变正确。

### 3. 两步双轴闭环 B2

| Selection / update | Global raw | Global + axis projection | Factor-local raw |
|---|---:|---:|---:|
| exact-select / learned-update | 95.31% | 100% | 100% |
| learned-select / exact-update | 97.57% | 97.57% | 100% |
| learned-select / learned-update | 89.06% | 97.57% | 100% |
| complementary relevant-axes rate，learned/learned | 78.13% | 95.14% | 100% |

原模型从 `learned/exact` 的 97.57% 降到 `learned/learned` 的 89.06%，
是 updater 引入的 8.51 percentage-point 损失。oracle axis projection
精确消除了这部分损失。把同一独立性放进网络后，raw learned update
与 projected update 完全重合，且 selection 也因更好的 outcome
partition 从 97.57% 提升到 100%。

## 决策

当前应优先推进 factor-local architecture，而不是先扩大同一个 global
decoder 或盲目增加训练数据。更大模型仍可能改善 `collision:v1` 和
随机 geometry NLL，但它不会自动保证无关 factor 不进入 likelihood。

下一阶段的预注册主实验应为：

1. learned public event router：只读公开 observation/action tokens，
   禁止 oracle axis、probe ID、split、seed 和 oracle palette role；
2. 三臂、三 seed：原 global、参数匹配的 wider-global、参数匹配的
   factor-local branches；
3. selector 与 updater 共享同一个 stochastic predictive model；
4. 保留 full 64-code joint belief，不把三个 marginals 的乘积当正式
   posterior；
5. 继续用当前 RuleGrid forced-cross-axis 与 B2 作为结构单元测试，
   通过后再进入多任务 exploration benchmark。

PushT 只有在被扩展为“多个隐藏动力学/接触规则 × 多目标”的 task family
时才适合检验跨场景探索迁移；单一 vanilla PushT 更偏连续控制能力，
不能替代这里的 hidden-rule identification test。

## Artifacts

- Global continuation：
  `runs/counterfactual_locality_distill100_w10000_seed2026072402/result.json`
- Factor-local continuation：
  `runs/canonical_routed_factor_executor_v3_distill100_w10000_seed2026072402/result.json`
- Global forced audit：
  `runs/forced_cross_axis_likelihood_audit_distill100_w10000_g64_b16_seed2026072401/result.json`
- Factor-local forced audit：
  `runs/forced_cross_axis_routed_v3_g64_b16_seed2026072401/result.json`
- Global projection B2：
  `runs/two_axis_projected_update_g16_s3_20260911/result.json`
- Factor-local B2：
  `runs/two_axis_compositional_routed_v3_g16_s3_20260911/result.json`

## Verification

- Targeted routed/projection tests：51 passed。
- Full repository regression：267 passed。
- 历史 `prp_wm/latent_rules.py` SHA256 已恢复为
  `3e21377ed789b43688801ed087ddeff37209dc1e86daa6f46db8db236a52b39c`；
  新结构位于独立实验模块，旧预注册运行的源码身份没有被改写。
- v3 routed checkpoint SHA256：
  `523c08f5bf6c13f0519399040def1d2078d0229183ae88b25162c925a3e492d0`；
  manifest 显式记录 `prp_wm/routed_executor.py` SHA256
  `a209b723e0c607b897fdf8cf652861621373f126a20a5a5c077fdfee5194056e`。
