# Meta-learning × JEPA 外部验证研究计划

日期：2026-07-24

## 结论

前面的方向**基本合理，但需要四个修正**：

1. 当前模型确实小，数据也少；但“规模不足”不是现阶段最有证据的主瓶颈。当前约
   5.4 万参数的 raw belief model 已能通过仓库内静态、held-out geometry composite
   和 active-prefix belief gate。尚未解决的是：训练仍依赖 public version-space
   teacher、executor 曾用 oracle canonical roles 预训练、learned action selection
   尚未在最终 DoorGame 闭环中替代 uniform selector，而且没有外部环境结果。
2. 不应追求“更多 benchmark 分数”，而应增加**能隔离缺失能力的 benchmark**。
   同时维护 RuleGrid、Push-T 家族和一个团队外的 meta-RL 确认集已经足够。
3. LeJEPA 主要是稳定 joint-embedding 表示、避免 collapse 的学习目标，不是
   meta-learning 算法。与本项目直接相关的是 action-conditioned
   **LeWorldModel（LeWM）**；它适合做像素 encoder/predictor，不应替代 persistent
   task belief、likelihood update 或主动辨识。
4. 标准 Push-T 值得做，但只能证明像素动力学与 planning。**Symbolic Alchemy**
   才是近期最直接的团队外因果 meta-learning gate；加入 episode-level hidden
   dynamics 的 Meta-Push-T 是连接 belief 与连续像素控制的 diagnostic bridge，
   不能成为唯一外部证据。

所以推荐路线不是“立即把模型放大并铺很多 benchmark”，而是：

```text
7–10 天内部 close-out
        │
        ├── Symbolic Alchemy：主因果 meta-learning gate
        │
        └── 标准 Push-T / LeWM：像素与 planning gate
                                  ↓
                        Meta-Push-T oracle headroom
                                  ↓
                LeWM state latent + persistent task particles
                                  ↓
                  passive adaptation → active dual control
```

## 当前方法处于什么位置

可以把成熟度分成五级：

| 等级 | 定义 | 当前状态 |
|---|---|---|
| M0 | exact/symbolic oracle 可行性 | 已通过 |
| M1 | learned component 在合成环境可工作 | 已通过 |
| M2 | raw observation 下的 belief 与局部闭环 | **当前所在** |
| M3 | 外部环境中的被动适配与规划 | 未验证 |
| M4 | 外部环境中的主动辨识优于强基线 | 未验证 |
| M5 | 跨 benchmark/ARC-AGI-3 泛化 | 未验证 |

支持 M2 的证据：

- `raw_palette_invariant_atom_matched` 的四个 folds 已完成；fold 0 模型为 54,024
  个参数、CPU 600 steps、4,800 tasks，静态、held-out composite 和 active-prefix
  factor-set audit 均为 100%。
- exact two-axis selector 在 B2 达到 100%；learned selection + learned update 为
  82.6%，误差主要来自 learned outcome partition。
- sequence/mixed finetune 后，DoorGame 在 uniform-without-replacement probes、
  budget 8 下，unmarked/marked 平均 win rate 分别约 80.1%/94.3%。
- learned likelihood 仍会跨 factor 污染：raw 路径的 high-confidence reversal 为
  2.083%，P99 true-query odds drop 为 35.712 nats；oracle factor projection 才能
  基本消除。

这些结果说明：**“多假设 belief 有用”已经不再是最急需证明的问题；现在要证明的是，
从非特权轨迹得到的 likelihood 是否可信，以及这种 belief 是否能在外部环境节省真实
交互。**

尚不能声称：

- 已自主发现 rule axes 或 causal variables；
- 已有 learned active controller；
- 已从 raw pixels 学得可规划的世界模型；
- 已在 ARC-AGI-3、Push-T、Alchemy 或 MetaWorld 上有效；
- 扩大当前网络和重复生成同分布 RuleGrid 数据会自然跨过上述缺口。

## 冻结的研究命题

主问题：

> 在未见的 episode-level dynamics 上，持久、多峰、因子化的 task belief，配合
> action-conditioned latent world model，是否比单一 context、普通 ensemble 和
> 在线梯度适配更准确地识别隐藏机制，并用更少交互完成目标规划？

四个可证伪子问题：

- **RQ1 — Representation：** LeWM/SIGReg 是否让 raw-pixel latent rollout 更稳定，
  并支持低成本 planning？
- **RQ2 — Belief：** Persistent-K4 是否在 held-out dynamics compositions 上优于
  matched K1、recurrent context 和普通 ensemble？
- **RQ3 — Active inference：** query/goal-relevant value of information 是否比
  uniform、global information gain 和 task-agnostic uncertainty 更省交互？
- **RQ4 — Composition：** 因子化 task belief 是否组合泛化，而不是记住有限 task IDs？

若 RQ2 或 RQ3 失败，不能用“模型还不够大”作为默认解释；必须先由 matched scaling
和 oracle headroom 实验定位失败层。

## Benchmark 选择

| Benchmark | 真正测到的能力 | 与主张的匹配度 | 决策 |
|---|---|---:|---|
| RuleGrid / DoorGame | likelihood locality、多假设、主动辨识的单元测试 | 高但内部 | 保留为 regression，不再扩成主 benchmark |
| 标准 Push-T | raw pixels、接触动力学、latent rollout、CEM/MPC | 中 | 立即做 bridge/reproduction |
| Symbolic Alchemy | latent causal structure、online inference、hypothesis testing | **最高** | 主外部 benchmark；先限时做环境审计 |
| Meta-Push-T | 隐藏物理、episode adaptation、主动试探、连续控制 | 很高 | 像素/控制 diagnostic bridge，先过 oracle gate |
| PointMaze / Two-Room | 长时序 planning、障碍与拓扑 | 中低 | 只在需要定位 planning horizon 时加入 |
| MetaWorld ML10/ML45 | held-out tasks、few-shot adaptation | 中 | 后置；控制和 reward confound 较多 |
| ARC-AGI-3 public games | 最终交互泛化 | 最终目标 | 全系统冻结后做 exploratory scorecard |

标准 Push-T **不能单独成为主结果**。它通常在固定任务/物理下训练和评测，成功可能仅
说明一个单 latent world model 学会了接触动力学，并不要求维护多个 task hypotheses。

Push-T 线优先使用 `stable-worldmodel` 作为实验底座：它已经提供 Push-T、LeWM、DINO-WM、
CEM/iCEM/MPPI 以及可控 visual/physical factors of variation。这样可以避免先重写
数据、环境和 planner，再把基础设施 bug 误判成方法失败。该平台仍处于快速迭代期，
因此必须固定 commit、环境镜像和数据 manifest，并与当前 Python 3.12 仓库使用隔离
环境。

## 三条工作流与闸门

### Track A：内部因果链 close-out

时间上限：7–10 天。到期后必须形成 pass、fail 或缩小 claim 的结论，不能无限延长。

目标不是再提高静态 factor accuracy，而是修复 learned likelihood 与 learned action
selection。

最小条件：

1. 当前 raw learned executor/belief；
2. factor-local likelihood heads；
3. locality + frozen-teacher distillation/parameter anchoring；
4. oracle projection 上界。

固定指标：

- cross-axis high-confidence reversal；
- P99 true-query log-odds drop；
- full predictive outcome NLL/Brier，而不是 MAP partition alone；
- outcome-partition pair F1；
- 旧 active-prefix、held-out geometry composite 保持率；
- learned/learned B2 success；
- DoorGame learned-selector budget-success curve。

Go gate：

- reversal `≤ 0.1%`，目标为 exhaustive 0；
- P99 odds drop `≤ 0.5 nat`；
- 旧 static/active/geometry gates 不低于 95%，正式模型恢复既有 100%；
- learned/learned B2 `≥ 90%`；
- learned selector 相对 uniform-without-replacement 的 paired RMST 改善置信区间下界
  大于 0。

No-go：

- 若只有 oracle projection 能过 gate，结论应写成“故障已定位但 learned locality
  未建立”，停止为旧 RuleGrid 增加模型宽度和训练步数。
- 若 belief 正确但 selector 失败，继续修 predictive outcome distribution；不更换
  benchmark，也不把失败归因于 K 太小。

### Track B0：标准 Push-T / LeWM 资格测试

时间：第 1–2 周，可与 Track A 并行。

先跑官方 checkpoint/eval，再训练最小 matched ablation：

1. official JEPA-WM 或 DINO-WM checkpoint；
2. LeWM（latent prediction + SIGReg）；
3. 相同 encoder/predictor 去掉 SIGReg；
4. simulator-state/oracle predictor 作为 planning ceiling。

固定观测只含 pixels 的 track 与允许 state diagnostic 的 track 必须分开报告。

指标：

- task success/target coverage；
- 1、4、8、16-step latent rollout error；
- representation covariance、effective rank 与 collapse incidents；
- CEM samples-to-success、wall-clock planning latency；
- frozen probes 对 agent position、block pose/angle 的误差，仅作诊断。

Platform gate：

- 官方 checkpoint 在相同 protocol 下可复现，结果落在官方或重新计算的 95% CI 内；
- observation/action/frame alignment、normalization 和 planner 均有 deterministic
  smoke tests；
- LeWM 相比 no-SIGReg 在 matched compute 下显著降低 collapse/rollout error，
  或提高 planning success。

若最后一项失败，不停止外部 benchmark；只停止“LeJEPA 是关键贡献”的分支，保留更强
的 DINO-WM/JEPA-WM encoder，继续检验 persistent belief。

### JEPA × PRP 的最小集成顺序

不要把 LeJEPA、factorized predictor、K4 和 active policy 一次加入。按以下顺序消融：

| 条件 | 唯一新增部分 | 回答的问题 |
|---|---|---|
| proper executor | 当前 calibrated outcome head | 基线 |
| `+ latent prediction` | action-conditioned JEPA loss | 是否改善 geometry/pixel OOD |
| `+ SIGReg` | 仅 state projection head | 收益是否来自 anti-collapse |
| `LeWM-only` | 去掉 proper outcome head | latent distance 能否安全充当 evidence，预期负对照 |
| `+ factor-local predictor` | mechanism-local delta/likelihood heads | 是否消除 cross-factor 污染 |
| `+ Persistent-K4` | persistent task modes | 多峰 belief 是否额外有用 |
| `+ active VOI` | query/goal-relevant action selection | 是否真正节省交互 |

LeWM 的 latent MSE 是 representation/planning objective，不是天然归一化的
\(p(o_{t+1}\mid o_t,a_t,h_k)\)。PRP 的 posterior update 必须保留 proper predictive
head 或经独立校准的 conditional density；否则会重复仓库中“query prediction 很准，
support likelihood 却淘汰真规则”的失败。

### Track B1：Meta-Push-T-Lite 的 oracle qualification

时间：最多 72 小时设计 + 1 周实现。**先 state，后 pixels。**

不要一开始自造新的物理系统。从 `swm fovs PushT-v1` 暴露的 factors 中选择三个
episode-level、动作相关且可辨识的 physical factors。候选可以包括 control gain/
delay、contact/friction、object inertia 等，但最终名称和值必须来自实际环境能力，
不得把不可控或等价的参数硬编码成“ground truth factors”。

任务结构：

- 每个 meta-episode 开始时采样隐藏机制 \(h=(h_1,h_2,h_3)\)，整 episode 固定；
- agent 看不到 physics/task ID；
- context phase 允许 0/1/2/4/8 个 probe actions；
- evaluation phase reset state 但保持 \(h\)，再执行 Push-T goal，使试探价值与最终
  初态变化解耦；
- visual nuisance 与 physics 独立采样，防止颜色泄漏 task ID；
- passive track 使用固定、配对 action prefix；active track 由 agent 自选 probes。

拆分原则：

- composition split：每个单轴值在 train 中都出现，但 held-out joint tuples 从未出现；
- value-OOD split：未见单轴值，必须单独报告，不能与 composition generalization 混合；
- 相同 initial state 与 action sequence 在不同 \(h\) 下产生 counterfactual pairs；
- test manifest 在模型、early-stop threshold 和 planner 全部冻结前不可打开。

建议首轮采用三轴三值，共 27 个组合：18 train、4 validation、5 test。若 oracle gate
通过，再扩为三轴四值或连续区间。这样可以先验证科学问题，而不是先生成大数据。

Oracle benchmark gate：

- oracle-\(h\) planner success `≥ 90%`；
- oracle-\(h\) 相对 history-blind/domain-randomized robust planner至少高 15
  percentage points，或动作成本低 20%；
- exact/state likelihood 在不超过 4 个 probes 内识别 \(h\) 的成功率 `≥ 90%`；
- optimal/query-relevant probes 明显优于 random probes；
- visual-only nuisance 不改变 physics posterior；
- 不存在大规模不可辨识等价类；若存在，评测 equivalence class，而不强求 raw ID。

若 gate 失败，允许修改一次 factor/value 设计；第二次仍失败则停止 Meta-Push-T，
因为此时 benchmark 本身没有足够的 adaptation/active-inference headroom。

### Track B2：Passive Meta-Push-T

先用固定 context trajectories 隔离 belief learning，不引入 learned action selection。

推荐架构：

\[
z_t^s = E(o_t),\qquad
p_k(z_{t+1}^s\mid z_t^s,a_t,h_k)
=\mathcal N(\mu_k,\operatorname{diag}\sigma_k^2)
\]

\[
w_{t+1,k}\propto w_{t,k}\,
p_k(z_{t+1}^s\mid z_t^s,a_t,h_k).
\]

边界必须清楚：

- SIGReg 只施加在共享的 **state/observation latent** \(z^s\)；
- 不对 particle weights、离散 task modes 或 factor posterior 强制 isotropic Gaussian；
- belief update 使用校准的 predictive density/energy，不能直接把未归一化 embedding
  distance 当 Bayes likelihood；
- state uncertainty、episode task uncertainty 与 parameter epistemic uncertainty
  分开报告；
- particle identity 跨 prefix 持久化；保留 uniform/stratified proposal floor，避免
  当前 GRAM proposal holes 重现。

`When Does LeJEPA Learn a World Model?` 的 identifiability 结果依赖 stationary、
additive-noise transitions，并在关键结论中使用 Gaussian latent；这不能直接外推到
离散、组合、episode-varying 的规则机制。因此 SIGReg 在这里是待检验的
anti-collapse regularizer，不是“已经发现因果变量”的理论证明。

最小参数匹配基线：

1. task-agnostic LeWM（no context）；
2. GRU/Transformer recurrent context；
3. probabilistic single-context/K1；
4. K=4 ensemble，但无 persistent task identity；
5. Reinfer-K4，每个 prefix 从头推断；
6. Persistent-K4；
7. oracle-\(h\) ceiling。

主指标是 held-out composition 上的 **adaptation AUC**：

\[
\operatorname{AUC}_{B\in\{0,1,2,4,8\}}
\text{Success after }B\text{ context actions}.
\]

诊断指标：

- true-factor/equivalence-class Recall@K、Coverage@K；
- predictive NLL、Brier、ECE；
- counterfactual rollout NLL；
- neutral/visual-nuisance posterior drift；
- particle lineage stability；
- regret to oracle-\(h\) planner；
- 参数量、训练 FLOPs、planning time。

Passive gate：

- Persistent-K4 的 adaptation AUC 对最强 learned baseline 的 paired 95% CI 下界
  大于 0；
- held-out composition Coverage@4 `≥ 90%`；
- shuffled history 回到 prior/明显下降，排除 robust-policy shortcut；
- visual-only nuisance posterior drift 不显著；
- K4 相对 K1 的收益不能由四倍参数量解释。

若 K4 对 recurrent context、ensemble 和 Reinfer-K4 都无优势，persistent
multi-hypothesis 是被证伪的主张；此时不通过增加 particles 掩盖结果。

### Track B3：Active Meta-Push-T

只有 passive gate 通过后才让模型自主选择 context actions。

策略比较：

1. uniform/random probes；
2. task-agnostic uncertainty/disagreement；
3. global information gain；
4. goal/query-relevant value of information；
5. exact or state-oracle depth-limited selector；
6. oracle-\(h\) planner。

首先使用显式两阶段 `probe → reset-with-same-h → goal` protocol；通过后才尝试将信息
价值和 task reward 统一到 dual-control MPC：

\[
U(a_{t:t+H}) =
\mathbb E[\text{goal value}]
+\beta\,\mathbb E[\Delta\text{query-relevant belief}]
-\lambda\,\text{action cost}.
\]

必须使用 posterior predictive distribution，而非 MAP outcome partition；允许按预先
冻结的置信阈值 early stop。

Active gate：

- 相对最强非-oracle baseline，成功率提高至少 10 points，或 adaptation actions
  减少至少 20%；
- 至少回收 oracle-vs-history-blind gap 的 30%；
- task-level paired 95% CI 排除 0；
- B=0/1/2/4/8 曲线整体报告，不只挑最好 budget；
- nuisance probes、unsafe probes 与 early-stop calibration 同时报告；
- 正式确认用 5 个全新 seeds，方向一致。

## Track C：主外部因果 benchmark — Symbolic Alchemy

Alchemy 与主命题高度匹配：其 latent causal structure 每 episode 重新采样，明确要求
structure learning、online inference、hypothesis testing 和 action sequencing。
Symbolic Alchemy 去掉视觉导航和连续控制，却保留同一隐藏 chemistry、跨 trial
记忆和离散实验决策，因而应当是 PRP-WM 的**第一个团队外主结果**，不是完成
Meta-Push-T 后才补的装饰性确认。

官方论文在 1,000 个 episodes 上报告 random heuristic `145.7±1.5`、symbolic VMPO
`155.4±1.6`、ideal observer `284.4±1.6`，说明普通 recurrent meta-RL 与随机启发式
之间差距很小，而显式 belief/planning 有很大 headroom。

最小主比较：

1. random heuristic 与 ideal observer 上下界；
2. recurrent context/K1；
3. matched ensemble；
4. Reinfer-K4；
5. Persistent-K4 或 K16。

模型不能读取 chemistry ID、真实 causal graph 或 exact posterior；官方 grammar/exact
belief 只能用于 oracle ceiling 和 scorer。主指标为 episode score，必须同时报告
true action-equivalence Coverage@K、next-outcome NLL/ECE、后半 trials 增益、无效
potion 使用数和 shuffled-transition control。

Alchemy gate：

- 1,000 个冻结 eval episodes、至少 3 个 training seeds；
- PRP 相对 K1 和 matched ensemble，至少恢复 random→ideal score gap 的 20%，即按
  上述官方数值达到约 `173.4`；
- task-level paired 95% CI 排除 0；
- 后半 trials 的得分与试验效率显著优于第一 trial；
- IID chemistry 与 topology/perceptual-map composition holdout 分开报告。

官方仓库已于 2024-06-08 归档，3D 环境依赖 Unity/Linux Docker，工程栈较旧，因此仍
采用限时策略：

1. 用 1 个工作日完成官方 environment + replay smoke；
2. 先做 symbolic/state observation 和官方 task split；
3. 只有 deterministic replay、seed、scoring 都稳定，才接 passive PRP-WM；
4. 不用 test episodes 调结构，只在冻结 validation prior 上调有限超参数；
5. 若环境维护成本超过两天，记录为 engineering no-go；Meta-Push-T 继续作为内部
   bridge，但此时不能声称已有团队外 meta-learning 证据，备选才是 MetaWorld ML10。

MetaWorld ML10/ML45 确实隐藏 task ID 并测试 held-out tasks，但同时引入复杂 skill、
reward exploration 和 manipulation control。它适合作为后续广度验证，不适合第一步
定位 persistent task belief 是否有效。

## 模型与数据到底要不要扩

要扩，但必须按失败层扩，而不是统一放大。

| 规模轴 | 何时扩 | 首轮上限 | 不应如何解释 |
|---|---|---:|---|
| representation model | 进入 pixels/Push-T 时 | LeWM 约 15M 参数量级 | 不能修复错误 likelihood 语义 |
| task/dynamics diversity | Meta-Push-T oracle gate 通过后 | 先 18 train combinations | 重复同分布 frames 不等于 meta-data |
| trajectories | learning curve 尚未饱和时 | 0.1M → 0.4M → 0.8M transitions | 不一次生成最大集 |
| particles K | K4 已有 coverage 且 residual multimodality 可见时 | K=1/4/8 消融 | 随机增宽不能修 proposal holes |
| planning samples | world model 已校准后 | CEM sample curve | 更多 samples 不能修 model bias |

LeWM 论文给出的量级约 15M 参数、单 GPU 数小时，已经足够作为 Push-T 首轮像素
baseline。当前 54K 网络与它相差约 280 倍，但二者解决的不是同一个问题：前者是受控
规则 belief probe，后者是 raw-pixel dynamics。合理的结论是“跨到 pixels 时需要扩大
representation capacity”，而不是“把现有所有模块同时扩大 280 倍”。

数据优先级：

1. 独立 task/dynamics combinations；
2. 同 initial state/action 的 counterfactual pairs；
3. exploratory + expert 的覆盖混合；
4. visual nuisance 与 dynamics 正交的数据；
5. 最后才是同分布轨迹数量。

## 实验统计与防止 benchmark chasing

- 开发阶段 3 seeds，正式结论 5 个未用于调参的新 seeds。
- episode/task 是统计单元，不能把 video frames 当独立样本扩大显著性。
- 尽可能配对 initial state、hidden dynamics、goal 和 planner RNG。
- 报告 hierarchical bootstrap 95% CI、每个 seed 和 worst factor group。
- 每方法最多 12 个 validation tuning runs；test split 只打开一次。
- 每个 phase 只有一个主比较、一个主指标和一个 go/no-go gate。
- 失败后不能同时更换模型、数据和 benchmark。
- 自建 Meta-Push-T 只能作为 bridge，不单独支撑“一般 meta-learning”主张。
- 新 benchmark 必须回答现有三个 benchmark 无法回答的问题；若新增一个，就退休一个
  development benchmark。
- 不把多个异质 benchmark 分数平均成一个总分。
- oracle 没有 headroom、任务不可辨识或 robust baseline 已解决时，停止 benchmark，
  不继续扩大模型。

## 六周执行表

| 周 | 工作 | 结束时必须交付 |
|---|---|---|
| 1 | 内部 likelihood close-out；Symbolic Alchemy 一日 smoke；official Push-T checkpoint smoke | locality pass/fail；两个外部环境可重放 |
| 2 | LeWM vs no-SIGReg reproduction；Alchemy random/ideal/context baselines | representation gate；Alchemy protocol manifest |
| 3 | Symbolic Alchemy K1、ensemble、Reinfer-K4、Persistent-K4 passive belief | score、Coverage@K 与 calibration |
| 4 | Symbolic Alchemy active planning、paired trials；三 seeds | 主外部 Alchemy gate |
| 5 | 枚举 Push-T physical FoVs；Meta-Push-T-Lite state oracle/history-blind/random-probe | Meta-Push-T oracle benchmark gate |
| 6 | 若 gate 通过，做 passive Meta-Push-T pilot；否则扩 Alchemy 新 seeds 或缩小 claim | cross-benchmark decision memo |

六周不是要求所有阶段必须成功。任何 gate 失败都会释放后续算力，并产生一个明确的
可发表/可复现负结论。

## 首轮算力与存储预算

按单张 24GB 级 GPU 粗估：

- 内部 close-out：10–30 GPU-hours；
- Push-T checkpoint + LeWM matched ablation：30–80 GPU-hours；
- Symbolic Alchemy baselines + PRP pilot：20–60 GPU-hours；
- Meta-Push-T state/oracle qualification：10–30 GPU-hours；
- passive Meta-Push-T 三-seed pilot：30–70 GPU-hours；
- 预留 planning evaluation 和失败重跑：30–50 GPU-hours。

六周上限约 130–290 GPU-hours；只有 Alchemy 与 Meta-Push-T oracle/passive gates
给出正面证据后，才批准 pixels Meta-Push-T 的额外 150–350 GPU-hours。数据、
cache、checkpoints 先预留 50–100GB，
并保存数据 manifest 与生成器 commit，而不是把大二进制直接放进本仓库。

## 最终决策规则

六周后只允许四种结论：

1. **继续并扩 pixels：** Symbolic Alchemy 主 gate、Meta-Push-T oracle 与 passive
   K4 gate 均通过；
2. **保留 symbolic claim：** Alchemy 有效、Meta-Push-T 无效；研究定位为因果
   meta-inference，而非通用像素 world model；
3. **缩小为 representation/planning：** LeWM/Push-T 有效，但 K4 或 active 不优于
   强基线；
4. **停止该路线：** oracle headroom 不存在，或 persistent K4 在 matched controls 下
   没有稳定增益。

不允许的结论是“结果还不够好，所以无条件再加数据、参数和 benchmark”。

## 主要外部依据

- [LeJEPA: Provable and Scalable Self-Supervised Learning Without the
  Heuristics](https://arxiv.org/abs/2511.08544)
- [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from
  Pixels](https://arxiv.org/abs/2603.19312)
- [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379)
- [LeWorldModel project page](https://le-wm.github.io/)
- [JEPA-WMs official repository](https://github.com/facebookresearch/jepa-wms)
- [stable-worldmodel official repository](https://github.com/galilai-group/stable-worldmodel)
- [Alchemy paper](https://arxiv.org/abs/2102.02926)
- [Symbolic Alchemy paper](https://arxiv.org/abs/2112.08360)
- [Alchemy official repository](https://github.com/google-deepmind/dm_alchemy)
- [MetaWorld benchmark descriptions](https://metaworld.farama.org/benchmark/benchmark_descriptions/)
