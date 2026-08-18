# 主动猜想—验证实验方案

**日期：2026-08-15**  
**状态：提案，尚未替换既有路线图。**  
**范围：** 验证“代理能否通过主动交互提出、区分、证伪并修复机制猜想”；不以
ARC-AGI-3 分数为近期目标。

## 1. 核心问题与结论标准

本方案只检验一个主命题：

> 面对每 episode 重采样、未公开的规则，具有可执行多假设与反例验证的代理，能否比
> 相同推断器下的被动或随机试探，以更少的**真实环境动作**获得足以完成新初态任务的
> 机制知识？

这里的“主动”不是泛指采取很多动作，也不是训练一个更强 policy。它必须同时满足：

1. 代理显式提交多个可执行或可检验的猜想；
2. 后续动作对至少两个仍存猜想给出不同预测；
3. 真实反馈被记入不可修改的历史；
4. 反例导致猜想被淘汰、修复、拆分或降低置信度；
5. 最终评测在同一隐藏机制、但**不同初态**下进行，避免把动作轨迹记忆误当作机制学习。

若只满足“最终能完成”，但没有 1–4 的轨迹证据，不能声称完成主动科学式交互。

## 2. 总体路线：三个逐层加难的 gate

```text
P0 生成式微型规则环境：验证主动因果链与评测本身
 │
 ├── 不通过：修环境可辨识性、猜想契约或 verifier；不上外部 benchmark
 │
 ▼
P1 XLand-MiniGrid：在外部、可组合 ruleset 上验证
 │
 ├── 不通过：定位为状态落地、猜想语言、实验选择或规划问题
 │
 ▼
P2 DeepMind Alchemy：独立的潜在因果结构确认
```

P0 是内部诊断，不能作为泛化主张；P1 是第一个外部结果；P2 才用于确认结果不只属于
grid/DSL 世界。每阶段结束只能作 **go / no-go / 缩小主张** 三种决定。

## 3. P0：生成式微型规则环境（主预试验）

### 3.1 环境约束

新增一个独立的、确定性的 `HypothesisGrid` 环境族，而不是将现有 RuleGrid 结果包装成
“自由发现”。环境应保持小到可穷举审计：

| 项目 | 冻结建议 |
| --- | --- |
| 画布 | 7×7 或 9×9 离散网格；categorical grid observation |
| 动作 | 移动、朝向/拾取、相邻交互三类；每一步有固定动作成本 |
| 对象 | player、2–4 个可交互对象、0–2 个门/标记物；颜色与形状为视觉 nuisance |
| 隐藏机制 | 每 episode 从 2–3 条 production rules 组合采样；整个 discovery/evaluation episode 固定 |
| 机制 DSL | 条件（相邻/持有/开关状态）→ 变换、移动、生成、门状态变化；显式 terminal predicate |
| 观测 | 仅当前 grid、合法动作、成功/失败/关卡终止；不暴露 DSL、对象 ID、规则、目标 predicate 或 simulator state |
| 试探后评测 | discovery phase 后 reset 到新初态，保留隐藏机制；evaluation phase 只按通关和动作数计分 |

每个生成任务必须在发布前由 generator 审计：

* 至少存在两个机制在前缀历史上均一致；
* 至少一个可达动作会使这些机制产生不同的可观测预测；
* 区分动作不是无意义 self-loop，且不会不可逆地毁掉全部完成路径；
* oracle mechanism planner 能在评测初态成功；
* 隐藏规则的等价类而非语法表面形式，是 ground truth 评分单位。

这避免了两个常见伪结论：环境本来不可辨识，或正确动作恰好等于随机动作。

### 3.2 代理契约

代理每一回合可调用不计环境动作的本地工具；唯一改变环境的接口是 `commit(action)`。

| 工件 | 必填内容 |
| --- | --- |
| `history.jsonl` | 只追加的 observation、action、terminal/event 记录 |
| `hypotheses/` | 至多 K=4 个候选；每个含 state schema、partial `step`、partial `goal`、置信度与版本号 |
| `backtest` | 对每个候选、逐 transition 报告预测覆盖率、匹配/失配与未声明部分 |
| `discriminator` | 针对候选对给出会导致预测不同的可达动作及预测差异 |
| `plan` | 仅在猜想达到使用条件后生成；计划附带逐动作预测 |
| `ledger` | 每次修改记录触发它的反例、被淘汰/修复的候选以及未解决的不确定性 |

允许部分预测：候选可以对 HUD 或不相关对象 abstain，但必须声明 coverage；不能用“全都
unknown”通过 backtest。`commit` 后若当前计划的下一状态预测在声明 coverage 内失配，
计划立即失效并回到猜想阶段。

### 3.3 四个处理组

四组使用相同的 observation、动作预算、假设语言和底座模型；只改变被检验变量。

| 组别 | 猜想/验证 | 动作选择 | 用途 |
| --- | --- | --- | --- |
| O（oracle） | 精确 DSL version space | 最优区分动作 | 环境与指标 headroom；不计入方法结论 |
| A（active） | 提议的多猜想 + backtest + 修复 | 最大化候选预测分歧且考虑到达成本 | 主处理 |
| R（randomized） | 与 A 完全相同的猜想与更新 | 在 A 的可达候选动作中随机选 | 隔离主动选实验的价值 |
| P（passive） | 与 A 完全相同的猜想与更新 | 固定、配对的探索 prefix | 隔离“主动交互”与“只是看更多数据” |
| G（graph） | 无机制猜想 | state-graph frontier / 新颖性优先 | 强无模型探索基线 |

R 和 P 的关键要求是共享 A 的 hypothesis builder；否则“主动优于随机”会混入模型调用、
prompt 或表示差异。每一 hidden mechanism 都在各处理组配对运行。

### 3.4 主要指标

**主指标（必须同时报告）**

1. **Mechanism-equivalence identification @ budget**：在 0/1/2/4/8 个 discovery
   动作后，候选集合是否含有与真实机制在评测可达域等价的程序。
2. **Evaluation success @ action budget**：不同初态的完成率，以及完成动作数。
3. **Discovery action efficiency**：达到既定 mechanism-equivalence 或达到足够
   可靠计划所需的真实 discovery 动作；与 O、R、P、G 比较。
4. **Counterexample repair**：发生预测失配后，下一版本是否消除该失配且不降低此前
   已解释 transition 的覆盖率。

**辅助指标**

* backtest precision、coverage、transition log score/Brier；
* 动作的预期 candidate disagreement 与实际 information gain 的校准；
* 猜想多样性、过早塌缩率、无效/不可达实验率；
* 计划的首步与整段预测一致率；
* token、模型调用、墙钟时间、每个环境动作的内部计算量。

机制正确性以行为等价类评估，不要求模型猜中 generator 的任意变量/函数名。

### 3.5 拆分与信息边界

* **开发：** 规则原子、颜色、形状、布局均可见；最多用于调试。
* **组合 OOD：** 原子规则均出现过，但其 2–3 条组合未在开发中出现。
* **规则 OOD：** 留出一种条件—效果组合或其组合深度；单独报告，不混入组合 OOD。
* **视觉 OOD：** palette、符号外观、对象位置独立重采样，禁止作为机制 ID 的捷径。
* **私有 test manifest：** 仅在 prompt、候选 DSL、预算、模型和 early-stop policy 冻结后
  打开；每条 run 记录版本 hash。

环境适配层不得把 generator 代码、对象命名、规则文本或 oracle state 放进 agent
filesystem/context。代理运行在受限目录，logger 与 evaluator 在外部进程。

### 3.6 P0 gate

先做不含目标方法的 **环境资格 pilot**（20 个开发机制），只用于冻结阈值与预算：

* O 必须在固定短预算内 100% 到达真实机制等价类且完成评测任务；
* O 的区分动作必须在配对机制上明显优于 uniform/graph，才能证明环境有主动推断
  headroom；
* 所有任务须满足可辨识性、可达性和 leakage 审计。

资格通过后冻结测试 manifest，才比较 A/R/P/G。P0 的 go 条件是：

* A 相比 R 与 P，在相同或更少 discovery 动作下提高评测成功；
* A 的 paired action-efficiency 差异的 95% bootstrap CI 下界大于 0；
* A 的改善不能只来自更多内部重试：完整 token/call ledger 必须公开；
* 至少一个组合 OOD 与一个视觉 OOD slice 保持正向效应。

若 A 仅优于 G 而不优于 R/P，结论只能是“有机制猜想有帮助”，不能声称主动实验选择
有效。若 O 无优势，先重做环境生成与 action language，不训练代理。

## 4. P1：XLand-MiniGrid 外部预试验

XLand-MiniGrid 以 deterministic production rules 和单一 goal 组成 ruleset；规则可组成
任务树，初态位置可随机化，并提供 `trivial-21k` 快速迭代基准和 ruleset generator。
这使它适合测试 P0 结果能否离开自建 DSL。原始 API 的 `timestep` 含 internal state，
所以必须经严格 adapter。

资料来源：[XLand-MiniGrid rules/goals 与 benchmarks](https://github.com/dunnolab/xland-minigrid)。

### 4.1 三个由易到难的 protocol

| 阶段 | 给代理的信息 | 回答的问题 |
| --- | --- | --- |
| P1a contract | 规则 DSL 与目标文本可读，只有初态变 | 后端、planner、日志与 evaluator 是否正确 |
| P1b hidden dynamics | grid、合法动作、公开的目标视觉/goal contract、sparse terminal/reward；隐藏 transition ruleset | 能否主动发现 transition rules 并迁移到新初态 |
| P1c hidden dynamics + goal | 只给 grid、合法动作、terminal signal；规则与 goal 均隐藏 | 状态落地、机制发现与 goal inference 的联合难度 |

P1a 是软件烟测，不能作为能力结果；主结论来自 P1b，因为它将 goal inference 从
transition-rule discovery 中剥离。P1c 只在 P1b 通过后开放。

### 4.2 Adapter 与拆分

* 对代理仅输出 rendered/structured grid、legal actions、`done`、稀疏 success/failure；
  禁止输出 `timestep.state`、ruleset、rule tree、goal predicate、seed 或环境源码。
* adapter 在独立包/进程运行，添加一条测试：改变 ruleset 的隐藏字段不会改变代理输入。
* 用 `trivial-21k` 只作开发与 smoke；正式测试由官方 generator 用冻结 seeds 生成，保存
  不公开的 manifest。
* 固定 9×9 小布局，先限制到 depth 2–3 的 rule trees 与少量对象；之后才扩大深度、
  grid 与视觉多样性。
* discovery/evaluation 两阶段保持同一 ruleset、重采样初态；这与 P0 对齐。

### 4.3 对照、样本量与 P1 gate

复用 P0 的 O/A/R/P/G 对照。每个 run seed 评测至少 200 个冻结 ruleset–initial-state
对，三个独立 agent/run seeds；以 ruleset 为配对统计单位，不能把单个 frame 或动作当
独立样本。

进入 P2 的条件：

* P1b 中 A 对 R 和 P 都有正的 paired action-efficiency CI 下界；
* 新初态 evaluation success 的改善与 mechanism-equivalence/预测改善同方向，排除
  “只学到探索技巧”解释；
* OOD rule-tree 组合、布局与 palette 三个 slice 不出现系统性反转；
* 没有 adapter leakage，所有轨迹可回放，模型、prompt、预算和 rerun policy 已冻结。

若 P1a 成功而 P1b 失败，优先审计状态抽取或猜想语言；若 P1b 成功而 P1c 失败，结论
限于 hidden dynamics，不宣称 goal discovery。

## 5. P2：DeepMind Alchemy 独立确认

Alchemy 每 episode 重采样潜在因果结构，原本就针对 latent-state inference、有效探索、
实验与规划设计，是对主命题最匹配的非网格确认集。

但它是 Unity/Docker 环境，官方仓库已归档，要求 x86-64 SSE4.2，且仅正式支持 Linux；
因此先做 48 小时环境资格检查，不把兼容性排错算进算法开发周期。

资料来源：[Alchemy 环境](https://github.com/google-deepmind/dm_alchemy)、
[Alchemy 论文](https://openreview.net/forum?id=eZu4BZxlRnX)。

### P2A 平台资格（限时 48 小时）

1. 在隔离的 Linux x86-64 Docker 环境启动官方 image；固定容器 digest 与 wrapper commit。
2. 复现官方示例 reset/step；核对 observation、action、episode resampling 和 seed。
3. 实现只暴露允许观测的 adapter；环境 latent chemistry、图结构和 oracle features 仅供
   离线 evaluator 使用。
4. 运行 random、固定 policy 与 oracle-state diagnostic，证明任务不是不可完成或泄漏。

平台不通过则记录为环境/依赖 no-go，不以本地兼容 hack 延长实验；P1 仍可独立产出。

### P2B 研究协议

先使用 Alchemy 的结构化观测和离散动作，隔离机制推断；像素感知后置。沿用
discovery/evaluation reset，使同一 chemistry 从试验阶段转移到新 stone/potion 初态。
主对照仍为 A/R/P/G，额外报告 oracle-chemistry planner 上界与 history-blind planner。

结论标准与 P1 相同，并另报告在未见 latent chemistry composition 上的 success、
试探动作数与 action regret。P2 只要没有主动效应，就不应将 P1 成功外推为一般因果
发现能力。

## 6. 共同的实现与验证顺序

| 周期 | 交付物 | 允许的工作 | 不允许的混杂 |
| --- | --- | --- | --- |
| 第 1 周 | P0 generator、oracle、manifest/audit、adapter boundary tests | 环境与准确 oracle | 同时训练大模型或接入 ARC |
| 第 2 周 | A/R/P/G 共用 hypothesis contract、trajectory ledger、回放 evaluator | 先跑 20-task dev pilot 冻结 gates | 以最终成功替代猜想/反例证据 |
| 第 3 周 | P0 3-seed frozen evaluation 与 go/no-go 报告 | 只修 P0 定位到的层 | 在同轮更换环境、DSL 与 agent |
| 第 4 周 | XLand P1a/P1b adapter、泄漏测试、oracle headroom | 外部环境 smoke 与小深度 rulesets | 向代理暴露 `timestep.state` / ruleset |
| 第 5–6 周 | P1 frozen evaluation、OOD 分面、结果报告 | 单变量修复并重跑 | 用公开 test ruleset 反复调 prompt |
| 之后 | P2A/P2B Alchemy | 仅在 P1 go 后开始 | 因 P2 依赖失败而修改 P1 结论 |

每次运行须产出：环境/manifest hash、模型版本、prompt hash、预算、动作序列、完整
hypothesis diff、backtest、预期与实际结果、计划中断原因、token/call/cost ledger。
所有图表以 mechanism/ruleset 为统计单位，并展示每个配对任务的 delta，而不是只给均值。

## 7. 预注册的失败解释表

| 现象 | 最可能定位 | 下一步 |
| --- | --- | --- |
| O 也无主动优势 | 任务不可辨识或区分动作不可达 | 重做 generator / action contract |
| A 的猜想回测好但 eval 失败 | state schema、goal 或规划错误 | 分开审计 partial state/goal/planner |
| A 优于 G，但不优于 R/P | hypotheses 有价值，action choice 无价值 | 修 discriminator/value-of-information，不改环境 |
| A 优于 R/P，只在开发集 | prompt/规则族过拟合 | 冻结 adapter，扩大组合/规则 OOD |
| P1a 成功、P1b 失败 | hidden rule induction 或状态落地失败 | 缩小 XLand grammar，检查 hypothesis language |
| P1b 成功、P1c 失败 | goal inference 是独立瓶颈 | 将 claim 限于 dynamics identification |
| Alchemy 失败但 P1 成功 | grid DSL 到 latent chemistry 不迁移 | 不声称一般主动因果发现 |

## 8. 本方案的边界

本方案刻意不把以下事情塞进第一轮：端到端 RL、图像 encoder 预训练、JEPA、长期跨游戏
记忆、大语言模型微调、ARC-AGI-3 scorecard、连续机器人控制。它们可以成为后续增强，
但会破坏对“猜想—验证本身是否有效”的归因。

同样，P0 的 oracle 只能用于环境资格与上界；不能向 A/R/P/G 公开机制 ID、对象角色、
规则文本、solver trace 或 generator 代码。只有在这个边界被自动审计后，主动交互结果
才有解释力。
