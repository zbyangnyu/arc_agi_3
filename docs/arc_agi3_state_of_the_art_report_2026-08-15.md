# ARC-AGI-3 现状调研报告

**更新日期：** 2026-08-15  
**范围：** ARC-AGI-3 的基准、评测口径、公开代表性系统、已知局限与研究空白。  
**不包括：** 本仓库的方案、路线选择或实施建议。

## 执行摘要

ARC-AGI-3 是一个没有文字规则、没有目标说明的交互式推理基准。智能体面对 64×64 网格与合法动作，必须通过实际交互自行识别对象、机制与目标，再在后续关卡中复用所学规则。它考察探索、世界建模、目标发现、规划与执行，而不仅是从静态输入中输出答案。[官方技术报告](https://arxiv.org/abs/2603.24621)

截至本报告日期，领域的结论不是“某个基础模型已经解决 ARC-AGI-3”，而是：

1. 在已经公开的 25 个 demo 游戏上，**coding-agent harness + 可执行世界模型 + 回放验证 + 规划**已可以接近或达到饱和；社区榜上 Tycho 为 100.0%，Retrodict 为 99.9%，Rodionov 的 verification agent 为 99.0%。这些都是公开集上的自报告成绩，而非 ARC Prize 的独立验证结果。[ARC Prize Community Leaderboard](https://arcprize.org/leaderboard/community)
2. 对未公开环境，公开可核验的信号仍远低得多。官方公布的 GPT-5.6 Sol max 在 ARC-AGI-3 Semi-Private 为 7.78%，在 25 个 Public Demo 为 13.33%。这显示公开集饱和不能外推为真正的泛化能力。[ARC Prize verified results](https://arcprize.org/results/openai-gpt-5-6)
3. 最一致的工程趋势是把 LLM 从“直接选下一动作”变为“生成并维护可执行、可检验的任务模型”。但各高分系统使用的模型、推理预算、模型选择与 run-selection 规则差异很大；当前没有可支持严格横向因果结论的统一消融。

## 1. 基准与评测口径

### 1.1 ARC-AGI-3 测什么

ARC-AGI-3 的游戏由一系列状态帧组成。系统没有得到自然语言规则、目标或奖励说明，必须自行探索，推断世界动力学与完成条件，并把机制迁移到同一游戏后续更难的 level。官方将能力拆为探索、建模、目标设定、规划与执行四项。[竞赛说明](https://arcprize.org/competitions/2026/arc-agi-3)

这与 ARC-AGI-1/2 的静态格子变换不同：动作会改变环境，探索本身也会消耗预算。因此“最终通关”不足以说明系统强；如何以少量动作获取足够信息同样被计分。

### 1.2 RHAE：完成率与动作效率的联合指标

官方使用 Relative Human Action Efficiency（RHAE）。对每个完成的 level，得分取人类基线动作数与智能体动作数之比的平方，并对单 level 的超人效率设上限；再以 level 序号加权聚合，未完成后续关会限制该游戏的最高分。所有真实环境动作——包括探索动作——都计入成本。[RHAE 方法说明](https://docs.arcprize.org/methodology)

直观含义是：若智能体为了解规则走了两倍于人类的动作，即使最终过关，该 level 的未封顶得分也仅约为 25%。因此 RHAE 对“先大量暴力枚举、后找到解法”的方案很不利。

### 1.3 三类不能混用的成绩

| 成绩类别 | 内容 | 可信度与用途 |
| --- | --- | --- |
| Public Demo | 已公开的 25 个游戏 | 便于复现与调试；极易被方案设计、公开痕迹和模型发布日期影响。社区榜明确标记为 self-reported。 |
| Semi-Private | ARC Prize 对部分前沿 API 模型的未公开评测 | 可用于评估基础模型的泛化；官方已验证，但不是离线竞赛提交。 |
| Kaggle 竞赛 | 110 个从未公开的私有游戏，55/55 分别形成 public/private leaderboard | 最接近最终泛化和离线部署约束；评测无网络，要求开源，最终结果尚未完全公布。 |

Kaggle 明确说明其评测使用 110 个从未见过的游戏，分成各 55 个的 Public/Private leaderboard。[Kaggle competition data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data) ARC Prize Community Leaderboard 也明确说明：除 ARC-AGI-1/2 semi-private 外，页面上的成绩是公共集 self-reported，而非独立复核。[社区榜说明](https://arcprize.org/leaderboard/community)

## 2. 当前成绩：应如何阅读

### 2.1 官方可验证的基础模型参考线

| 系统 | 集合 | RHAE | 解释 |
| --- | --- | ---: | --- |
| GPT-5.6 Sol（max） | Semi-Private | 7.78% | 当前官方公布的强基础模型参照；并非针对单一游戏的专用 harness。 |
| GPT-5.6 Sol（max） | 25 个 Public Demo | 13.33% | 官方用于说明跨集合差异的公开集成绩。 |
| GPT-5.6 Terra（max） | Semi-Private | 0.80% | 同系列较弱变体，显示基座模型与推理能力的影响很大。 |

数据来自 ARC Prize 的模型结果页；该页同时显示 Sol 的推理档位从 low 的 0.33% 到 max 的 7.78%，说明单纯提高推理规模也有显著影响。[GPT-5.6 Series results](https://arcprize.org/results/openai-gpt-5-6)

### 2.2 Public Demo 上的公开 harness 代表结果

下表按 ARC Prize Community Leaderboard 与系统作者公开材料整理。除非另行注明，均为 25 个 Public Demo 上的自报告结果；不能视为私有集表现，更不能直接当作竞赛排名。

| 系统 | 报告 RHAE | 报告成本 | 主要机制 |
| --- | ---: | ---: | --- |
| Tycho | 100.0% | $2,986 | 维护游戏级对话历史；按需委派 builder 产生可执行世界模型，经过证伪测试后规划。 |
| Retrodict | 99.9% | $654 | 每帧都记录；假设必须能反向解释已观测历史，才允许继续消耗真实动作。 |
| Rodionov verification agent | 99.0% | $400 | Coding agent 写 Python 世界模型，以记录 transition 进行验证，再规划。 |
| Schema | 98.98%（Opus/Fable）；95.35%（Sol） | 未统一披露 | 可编辑世界程序、完整回放、模型内搜索、失配即废弃计划。未列入社区榜，作者在 blog 中自行披露。 |
| NOOA | 85.1% | $332 | CodeAct agent 构造可复用 NumPy 世界模型 helper，并用 memory/Markdown 跨关保存知识。 |
| OPINE-World | 78.4% | $1,040 | actor 与对抗性 tester 构成类似 CEGIS 的循环，改写并测试可执行游戏引擎。 |
| DreamTeam | 38.1% | $18,000 | 六个固定角色经共享文件工作区分工，运行时构建/修订世界模型。 |
| Read-Grep-Bash | 50.2% | 未披露 | Coding agent 通过搜索和 Python 脚本分析游戏日志。 |

Tycho、Retrodict、Rodionov、NOOA、OPINE-World、DreamTeam 与 Read-Grep-Bash 的描述、成绩和披露成本见[官方社区榜](https://arcprize.org/leaderboard/community)。成本是作者或榜单记录的运行成本，并非在相同 token 价格、硬件、动作预算或运行次数下的标准化测量。

### 2.3 Schema 的分数为何需要特别小心

Schema 是近期最受关注的 harness。它报告：同一 Opus 4.8/Fable 5 模型搭配，通用 Claude Code scratch baseline 为 42.83%，Schema 为 98.98%；Sol 配置报告 95.35%。其核心约束是：

1. 当前世界模型必须是可运行的 `step()` 程序；
2. 使用该程序规划前，必须对所有已记录 transition 进行验证；
3. 只能通过 `commit_actions` 执行真实动作；一旦预测失配，立即丢弃尚未执行的计划。

这是一项重要的 harness 设计证据，但不是已完成的私有集验证。作者明确说明两组分数均为 Public set 自报告、未被 ARC Prize 独立验证；Claude/Fable 成绩还按每个游戏保留两次模型运行中的较高分，Sol 配置也有相似的 pairing/run-selection。因而，42.83%→98.98% 是有价值的同模型配对对照，但不能解释成在严格固定单模型、固定单次运行协议下的通用提升，也不能外推至 Semi-Private。详见[Schema 作者的结果与披露](https://schema-harness.github.io/)。

### 2.4 离线竞赛线不等于云端高分 harness

竞赛评测不允许联网，且要求代码与方法开源。[ARC Prize 2026 rules](https://arcprize.org/competitions/2026) 因此依赖闭源云端 Opus、Fable、GPT 系列的公开 harness 本身不能作为最终 Kaggle 提交。

已公开的 Milestone #1 获胜系统 The Duck 更接近这条约束线：它使用可本地服务的 Qwen 3.6 27B FP8，以 Python REPL 提供图像与文本化网格、helper 函数和动作接口，并在单 GPU/小 token throughput 约束下工作。作者报告在 25 个 Public Demo、每游戏 20 次尝试上的均值为 1.6002 ± 0.4475，且明确指出性能不均匀、受上下文管理与感知限制。[The Duck technical overview](https://tufalabs.ai/research/duck-harness/) 该结果远低于云端 public harness，但其价值在于满足可复现的离线竞赛路径，而非 public 分数本身。

## 3. 技术路线图

### 3.1 图搜索与反应式探索

典型做法是将屏幕预处理为连通组件，哈希状态帧，将动作边加入状态图，并优先扩展仍有未测试动作的 frontier。它的优势是不依赖语言模型推断语义，容易离线运行，也能在小状态空间中稳定找到解。ARC-AGI-3 Preview 的第三名公开方案即属此类：通过 frame processor 与 level graph explorer 实现探索。[Explore It Till You Solve](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore)

局限也明确：状态空间一大、画面含 HUD 或随机性、正确路径须抽象机制而非枚举边时，动作数迅速膨胀。官方对 preview 的复盘也承认，部分早期游戏过于容易被随机/暴力搜索解决。[ARC-AGI-3 Preview learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)

### 3.2 直接 LLM/VLM + 长上下文、工具与记忆

这一路线把当前帧、历史与动作 API 交给多模态推理模型；通过 Python、文件工作区或结构化摘要让模型选择下一动作。Read-Grep-Bash、TELL、OpenClaw 和 Continual Harness 均属于从“直接推理”走向“带工具/记忆推理”的谱系。[社区榜条目](https://arcprize.org/leaderboard/community)

它解决了两个现实问题：64×64 网格与长轨迹会快速耗尽上下文；而外部代码工具可让模型压缩、检索和计算历史。官方技术报告也将 context management 指为关键问题，并记录了允许模型以 Python 选择性检索/转换交互历史的 Duke harness。[ARC-AGI-3 technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

不足是：若“规则”只保留在自然语言笔记或上下文中，系统难以准确检验预测、枚举未来，且一个隐性错误假设容易驱动一连串昂贵动作。

### 3.3 可执行世界模型（当前公开集的主导路线）

可执行世界模型将隐式判断外化为程序。一个典型程序包括状态抽取/渲染、`step(state, action)`、终止或目标判断、以及可供搜索的状态接口。LLM 的职责转为写/改程序、解释反例、调用验证器、选择是否应建模或重建模。

这类系统的共同循环是：

```text
观察历史 → 提出可执行模型 → 回放验证 → 在模型内计划/搜索
                                  ↓
                          执行少量真实动作
                                  ↓
                      预测吻合则继续；失配则修正或换模型
```

Rodionov 的工作是这一方向的早期完整实证：GPT-5.5 high 在 25 个公开游戏上报告 58.12% RHAE、完整解决 15 个游戏；其系统有固定的世界模型接口、验证程序和计划执行器，但无游戏特定逻辑。[论文](https://arxiv.org/abs/2605.05138) 后续消融发现，验证处理在多种模型/推理档位下排名第一，但基座模型与推理档位的影响同样很大，不能简单归因于“只要加世界模型就会成功”。[后续消融](https://arxiv.org/abs/2607.15439)

Tycho 将该路线称为 **active abstraction**：除了形成可执行抽象，还显式决策何时建模、何时修模型、何时绕过模型直接行动。它在相同 Opus 4.8 条件下比较四种编排策略，报告“agent 请求 builder 委派”优于直接推理与固定修复策略；随后以更强模型达到公开集 100%。[Tycho paper](https://arxiv.org/abs/2607.28287)

Schema 的贡献在同一主干上更强调过程强制：状态 grounding 与机制发现共用一个可编辑程序；历史反证的优先级高于当前计划；规划只在通过验证的程序中进行。它并不等价于“写出程序即可”，而是强调**表示、机制和实验设计必须共同迭代**。[Schema](https://schema-harness.github.io/)

### 3.4 多智能体/工作区优化

DreamTeam 以共享工作区保存世界模型、计划、假设和反例，把“hypothesize、probe、plan、strategize”等角色分给固定 agent。论文报告在同一 25 游戏公开协议下以更少环境动作把此前 SOTA 从 36% 提升到 38.4%。[DreamTeam paper](https://arxiv.org/abs/2605.09650)

该路线的优势是任务分解与可审计工件；缺点是调用和协调成本高。以社区榜记录为例，DreamTeam 的 $18,000 报告成本远高于后来单一 coding-agent world-model 系统的数百至数千美元，因此它尚未成为公开集上最有效率的主导形态。[社区榜](https://arcprize.org/leaderboard/community)

### 3.5 显式探索—验证—规划（epistemic agent）

AERA 将行为分为 EXPLORE、VERIFY、PLAN 三阶段：探索时选择预期降低不确定性的动作，验证阶段做少量证伪动作，再进入规划；出现意外观测就退回探索。其轻量模型实验主要用于说明“探索纪律”可以胜过随机/不探索基线，而非建立 SOTA。[AERA paper](https://arxiv.org/abs/2605.25931)

这条路线的重要性在于指出：探索并非“动作越多越好”，而是一个信息收益与动作效率的 trade-off。其局限是后验不确定性以语言输出长度等弱代理估计，实验模型很小，且论文对 public 集 shortcut 的批评仍需独立复核。

### 3.6 学习式视觉/RL/持续学习

Preview 竞赛的第一名 StochasticGoose 是 CNN action-learning agent：学习哪些动作会改变帧，比随机探索更有效，最终在三个私有 preview 游戏完成 18 个 level、得分 12.58%。[官方 preview 复盘](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)

此后，Vision–Continual Learning v1 报告在 Public Demo 取得 63.1%，通过跨游戏/跨 level 持续更新模型权重；但其报告成本为 $4,788，且同样是 self-reported public 分数。[社区榜](https://arcprize.org/leaderboard/community) 现有公开证据尚不足以说明端到端学习式世界模型已在未知私有游戏上优于 coding-agent harness。

## 4. 为什么公开集分数快速接近 100%，但现状仍不清楚

### 4.1 公开集与私有集不是同一问题

Public Demo 的游戏、轨迹和大量分析已被公开，harness 可在这些游戏上反复开发。公开 100% 所展示的是“该流程在公开游戏上能做到近人类甚至超人动作效率”，不自动证明对未见机制的泛化。Schema 作者自己仅声称 public-set 成绩，不对 Semi-Private 作数值外推。[Schema disclosure](https://schema-harness.github.io/)

此外，Tycho 与 Rodionov 的高分跟随于其使用的后期模型；Rodionov 的消融论文明确提醒，后续强模型的公开集近饱和应理解为 public-set saturation，未见集表现尚未测试。[Rodionov ablation](https://arxiv.org/abs/2607.15439)

### 4.2 评测协议的选择效应

不同系统可能有不同的：模型路由、重跑次数、每游戏预算、失败后 fallback、动作上限、提示和公开游戏开发集。RHAE 又对动作数取平方并有上限；一旦大部分 level 饱和，分数对“完成剩余困难关”和“评测选择规则”格外敏感。

因此，阅读一项公开分数时至少应检查：

- 是否是单一固定模型、固定提示、固定单次运行；
- 是否以同一协议做了 direct-agent 基线和消融；
- 是否公开了逐游戏 replay、所有 run 及模型调用预算；
- 模型是否可能在预训练或后训练阶段接触过公开游戏/讨论；
- 是否在未见环境、离线硬件限制下复现。

### 4.3 public games 的 shortcut 风险

官方在 Preview 后已承认部分早期游戏对暴力搜索过于友好。[官方复盘](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings) 一篇 AERA 预印本进一步声称 25 个 public 游戏都存在非智能搜索或库层漏洞的可达路径，因此认为 private 评测才是有效测试；这是作者的审计结论，并非 ARC Prize 的官方判定，仍应独立验证后再据此评价基准。[AERA paper](https://arxiv.org/abs/2605.25931)

## 5. 当前共识、分歧与未解问题

### 较强共识

1. **动作层面的在线探索不可省略。** 静态 CoT 或单次视觉理解远不够；需要记录、检验并利用交互历史。
2. **外部可检验工件很有价值。** 代码、状态表、历史日志和 replay 可减少上下文遗忘，让模型内搜索替代真实试错。
3. **验证比“生成一个看似合理的解释”更关键。** 高分 coding-agent 系统普遍要求回放历史或用反例触发重建模。
4. **模型能力仍是主要变量。** 同一或相似 harness 换更强模型、提高推理档位会显著改变结果，不能只把提升归因给外壳。

### 仍有实质分歧

1. **高 public 分究竟是怎样的能力证据？** 一种观点认为这是 harness 把模型的潜在科学推理能力有效释放；另一种观点认为公开游戏、选择性重跑与当前模型的后发性使其缺少 held-out 结论。
2. **世界模型应是自由 Python、受限 DSL，还是神经 latent？** 自由代码表达力强且便于搜索，但难安全验证；DSL 易审计但覆盖有限；latent world model 可能更高效，但难作为严格 evidence。
3. **多智能体分工是否值得成本？** 共享工作区和 adversarial testing 能增加稳健性，但目前单 coding agent 的公开成本/分数比通常更好。
4. **如何在离线约束下取得同等效果？** 当前显眼的 public 饱和系统多依赖闭源云模型和大量推理；Kaggle 单 GPU/无网限制下，最强可复现离线能力仍是开放问题。

### 下一阶段真正应观察的证据

- 110 个未公开竞赛游戏的最终 Public/Private leaderboard 与可复现开源代码；
- 固定模型、固定预算、固定单次运行下，harness 的逐模块消融；
- 公开 games 之外的全量 replay、失败案例与跨随机种子方差；
- 同一方法在后发布、无公开讨论痕迹的新环境上的结果；
- 端到端可离线运行的系统与云端 coding-agent 的能力—成本曲线。

## 6. 结论

ARC-AGI-3 的公开研究在 2026 年夏天发生了一个清晰转向：从“让模型直接看图并决定动作”，转为“让模型外化、验证并使用可执行任务理论”。Schema、Tycho 和 Rodionov 系统是这一转向最具代表性的实例；它们的区别更多在世界模型的编排、验证纪律与模型选择，而不是是否使用了程序化模拟。

然而，现阶段不能据此宣布 ARC-AGI-3 已被解决。近 100% 的结果集中在公开 demo，且大多为自报告；官方可验证的 Semi-Private 基础模型成绩仍在个位数。最终应以未见、离线、可复现的竞赛评测为准。现在最可靠的总体判断是：**harness 设计已被证明会极大改变公开游戏表现；这种收益在真正未见环境中的幅度仍是未决问题。**

## 参考来源

- [ARC-AGI-3 technical report（ARC Prize）](https://arxiv.org/abs/2603.24621)
- [ARC-AGI-3 scoring methodology（ARC Prize）](https://docs.arcprize.org/methodology)
- [ARC Prize Community Leaderboard](https://arcprize.org/leaderboard/community)
- [ARC Prize GPT-5.6 verified results](https://arcprize.org/results/openai-gpt-5-6)
- [ARC Prize 2026 / ARC-AGI-3 competition](https://arcprize.org/competitions/2026/arc-agi-3)
- [Kaggle ARC-AGI-3 competition data description](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data)
- [Schema report and artifacts](https://schema-harness.github.io/)
- [Tycho](https://arxiv.org/abs/2607.28287)
- [Executable World Models for ARC-AGI-3](https://arxiv.org/abs/2605.05138)
- [Coding-agent ablation: models, simplification, verification](https://arxiv.org/abs/2607.15439)
- [DreamTeam / Workspace Optimization](https://arxiv.org/abs/2605.09650)
- [The Duck: Milestone #1 winner](https://tufalabs.ai/research/duck-harness/)
- [Graph-Based Exploration / Explore It Till You Solve](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore)
- [AERA: Explore Before You Solve](https://arxiv.org/abs/2605.25931)
