# ARC-AGI-3 技术现状调研报告

**更新日期：2026-08-15**  
**范围：** 外部公开资料；不评价或讨论本仓库的方法。  
**结论先行：** ARC-AGI-3 当前最显著的进展不来自重新训练一个专用网络，而来自
把前沿模型置于一个严格的“观察—建模—证伪—规划—执行”harness 中。公开的 25 个
游戏已被这类系统接近或达到饱和；但尚无相应的、ARC Prize 独立验证的 Semi-Private
高分。因此，“公开集接近 100%”是很强的系统工程结果，**不是**已证明的未知环境泛化。

## 1. 基准与阅读口径

ARC-AGI-3 是交互式、回合制的抽象游戏集合：代理收到 64×64 色彩网格和合法动作，
没有规则、目标或对象说明；它需要在交互中探索、形成世界模型、推断目标并规划。
官方把它定位为对探索、建模、goal-setting、规划/执行四种能力的联合测试。

成绩使用 **RHAE (Relative Human Action Efficiency)**：每个已完成关卡的分数为

```text
(人类上中位数首次游玩动作数 / 代理动作数)^2
```

单关上限为 1.15，后续关权重更高；未完成所有关卡则游戏分数存在上限。因此，RHAE
同时惩罚“没有通关”和“靠大量试错通关”。它并不衡量代理的推理 token、模型调用、
墙钟时间或人工开发成本。

资料来源： [评分方法](https://docs.arcprize.org/methodology)、
[官方技术报告](https://arxiv.org/abs/2603.24621)。

### 三种不能混为一谈的评测

| 层级 | 含义 | 当前可作出的结论 |
| --- | --- | --- |
| Official Semi-Private | ARC Prize 对未公开环境进行的模型评测 | 最接近“未知环境的首次适应”证据 |
| Kaggle Private / Competition | 离线、本地硬件、开源代码条件下的竞赛评测 | 最接近可提交方案的证据 |
| Public Demo（25 游戏） | 游戏、回放和开发反馈均可获得 | 最适合调试 harness；不能单独说明泛化 |

ARC Prize 社区榜明确说明：社区项通常为作者自报告的 Public 成绩；官方不独立验证它们。
因此，下文的 Public 分数均应读作“已披露实现的公开集表现”，而不是官方 SOTA。

资料来源：[社区榜及其验证说明](https://arcprize.org/leaderboard/community)、
[竞赛规则](https://arcprize.org/competitions/2026/arc-agi-3)。

## 2. 当前成绩版图

### 2.1 已验证的基线：仍远未饱和

截至本报告日期，官方展示的最强已验证 ARC-AGI-3 Semi-Private 单模型结果为
**GPT-5.6 Sol / Max：7.78%**；同一设置在 Public 为 13.33%。同系列的 effort
缩减会迅速下降，例如 Sol/High 为 2.15%，Terra/Max 为 0.80%。这表明更强模型和
测试时推理有实质作用，但裸模型或极简外壳仍不能可靠完成陌生游戏。

资料来源：[ARC Prize 的 GPT-5.6 验证结果](https://arcprize.org/results/openai-gpt-5-6)。

### 2.2 Public harness：接近饱和，但证据边界明确

下表是当前社区榜和作者公开材料中较有代表性的系统。成本为作者/社区榜披露的整套
Public 评测成本，口径并不完全统一，只能粗略比较。

| 系统 | Public RHAE | 方法摘要 | 证据状态 |
| --- | ---: | --- | --- |
| Tycho | 100.0% | 可执行世界模型；按需委托 builder、验证、规划或绕过模型 | 社区榜，公开集 |
| Retrodict | 99.9% | 所有规则假设必须反推解释已记录 history，才可花真实动作 | 社区榜，公开集 |
| Schema | 98.98% | 统一的 state/机制程序；回测、BFS、失配即弃计划 | 作者自报，公开集 |
| Rodionov verification | 99.0% | coding agent 写 Python 世界模型，重放验证并规划 | 社区榜/论文，公开集 |
| NOOA | 85.1% | CodeAct + 可复用 NumPy world-model 工具 + memory | 社区榜，公开集 |
| OPINE-World | 78.4% | Actor + CEGIS model-builder + exact replay + 规划 | 社区榜，公开集；作者披露有服务限流中断 |
| DreamTeam | 38.1% | 六角色共享工作区、显式世界模型与失败路由 | 社区榜，公开集 |

Tycho 报告中 GPT-5.6 Sol 与 Opus 5 均完成 25/25 游戏、183/183 关卡并拿到 100 RHAE；
但模型晚于公开游戏，且其作者也只报告 Public 结果。其重要性在于验证了“选择何时
建模、何时修复、何时直接行动”往往比提高单次 transition 拟合更重要。

资料来源：[ARC Prize 社区榜](https://arcprize.org/leaderboard/community)、
[Tycho 论文](https://arxiv.org/abs/2607.28287)、
[Rodionov 消融报告](https://arxiv.org/abs/2607.15439)。

### 2.3 竞赛约束下的路线：Duck

Public 集上的云端 coding agent 不可直接等同于竞赛方案。ARC Prize 竞赛要求无网络、
代码开源，实际 Kaggle 运行受本地硬件和模型大小限制。

Tufa Labs 的 **Duck** 是 ARC-AGI-3 Milestone #1 获胜方案：使用 Qwen 3.6 27B FP8、
本地 vLLM、Python REPL 和图像/文本双重网格观察。它在 Public 25 游戏、每游戏 20
次尝试中报告平均 1.6002±0.4475 RHAE；这一绝对分数很低，却说明在单 GPU、本地开源
模型约束下，性能与云端前沿 coding agent 间仍有巨大差距。该项目同时公开了 Kaggle
notebook、源码、运行轨迹与查看器，因而是研究可复现提交工程的最佳入口之一。

资料来源：[Duck 技术说明](https://tufalabs.ai/research/duck-harness/)、
[源码与完整运行包](https://github.com/Tufalabs/duck-harness)。

## 3. 方法谱系

### 3.1 系统搜索、图探索与轻量 RL：必要的强基线

最早的有效方案并非 LLM 推理，而是以图为中心的探索：将 frame 作为节点、动作作为边，
记录已试动作，优先探索距离最近的未测试 state-action。视觉模块可做连通域分割、HUD
屏蔽和 click 优先级。它避免重复试错且不依赖训练；代价是状态空间增大后迅速失控，
也无法主动判断“哪一个实验最能揭示机制”。

Preview 阶段的官方第一名 StochasticGoose 用一个小 CNN 预测会引起 frame 变化的动作；
第二名 Blind Squirrel 使用状态图、回溯标注距离并训练 ResNet18 value model。纯图搜索
的后续研究在 preview 私有评测上报告 6 个游戏中 52 关的中位 30 关，说明系统状态追踪
本身是很强、必须保留的基线，而不是过时方案。

资料来源：[ARC Prize Preview 复盘](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)、
[Graph-Based Exploration 论文](https://arxiv.org/abs/2512.24156)。

### 3.2 Coding-agent + REPL：把视觉游戏变成可操作的编程问题

这一层为模型提供结构化 observation history、Python 变量、图像处理与辅助函数，模型
可以检查、计算和保存中间结论，而非仅在对话上下文中盯着像素。Duck 是本地化版本；
Read-Grep-Bash、早期 Rodionov 系统和 NOOA 则代表云端 coding-agent 版本。

其价值不是“给模型一个计算器”这么简单，而是提供三种持久外部状态：可查询的历史、
可执行的假设、可重用的工具。其风险也很直接：工具、prompt、文件命名和可访问资源会
成为信息泄漏或针对公开游戏的隐式先验通道，必须隔离游戏源码、网络和非预期文件。

### 3.3 可执行世界模型：当前主导路线

几乎所有高分 Public harness 都收敛到相同骨架：

```text
frame/history → 状态与对象假设 → 可执行模型
           ↑                         ↓
真实反例 ← 单步验证执行 ← 模型内搜索/计划
```

典型程序至少应明确三件事：状态如何从 frame 落地、`transition(state, action)` 如何演化、
什么状态构成目标或关卡完成。其收益是：一旦机制正确，后续候选动作可在模拟器内免费
搜索；跨关卡继承的是规则而非具体动作序列。

* **Rodionov** 将 Python world model、历史验证、简化和 planner 串起；后续消融发现，
  更强底座模型和更长 reasoning 的作用很大，但要求回放验证的完整系统在四组
  model-effort 条件中均排名第一。
* **Schema** 进一步把 state grounding 和 mechanism discovery 合并为同一个可编辑程序，
  强制 `step()`、全 history 回测和 prediction-error 后丢弃计划。它还要求代理使用能
  区分候选理论的实验。
* **Tycho** 的贡献是把模型维护本身变成元决策：建模、修复、使用或绕过模型。其消融中，
  “自动修复”有更准确的 transition reproduction，却低于 actor-requested delegation，
  说明模型拟合精度不是唯一瓶颈。

资料来源：[Rodionov 原始论文](https://arxiv.org/abs/2605.05138)、
[Rodionov 消融](https://arxiv.org/abs/2607.15439)、
[Schema 技术说明](https://schema-harness.github.io/)、
[Tycho](https://arxiv.org/abs/2607.28287)。

### 3.4 CEGIS、多假设与主动实验：OPINE-World

OPINE-World 是主流“单一 worldcoder”之外最清楚的一条变体。一个 actor 玩游戏，另一个
独立的合成代理改写 `game_engine.py`；候选模型只有精确复现所有 transition、并经过
对抗子代理审查后才获准使用。每一步计划在真实环境中复核，反例触发下一轮 synthesis。
它用 ontology error 作为廉价诊断，指向当前对象类型无法解释的现象；正确性仍由 exact
replay 决定。该系统报告 20/25 游戏、160/183 关、78.4 RHAE，但五个游戏受到限流影响，
作者明确把结果视为初步。

资料来源：[OPINE-World 源码与方法说明](https://github.com/david-courtis/opine-world)。

### 3.5 多智能体工作区与持续记忆：有用但非充分

DreamTeam 将 hypothesis、model building、planning、probe、strategy 和 failure routing
分配给六个固定角色，在共享文件工作区中累积“可检查的外部资产”。早期结果为 38.4 RHAE，
平均动作数比匹配基线少 31%；社区榜收录的公开 release run 为 38.1%。NOOA 则以 Python
对象、类型接口、REPL 与持久 memory 组织 CodeAct，报告 85.1%。

现状说明：多角色和 memory 能缓解上下文丢失、分工和自我修正问题，但没有可靠的程序
验证与模型内规划时，它们并不足以达到公开集顶端；反之，最强世界程序系统通常也都保留
某种持久 workspace。

资料来源：[DreamTeam 论文](https://arxiv.org/abs/2605.09650)、
[DreamTeam 源码](https://github.com/NVIDIA/dream-team)、
[NOOA 项目](https://github.com/NVIDIA-NeMo/labs-OO-Agents)。

## 4. Schema 的单独评估

Schema 是当前最有传播力的 Public harness 案例，但应按其自身披露理解：

* Opus 4.8 + Fable 5 的 98.98% 是同一模型配对下相对 Claude Code scratch snapshot
  42.83% 的对照；两者差 56.15 个百分点。
* 对 GPT-5.6 Sol 的 95.35%，其首轮为 Sol/xhigh；低于 80 的游戏再以 Sol/max 重跑，
  取每游戏更高分。Claude 配对也使用 Opus 4.8 后备 Fable 5 的固定规则。
* 作者明确声明两个分数都是 Public、自报告、未获 ARC Prize 验证；也明确指出
  Sol 的 13.33% Public 与 7.78% Semi-Private 不能用于外推 Schema 的未知集分数。

因此，Schema 可被视为目前对“强模型 + 强制程序化科学循环”最清楚的公开证据之一；
不能被视为 ARC-AGI-3 已被攻克，亦不能与 Official Semi-Private 分数放到同一排行榜。

资料来源：[Schema 原始披露](https://schema-harness.github.io/)。

## 5. 目前已形成的共识

1. **Harness 不是细枝末节。** 同一底座模型在有无明确世界建模、回放验证和计划约束时
   能出现数量级差异；强模型仍决定状态落地和发现机制的能力上限。
2. **预测 transition 正确不等于会玩。** 代理还要发现目标、保留必要的隐状态、选择值得
   建模的时刻、选择有区分力而非纯信息量大的动作，并在计划失配时适当停止。
3. **历史回放是必要但不充分的门槛。** 它过滤与过去矛盾的程序，却不能保证不存在未观测
   条件、错误本体或错误目标。真正的进步来自“反例驱动地改变 state 表示和机制”。
4. **动作效率与内部算力脱钩。** RHAE 对真实游戏动作的惩罚很严格，但可以允许大量模型
   推理、代码生成、回放和模拟搜索。应同时报告 token、调用数、成本、时延和并行/重跑
   策略。
5. **公开集几乎不再能作为能力终点。** 对公开游戏进行长时间调试会让 prompt、工具与
   orchestration 吸收游戏家族先验；真正需要的是预先冻结的 harness 在 Semi-Private /
   Private 集上的独立复现。

## 6. 尚未解决的关键问题

| 问题 | 目前状态 | 需要什么证据 |
| --- | --- | --- |
| 公开 99–100% 是否能迁移到未知游戏？ | 未知 | 冻结实现的官方 Semi-Private/Private 评测 |
| 世界程序是否真是关键因果变量？ | 部分支持 | 同模型、同 token/调用预算、固定 fallback 的组件消融 |
| state grounding 能否在新视觉本体下稳定工作？ | 脆弱 | 新对象、隐藏状态、动画/HUD 扰动下的 OOD 测试 |
| 主动实验是否比系统搜索省动作？ | 机制上合理，证据有限 | 与 graph search / uniform probing 的配对 action 曲线 |
| 本地开放模型能否承载该能力？ | 仍明显落后 | 固定单 GPU、无网络、同一私有集上的多模型比较 |
| 成本能否降至实用水平？ | 不清楚 | 逐游戏 token、调用、GPU、时延和重跑账本 |

## 7. 对“当前现状”的简明判断

ARC-AGI-3 现在呈现出一个分裂但健康的研究格局：

* **在可见的 25 个游戏上，** 前沿模型配合程序化 world-model harness 已具备接近人类、
  甚至低于人类动作数的表现；这证明代理外部结构可以显著放大基础模型能力。
* **在真正未知的环境上，** 已验证结果仍只有 7.78%，没有证据表明 Public 的近满分会
  保持到 Semi-Private 或 Kaggle Private。
* **在可提交的离线设定下，** 开放模型与单 GPU 约束仍是瓶颈；Duck 的里程碑胜利是
  重要的工程基线，而非问题已解。

所以，最准确的表述不是“ARC-AGI-3 被攻克了”，而是：**公开集已显示出可执行世界模型
和验证式 harness 的巨大杠杆；这个杠杆在未见游戏、严格离线和统一计算预算下是否仍然
有效，仍是当前最关键的未决问题。**

## 参考资料

1. [ARC-AGI-3 Technical Report](https://arxiv.org/abs/2603.24621)
2. [ARC-AGI-3 scoring methodology](https://docs.arcprize.org/methodology)
3. [ARC Prize verified GPT-5.6 results](https://arcprize.org/results/openai-gpt-5-6)
4. [ARC Prize community leaderboard](https://arcprize.org/leaderboard/community)
5. [Schema harness](https://schema-harness.github.io/)
6. [Tycho](https://arxiv.org/abs/2607.28287)
7. [Executable World Models / Rodionov](https://arxiv.org/abs/2605.05138)
8. [Duck harness](https://github.com/Tufalabs/duck-harness)
9. [OPINE-World](https://github.com/david-courtis/opine-world)
10. [ARC-AGI-3 Preview learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
