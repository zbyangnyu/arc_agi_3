# PRP-WM 在 ARC-AGI-3 技术版图中的位置

**更新日期：2026-08-15**  
**比较对象：** [ARC-AGI-3 技术现状调研报告](arc_agi_3_landscape_report_2026-08-15.md)中的外部系统。  
**结论先行：** PRP-WM 目前是一个有严格机制证据的**因果 likelihood、持久 belief 与
query-relevant information acquisition 研究系统**，而不是 ARC-AGI-3 端到端 agent。
成熟度位于“合成隐藏规则环境中、raw-ish observation 下的局部 belief 闭环”（M2），
尚未到外部环境适配（M3/M4），更未到 ARC-AGI-3 泛化（M5）。

这意味着它不能与 Schema、Tycho、OPINE-World 或 Duck 的 RHAE 放在同一排行比较；
但它研究的“如何避免错误预测证据淘汰真规则、如何让试探服务于最终决策”正好击中这些
harness 目前大多未被严格单独验证的部分。

## 1. 端到端能力对齐

ARC-AGI-3 的高分 harness 大致执行如下闭环：

```text
像素帧 → 对象/状态落地 → 假设或世界程序
     → 历史回放验证 → 目标推断/规划 → 真实动作
     → 预测失配成为反例 → 重写状态或规则
```

PRP-WM 当前闭环是：

```text
RuleGrid 公开状态、动作、反馈
     → 已知候选机制上的 predictive likelihood
     → persistent factor belief / posterior
     → 固定 probe bank 上的 query-relevant selection
     → 反馈后更新 belief
```

二者重叠在“由交互证据修正对机制的判断”，但 PRP-WM 的输入状态、规则空间和 probe
语言目前仍被强约束；外部 harness 需要从陌生像素中自行产生这些对象。

| 能力 | 外部领先 harness 的常见做法 | PRP-WM 当前状态 | 定位 |
| --- | --- | --- | --- |
| 状态落地 | 从 64×64 frame 发现对象、关系、隐状态与目标 | 色彩等变/事件表征已有实验；但仍存在 oracle canonicalization 历史依赖 | 未完成 |
| 假设空间 | 自由生成、编辑 Python 世界程序 | 已知的三因子、各四值共 64 code；尚未自主发现 rule axes | 受控机制研究 |
| 预测模型 | `step(state, action)`、renderer、goal 的可执行程序 | calibrated outcome density / likelihood；不是自由可执行 simulator | 部分完成 |
| 历史证伪 | exact replay、预测失配后修程序 | proper likelihood、cross-axis/reversal 审计；未做程序级 exact replay | 强的统计证据层，缺工程闭环 |
| 多假设 | 常见为一个当前程序，必要时重写 | persistent K4 / joint-belief 方向；当前已有 codebook 限制 | 研究优势，但未完成 raw 版本 |
| 主动探索 | 区分理论的实验、短计划 | exact query-relevant VOI 已通过；learned selection/update 仍非稳健 | 有 headroom，未闭环 |
| 规划和执行 | BFS/A*/自定义 planner、逐步 prediction check | 固定 probe family；不存在通用 action generation/goal planner | 缺失 |
| 跨关卡记忆 | 规则程序、workspace、knowledge file | 同一 RuleGrid episode 内 belief；未证明 ARC 式跨 level/game 迁移 | 缺失 |
| ARC-AGI-3 评测 | public scorecard / Kaggle private | 尚无 ARC-AGI-3 agent、scorecard 或离线提交 | 尚未开始 |

## 2. 现有证据真正证明了什么

### 2.1 已有强证据

1. **错误的 nuisance evidence 会破坏 active inference，结构可以阻断它。**
   17,177 参数的 factor-local executor 在当前受控设置中将 forced cross-axis 的
   catastrophic reversal 从 0.5208% 降至 0，P99 true-query log-odds drop 从 18.352
   nat 降至数值噪声，并保留旧 hard gates。这个结论不是“模型很准”，而是：若一个
   动作不作用于某个隐藏因子，该因子的微小预测偏差不能在全 grid likelihood 中被错误
   放大。

2. **query-relevant value of information 可优于 global information gain。**
   在 exact two-axis control 中，query-success greedy、query-MI 与 depth-2 DP 均在
   两个 probe 内达到 100% query success；global-EIG 在同一预算只达 50%，因为它优先
   取得对当前决策无关的 nuisance 信息。

3. **失败已被定位而非用规模掩盖。**
   3 seed、41,564 参数的 matched P1 实验表明 factor-local 模型稳定消除
   cross-axis odds corruption，却在 B2 均值 96.875% 处未达 98% gate，且不优于
   matched wider-global。残差被定位为 `collision:v1` 的 outcome-partition fidelity，
   因此对“factor locality 是剩余 B2 的主要变量”给出了正确的 NO-GO。

### 2.2 已有但只能视为上界/诊断的证据

* oracle axis projection 和 oracle-canonical role/palette 可以证明错误位于何处，
  不能证明公开代理已经能找到该结构。
* exact selector / exact update 的 B2=100% 证明探测语言和效用定义具有 headroom；
  它不证明 learned perception、world model 或主动策略已成功。
* 固定的 64-code 候选机制与预制 probe bank 使版本空间推理可精确审计；它同时移除了
  ARC-AGI-3 最难的“表示/假设/动作语言从何而来”问题。

### 2.3 当前明确的失败与未完成项

* learned/learned B2 曾为 82.6%，早期误差以 selection model 的 outcome partition
  误差为主；factor-local 的特权版本改善该问题，但 P1 的三 seed 结果仍是 96.875%，
  说明尚未可靠跨过 acquisition gate。
* 没有自主的 object extraction、机制变量发现、goal discovery、程序 synthesis、
  exact replay verifier、planner 或 action generator。
* 尚未在 Symbolic Alchemy、Push-T、Meta-Push-T 或 ARC-AGI-3 上做外部验证。
* P1 已 NO-GO；路线图正确地要求先处理 P1.1 outcome-partition fidelity，而非把当前
  机制结论扩张为端到端能力结论。

依据： [factor-local 验证](factor_local_executor_validation_2026-07-24.md)、
[P1 matched 报告](p1_matched_results_2026-07-24.md)、
[主动探索审计](exploration_information_audit.md)、
[系统路线图](prp_wm_system_roadmap_2026-07-24.md)。

## 3. 与外部路线的关系

### 与 Schema / Rodionov / Tycho

这些系统把“假设”表现为可运行 Python 程序，对完整交互历史回放，随后在程序内规划。
它们的最小解释单位是一个当前世界程序。PRP-WM 目前没有这层程序合成/回放/规划工程，
所以不能把自己的 density 视为等价的 executable world model。

但 PRP-WM 提出了这些系统很需要、却尚未由其公开结果严格证明的两个问题：

* 多条部分正确的假设应如何保留、定量比较，而不是直接覆写为一个“当前程序”？
* 一个动作产生的证据应更新哪些机制假设；怎样防止无关预测误差把正确理论排除？

因此关系是**互补而非替代**：前者擅长生成与执行程序化假设；PRP-WM 的研究目标是让
假设比较与试探选择具有更可靠的因果统计结构。

### 与 OPINE-World

OPINE-World 是概念上最近的外部邻居：它通过 actor、CEGIS builder、exact replay 和
counterexample 来维护 world program，并以 ontology error 诊断当前对象类型是否足够。
PRP-WM 与它共享“机制不确定性”和“应以反例更新”的问题设定，但实现重点相反：

| OPINE-World | PRP-WM |
| --- | --- |
| 从像素和交互中发明 object type / Python engine | 在给定候选机制空间内刻画 likelihood / posterior |
| exact replay 决定候选程序是否接受
| calibrated predictive density 与 posterior audit 量化证据强度 |
| 主要维护可执行当前模型 | 目标是维护多峰、joint 的机制 belief |

它同时是最能显示 PRP-WM 缺少端到端接口的参照：没有自由对象本体与 `game_engine.py`，
就不能直接在陌生游戏上行动。

### 与图探索/RL 和 Duck

图探索、轻量 RL 与 Duck 解决的是「在有限动作/本地模型/单 GPU 下仍能探索什么」。
PRP-WM 目前没有与这类系统竞争的 action-space coverage 或执行效率数据。其探测是在
小的、人工构造的 probe bank 中选择，而不是在大而陌生的 action space 中发现可行的
动作。因此其短板不是“尚少一个更好的 selector”，而是 action representation、
reachability、程序执行与真实环境接口还不存在。

### 与 DreamTeam / NOOA

多代理 workspace 系统以外部文件、角色分工和 memory 缓解上下文局限。PRP-WM 已具备
更严格的机制证据审计思想，却尚无长期知识资产、任务级工作区或跨游戏记忆。两者都不是
对方的子集。

## 4. 公平的总体定位

```text
研究机制成熟度
M0  exact/oracle feasibility             已完成
M1  learned components in synthetic env  已完成
M2  raw-ish belief + local closed loop   当前所在
M3  external passive adaptation/planning 未验证
M4  external active inference advantage  未验证
M5  ARC-AGI-3 generalization             未验证

ARC-AGI-3 agent readiness
perception → hypothesis language → executable model → planner → evaluator
   部分            缺失              缺失         缺失       缺失
```

最准确的一句话是：

> PRP-WM 已经是一个对“在已定义机制空间内怎样可靠地相信、排除和试探规则”有强实验
> 约束的研究原型；它还不是一个能够在未见 ARC-AGI-3 游戏中自己定义状态、发明机制、
> 规划并执行的代理。

这不是负面判断。外部高分 harness 当前最薄弱的证据恰好是：在自由程序假设之间，
它们的 belief、证据归因和动作价值是否校准、是否可迁移。PRP-WM 的贡献潜力在于将该
层从 prompt heuristic 变为可测量、可证伪的组件；但在获得外部验证前，应将主张限制
为这一层，而不是 ARC-AGI-3 性能。

## 5. 对外陈述时的推荐表述

**可以说：**

* “我们研究 episodic hidden-rule environments 中 factor-local predictive evidence、
  persistent belief 与 task-relevant experiment selection。”
* “在受控 RuleGrid 实验中，我们定位并消除了 cross-factor likelihood contamination，
  同时将残余 acquisition 误差定位为 outcome partition fidelity。”
* “我们把它作为通往外部 interactive reasoning 的可证伪组件研究，而非将内部成绩当作
  ARC-AGI-3 成绩。”

**现在不应说：**

* “已经具备 ARC-AGI-3 world model / agent。”
* “已经从像素自主发现因果变量或规则。”
* “因子化 belief 已优于所有 alternative，或已证明能节省真实外部交互。”
* “可与 Schema、Tycho、Duck 的 RHAE 直接比较。”
