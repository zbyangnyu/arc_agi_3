# PRP-WM v0.1：反证优先的研究规格

> 状态：设计草案  
> 目标：把 v0 中混合在一起的“多模态表示、跨时持久、主动选择”拆成可独立证伪的命题。

## 0. 结论先行

原 v0 的研究方向合理，但当前设计还不能可靠验证它声称的核心机制。主要原因不是模型容量不足，而是实验归因和粒子语义尚未闭合：

1. `K=4` 同时获得更多预测分支、集合监督和未来监督，弱于它的 `K=1` 不能证明“多假设”有效；
2. 粒子 latent 被 GRU 改写后，旧粒子的似然权重不再天然对应新粒子，因而不是自洽的 Bayesian filter；
3. 对每个 query 独立做 Hungarian matching，可能让同一粒子在不同 query 上代表不同规则；
4. 像素级 disagreement 不是严格的规则信息增益，且容易受背景、变化面积和相关像素重复计数影响；
5. value、risk、候选动作生成和 on-policy 训练会混淆对核心机制的判断；
6. 只按 layout/seed 切分会测到规则模板识别，而不是对未见规则组合的泛化。

因此，v0.1 不直接实现原来的完整 10M–20M 模型。先做一个具有精确版本空间和精确主动实验上界的符号 benchmark；只有上界与强基线之间存在足够 headroom，才训练神经粒子模型。

## 1. 要验证的三个命题

不要把三者合成一个总命题。分别预注册：

### H1：多模态表示

在历史仍有歧义时，`K` 个规则模式在保留的跨布局诊断集上，是否优于计算量匹配的单 belief、多峰输出基线？

这里要排除“只是多了四个 decoder 输出”的解释。

### H2：跨时持久

在都能输出 `K` 个相干模式的前提下，带 lineage 的递归规则模式，是否优于每一步从完整历史重新推断的无序 `K` 模式？

这里的“持久”必须表现为跨 prefix 的行为身份连续，而不只是 latent 向量相近。

### H3：主动证伪

冻结同一个已训练模型后，基于其预测 belief 选择实验动作，是否比 random、coverage/change-seeking 和普通 disagreement 更少使用真实交互，同时不降低最终预测正确率？

模型不能一边闭环采样一边继续更新慢参数，否则“动作选择有效”和“on-policy 再训练有效”无法区分。

## 2. 明确粒子的语义

### 2.1 v0.1 中的定义

粒子不是 SMC 中的无偏 posterior sample。它是一个有限的、带置信权重的 **predictive hypothesis mode**：

\[
\mathcal B_t=\{(r_t^k,w_t^k)\}_{k=1}^{K},\qquad \sum_k w_t^k=1.
\]

一个模式是否代表一致的规则，不由 latent 距离定义，而由它在诊断面板上的行为签名定义。

给定诊断面板：

\[
\mathcal P=\{(s_q,a_q)\}_{q=1}^{Q},
\]

规则 `r` 的行为签名为：

\[
B_{\mathcal P}(r)=\bigl[T_r(s_q,a_q)\bigr]_{q=1}^{Q}.
\]

若两个规则在该面板上的签名相同，则 v0.1 把它们视为预测等价，不要求恢复生成器内部的真实 rule ID。

### 2.2 先采用 mode filter，不宣称严格 Bayes

更新器先根据新证据提议新的模式。由于模式已经移动，随后必须用全部已见历史对新模式重新评分：

\[
r_{t+1}^{1:K}
=U_\psi(r_t^{1:K},e_t,\operatorname{residual}_t^{1:K}),
\]

\[
s_{t+1}^k
=\log b_k
+\beta\sum_{i=0}^{t}
\overline{\log p_\theta(y_i\mid x_i,a_i,r_{t+1}^k)},
\qquad
w_{t+1}=\operatorname{softmax}(s_{t+1}).
\]

`b_k` 是与当前历史无关的固定 slot prior，首版取 `1/K`；不能使用已经包含旧证据的 `w_t^k`，否则历史重放会重复计数。只保留一个 validation-calibrated evidence scale `β`，不再同时调与它不可辨识的 temperature。

support 最长只有 8–16，历史重放很便宜。这样置信权重至少对应更新后的模式；但因为 `r` 是 learned proposal，`w` 仍称 confidence/evidence weight，不称严格 posterior。updater 应收到每个模式的 residual map/token，而不只是三个标量误差，否则它不知道预测在空间上错在哪里。

v0.1 删除“先用旧似然更新权重、再任意移动 latent”的组合，也不使用 `epsilon_w` 掩盖粒子死亡。若探索时确实需要保留极小概率，只在动作评分的副本上平滑权重，不修改 belief 本身。

如果后续要宣称 SMC，需要显式定义 proposal density、target density、importance correction、resampling 和 rejuvenation；这不属于 v0.1。

### 2.3 数值规则

- transition loss 必须按有效/信息单元归一化，不能把 `64×64` 像素和当作可跨样本比较的能量；
- 所有权重更新和 mixture likelihood 在 log-space 计算；
- evidence scale `β` 只能在 validation set 校准，不能把校准后的 confidence 重新解释为 posterior；
- 变化位置、非变化位置和整帧 exact match 分开报告。

若以后恢复像素 delta decoder，belief score 必须来自 proper outcome score，不能使用由真实 change label 决定的 `5×` 像素权重。令 `m_ij=1[y_ij≠x_ij]`，并禁止变化颜色头再次输出旧颜色：

\[
\log L_k=
\sum_{ij}\left[
(1-m_{ij})\log(1-\hat c_{ij}^k)
+m_{ij}\left(\log\hat c_{ij}^k+\log\hat q_{ij}^k(y_{ij})\right)
\right].
\]

terminal/reward 若存在也必须进入同一个完整 outcome score。balanced change loss 可以单独帮助训练，但不能伪装成 Bayesian likelihood。

## 3. Stage 0：先建立精确可解的最小环境

### 3.1 Stage 0-A：GF(2)-RuleProbe 管线测试

先用一个可以手算的四规则环境检查协议实现：

\[
r,a\in\{0,1\}^2,
\qquad
s_{t+1}=s_t\oplus\langle r,a_t\rangle_{\mathrm{GF}(2)}.
\]

`a=00` 没有信息；重复已有 probe 只重复同一线性约束；与历史线性独立的 probe 才减少规则熵。oracle 恒定用 2 个有效 probe 确定规则，uniform random with replacement 的理论期望是 `10/3` 步，天然有 40% 的动作差距。

当前 scaffold 已实现精确 log-space Bayes、oracle EIG、uniform、coverage、change-seeking、inference/privileged schema 分离和配对、按 rule/state 分层的 bootstrap。解析真值为：

- oracle restricted mean：`2.000`；
- budget=12 的 uniform restricted mean：`3.3319`；
- exact relative reduction：`39.97%`。

固定种子的 1000 个平衡 rollout trials（8 种 rule/state 配置、每种随机策略 8 次重复）得到：

- uniform restricted mean：`3.3354`；
- Monte Carlo relative reduction：`40.04%`；
- uniform − oracle 的 mean difference 95% CI：`[1.3015, 1.3705]`；
- 23 个单测全部通过（其中 5 个检查冻结 config、reference artifact 与 source manifest）。

这只说明实验管线与 Gate 0 判定能工作，不能作为 H1/H2 的证据。

代码只有在 `trials≥500`、trial 数能被 8 个 rule/state strata 整除、随机策略每个 trial 至少 4 次重复、bootstrap resamples≥1000 且预算至少为 2 时，才允许 `gate_eligible=true`；小样本即使偶然超过 25% 也不能显示通过。

### 3.2 Stage 0-B：多机制符号网格

- `8×8` 离散网格，直接提供符号颜色，不训练视觉编码器；
- 4–8 个有限离散动作，所有 probe 都安全且成本相同；
- deterministic transition；
- 无 reward、goal、death、value、risk、点击候选生成和多步 MPC；
- 每个 episode 的未知规则来自有限集合，可枚举精确 version space；
- `K=4` 的主实验只保留任一评测 prefix 下至多 4 个行为等价类；超过 4 的情况单独作为容量压力测试。

这一步验证的是“规则辨识”，不是 ARC-AGI-3 通关。

### 3.3 第一批规则族

至少使用三个彼此不同的机制族，避免只适配“撞墙”这一种模式：

1. 碰撞：停止 / 反弹 / 穿过 / 推动；
2. 触发器：切换 / 删除 / 生成 / 改色；
3. 接触关系：交换 / 跟随 / 远离 / 不作用。

每个任务随机化布局、颜色置换、对象位置和无关干扰物。颜色与规则、动作可用性与规则之间必须独立随机，避免捷径。

### 3.4 歧义前缀的验收条件

生成器只接受满足以下条件的任务：

1. support prefix 对至少两个规则完全 observationally equivalent；
2. 至少存在一个合法动作把当前 version space 分成两个或更多 outcome group；
3. 也存在表面变化大但不能区分规则的干扰动作；
4. 保留诊断面板足以定义目标规则族的行为等价类；
5. 真实规则、辨识动作和目标后果都不能由布局统计单独预测。

### 3.5 精确 oracle

历史 `D_t` 下的版本空间为：

\[
\mathcal V(D_t)=\{r:T_r \text{ 与 }D_t\text{ 中全部转移一致}\}.
\]

对候选动作 `a`，oracle 根据每个规则的精确后果把版本空间分组，并计算：

\[
\operatorname{EIG}(a)
=H(R\mid D_t)
-\mathbb E_{y\sim p(y\mid D_t,a)}
H(R\mid D_t,a,y).
\]

在 deterministic、均匀 version space 中，它等价于 outcome group 分布的熵。这个 oracle 是 benchmark 是否有价值的先决检查，不是可比较的学习方法。

### Gate 0-B：benchmark headroom

在至少 500 个配对任务上，把预算内未辨识任务作为右删失样本，oracle EIG 相比 uniform random 的受限平均辨识步数至少下降 25%，且配对、分层 bootstrap 的 mean difference 95% CI 不跨 0。若失败，先重做任务生成器，不训练神经模型。

## 4. Stage 1：最小神经模型

### 4.1 保留的模块

```text
symbolic grid + action
        ↓
small transition/evidence encoder
        ↓
K-mode recurrent belief updater
        ↓
shared rule-conditioned one-step decoder
        ↓
next-grid distribution + mode weights
```

首版使用小 CNN 或逐格 embedding；不要上 U-Net、RGB quantizer、value/risk heads。模型只有在 `8×8` 任务上通过 Gate 1–3 后才扩展到 `64×64`。

### 4.2 行为面板级、质量感知的 hard assignment

设 prefix 下仍兼容的第 `m` 个规则在完整诊断面板上的目标签名为 `Y_m^{1:Q}`。匹配 cost 必须联合所有 query：

\[
C_{km}
=\frac1Q\sum_{q=1}^{Q}
-\log p_\theta
\left(Y_m^q\mid s_q,a_q,r_t^k\right).
\]

\[
\hat\mu_m(z)
=\sum_k w_k\mathbf 1[z_k=m],
\qquad z_k\in\{1,\ldots,M\},
\]

目标模式质量来自生成器 prior 在当前 version space 上的条件质量：

\[
\mu_m(D_t)
=
\frac{
\sum_{r\in E_m\cap\mathcal V(D_t)}p_{\mathrm{gen}}(r)
}{
\sum_{r\in\mathcal V(D_t)}p_{\mathrm{gen}}(r)
}.
\]

对 `K=4`，直接枚举最多 `M^K` 个 row-hard assignment：

\[
\mathcal L_{\mathrm{assign}}
=\min_z
\left[
\sum_k w_k C_{k,z_k}
+\lambda_m
D_{\mathrm{KL}}
\left(
\mu(D_t)\middle\Vert
\frac{\hat\mu(z)+\delta}{1+M\delta}
\right)
\right].
\]

row-hard 约束保证一个 mode 不会在 matching 中同时分给多个规则类；联合签名 cost 保证它必须用同一个 `r_k` 解释整组 query。还要报告每个 mode 内部的 signature entropy，防止一个宽分布 mode 自己包办多个互斥规则。

不能对每个 query 独立重新 matching。否则一个粒子可以在 query A 上像规则 1、在 query B 上像规则 2，仍得到很低的 loss。普通 Hungarian 可作为等质量 smoke test，但它忽略 `w`，会把 alternative 塞进近零权重粒子；熵正则 OT 又允许一行拆给多个 target，因此都不作为主目标。

主实验让 `M≤K`。若 `M<K`，多个粒子可以共同承载同一真实模式的质量，但不得捏造额外高权重后果。若 `M>K`，报告由 oracle K-medoids 行为覆盖决定的容量上限，不把容量失败解释为推断失败。

### 4.3 跨 prefix 的身份

对一个 task bundle，在完整 prefix 序列上做一次联合 assignment，或用最早可区分时得到的 assignment 固定 lineage。报告：

- identity-switch rate；
- 每个粒子的 panel signature stability；
- 仍兼容模式的存活率；
- 被证伪模式的权重衰减速度。

先把 persistence 当作架构属性和评测指标，不加入原 v0 的 persistence regularizer。原正则可能惩罚收到新证据后的合理修正；只有观察到明确 label-switch 后，才设计针对性的 temporal assignment loss。

### 4.4 最小损失

\[
\mathcal L
=\mathcal L_{\mathrm{support\text{-}mix}}
+\lambda_j\mathcal L_{\mathrm{joint\text{-}query}}
+\lambda_a\mathcal L_{\mathrm{assign}}.
\]

其中：

- `L_support-mix`：必须使用看到 `y_i` 之前、只基于 `D_{<i}` 的 belief 做 prequential 预测：

\[
\mathcal L_{\mathrm{support\text{-}mix}}
=-\sum_i\log\sum_k
w_i^k p_\theta(y_i\mid x_i,a_i,r_i^k).
\]

先记这项 loss，再把 `y_i` 交给 updater。不能用读过 `y_i` 的 `r_{i+1}` 重构同一个 target；

- `L_joint-query`：同一个模式必须联合解释整组 query：

\[
\mathcal L_{\mathrm{joint\text{-}query}}
=-\log\sum_k w_k
\exp\left(-\sum_q\ell_{qk}\right);
\]

- `L_assign`：完整行为签名的质量感知 hard assignment。

初版删除独立 `L_div`、`L_persist` 和 future-weight calibration。hard assignment 已对真实模式及其质量施压；额外 diversity margin 会在真实模式数小于 `K` 时捏造结果，而 future hindsight target 会鼓励从 layout/seed 猜真实规则。

训练特权信息只能用于 train loss。validation/test 的动作选择、belief update、停止判据不得访问候选规则、未来 query 或隐藏 rule ID。

## 5. Stage 2：冻结模型后验证主动选择

### 5.1 完整结果上的预测模式信息分数

Stage 0 的枚举 belief 可以计算严格 EIG。learned mode filter 的权重不是严格 posterior，因此对 deterministic mode prediction，先按预测的完整下一状态分组、再计算的熵下降应称 **mode-index information score**，不要过度解释为参数或规则的真实信息增益。

对 stochastic prediction，从 mixture 中 Monte Carlo 采样完整下一状态 `y`：

\[
w_k(y,a)
=\frac{w_kp_k(y\mid a)}{\sum_j w_jp_j(y\mid a)},
\]

\[
J_{\mathrm{mode}}(a)
=H(w)-\frac1L\sum_{l=1}^{L}H(w(y_l,a)).
\]

当 `y_l` 独立采自完整 outcome mixture、`p_k` 是完整状态的 joint density 时，这估计当前模型中的 `I_model(K;Y|D,a)`。它不是实际 updater 执行后的 belief entropy reduction，因为新证据会让 mode 移动、合并或换标。实施时在 validation 上冻结权重校准、样本数 `L` 和 tie-breaking，并对所有动作使用共同随机数降低排序方差。

不要把逐格 JS 再乘 change mask 后称为严格 mutual information。逐格近似可以作为便宜 heuristic 基线，但完整状态内的相关像素不能被当成独立证据重复累计。

若最终目标是诊断面板上的预测，而不是恢复所有无关规则细节，后续应比较 prediction-oriented acquisition：只奖励能改善保留诊断分布的实验。

### 5.2 核心阶段的动作效用

\[
a_t^*=\arg\max_a J_{\mathrm{mode}}(a).
\]

核心实验不含 value/risk/cost 的任意线性权重。等 H1–H3 成立后，再把风险作为约束而非早期混合项，例如：

\[
\max_a \operatorname{EIG}(a)
\quad\text{s.t.}\quad
P(\text{failure}\mid a)\le\delta.
\]

## 6. 因子实验与强基线

### 6.1 表示轴

至少比较：

1. `Single-belief + 4 coherent heads`：一个 history belief latent，同样输出 4 个相干结果、使用同样的联合签名 hard-assignment loss；
2. `Reinfer-K`：每一步从完整 `D_t` 重新推断 4 个无序 modes，不保留 lineage；
3. `Persistent-K`：4 个递归模式和权重共同更新；
4. `Categorical q(r|D)`：若使用精确 DSL/transition，则明确标成 structured oracle；若作为 learned 强基线，必须说明它得到的 rule-ID/版本空间监督；若它与粒子法持平，这是有价值的否定结果；
5. `Independent ensemble`：4 个独立或 bootstrap dynamics heads，用于区分规则歧义与参数不确定性；
6. `K=1 capacity matched`：宽度/参数量匹配，但只能给单一相干预测。

所有学习方法共享训练数据、encoder、decoder 表达力和监督信号；同时报告参数量、每步 decoder 调用数和 FLOPs。`K=1` 不能只用逐格 unimodal head 后与 `K=4` mixture 比较。

### 6.2 动作轴

对同一个冻结模型比较：

1. uniform random；
2. coverage/change-seeking；
3. pixel/cell disagreement heuristic；
4. learned full-state mode-index information score；
5. oracle version-space EIG（只作上界）。

由此形成至少 `3 个主表示 × 2 个主策略` 的正交主实验；其余方法作为附加基线。

Stage 0-A 的 fixed-tie change-seeking 只用于展示 pathology。Stage 0-B 的公平基线在变化分数并列时使用 coverage 二级规则或随机 tie-breaking，并做多次 rollout。test 前在 validation 上确定“最强非 oracle 基线”，不能看 test 结果后挑选。

## 7. 数据划分与统计协议

### 7.1 三种 split 必须分开报告

1. `Layout-ID`：同一规则族，新布局/新颜色；只证明 within-family amortization；
2. `Parameter-OOD`：保留规则模板，测试未见参数组合；
3. `Composition-OOD`：训练见过 primitives，但没见过该机制组合；这是最接近“新规则组合”的核心 split。

不能只把同一规则程序的不同 seed 分到 train/test 后称为规则发现。

### 7.2 统计单位

- 至少 5 个独立训练 seed；
- 每个 split 至少 500 个均衡的 hidden-rule tasks；
- 所有方法在同一 task、初态、动作集合上配对评估；
- random policy 每个 task 使用多个 RNG rollout；
- training seed 与 task 是 crossed factors：以 training seed 为外层，规则族为固定 strata、task/rollout 为内层，并始终保持方法间配对；也可使用 two-way/multiway bootstrap；
- 多个主要比较使用 Holm 校正；
- 达到阈值所需动作数存在预算删失时，用 survival curve/受限平均辨识时间，不把未成功样本简单丢弃。

## 8. 主指标

### 8.1 首要指标

1. `Diagnostic mixture log loss vs interactions`，并汇总为 AULC；
2. 达到行为预测等价所需的真实动作数；
3. 保留诊断面板上的整帧 exact transition accuracy；
4. confidence/predictive calibration：Brier/NLL/ECE；
5. joint query NLL，而不只逐 query marginal NLL。

### 8.2 诊断指标

- behavior-signature Coverage@K；
- oracle-best particle error（只诊断覆盖，不作主要胜负指标）；
- effective particle count；
- identity-switch rate；
- change/no-change 分层 NLL；
- active acquisition regret：选中动作的 oracle EIG 与最优 oracle EIG 的差；
- oracle headroom recovery：学习策略收回了 random 与 oracle 间多少差距。

### 8.3 指标的可执行定义

- `Coverage@K`：目标类 `m` 在完整诊断面板上的平均 joint-signature NLL 低于 validation 冻结的 `δ_cov` 时算被覆盖；主报告 `Σ_m μ_m·covered_m`，另报 task-level all-covered rate；
- `identity-switch rate`：先对整个 prefix 序列做全局最小 cost lineage assignment，只在相邻 prefix 都仍兼容的目标类上计分；被证伪类不进分母，重复粒子按全局 assignment 处理；
- `辨识动作数`：离线 evaluator 判定首次达到、且之后持续保持 behavioral-equivalence 阈值的动作位置；模型自己的 stop 决策另报，stop threshold 只能在 validation 冻结；
- calibration：先按完整预测签名把多个粒子的权重聚合到行为类，再计算 Brier/NLL/ECE，不能直接校准没有固定语义的 slot index；
- 整帧 exact accuracy：使用完整状态 joint MAP 或最高权重相干 mode，不能逐格 marginal argmax 拼出一个不存在的 Frankenstein frame；
- “最终 loss 不恶化”：在 validation 上预注册非劣界 `δ_NI`，要求方法差异的 95% CI 上界不超过它，而不是只看差异是否“不显著”。

训练 NLL、assignment cost 使用 nats；信息分数统一使用 bits。任何把两者放进同一 utility 的后续版本都必须显式归一化。

### 8.4 暂定通过门槛

在看 test 结果前冻结阈值。完成小规模 pilot 后可只用 validation 调整一次：

- Gate 1：Persistent-K 的 panel Coverage@4 ≥ 90%，且其 mixture AULC 显著优于计算匹配的最强单-belief多峰基线；
- Gate 2：Persistent-K 相比 Reinfer-K 的 identity switch 显著更低，并在 4 次交互后的 diagnostic loss 上有独立增益；
- Gate 3：冻结的 Persistent-K + mode-index information score 相比最强非 oracle 探索基线，辨识动作数至少下降 20%，95% CI 不跨 0，且最终 diagnostic loss 满足预注册非劣界；
- OOD：上述方向至少在 Composition-OOD 上保持，不能只在 Layout-ID 成立。

若强多峰基线、严格 split 或 calibration 后优势消失，应判定核心命题暂未获支持，而不是继续叠加模块。

## 9. 失败定位矩阵

| 观察 | 优先结论 | 下一步 |
|---|---|---|
| oracle EIG 不优于 random | benchmark 没有足够可辨识 headroom | 重做歧义前缀和 probe 动作 |
| oracle 好，神经 Coverage@K 差 | 多模式推断/训练失败 | 查 panel matching、容量和捷径 |
| Coverage 好，权重 NLL/ECE 差 | confidence gating 不校准 | 改 proper scoring 与 evidence scale |
| belief 准，learned information score 选错动作 | acquisition 近似失败 | 查完整状态采样和相关像素重复计数 |
| active 优于 random，不优于 change-seeking | 任务只奖励大变化，不要求证伪 | 增加无信息大变化干扰动作 |
| ID 有效、Composition-OOD 失效 | 学到模板识别而非组合规则推断 | 加强组合 split 和生成器 |
| Persistent 与 Reinfer-K 相同 | 持久 lineage 没提供额外信息 | 接受否定结果，勿加 persistence loss 粉饰 |

## 10. 与 ARC-AGI-3 的关系

原方案选择 `64×64`、16 色和按键/坐标动作，与 ARC-AGI-3 的接口是吻合的。但真实 benchmark 还包含 v0.1 明确不解决的部分：

- 一个 observation 可能是 frame sequence，而不是单帧 Markov state；
- 无规则说明，也无明确 goal，需要同时发现目标；
- 每个环境含多个机制，后续 level 要组合早期习得的概念；
- 真实分数按动作效率计算，探索与执行共享昂贵预算；
- 对象持久、遮挡、隐藏状态和到达 probe 状态所需的多步计划都很重要。

所以 v0.1 的正确定位是 **机制可行性实验**，不是 ARC-AGI-3 agent v0。通过 Gate 0–3 后，再按顺序加入：

1. `64×64` 编码器和 frame-sequence state belief；
2. 从早期 level 到后期 level 的规则记忆与组合；
3. goal/terminal 假设；
4. 风险约束和动作成本；
5. 到达诊断状态的多步 planning。

## 11. 预计 1–2 周的实施顺序

1. 已完成：GF(2)-RuleProbe、精确 version space/oracle EIG、数据契约和 Gate 0-A；
2. 第 1–3 天：多机制 `8×8` 规则 DSL、生成器性质测试和 Gate 0-B；
3. 第 4–5 天：K=1 与 single-belief 4-head 强基线；
4. 第 6–8 天：Reinfer-K、Persistent-K、panel-level matching；
5. 第 9 天：冻结模型的 random/change/mode-index information 闭环评测；
6. 第 10–12 天：5 seeds、OOD splits、crossed/two-way bootstrap、消融报告；
7. 只有 Gate 0–3 全部通过，才开始原 v0 的视觉和风险模块。

## 12. 研究贡献的准确表述

若实验成功，最稳妥的表述不是“首次用 world model disagreement 主动探索”。更准确的是：

> 学习一个在单个新任务内部持续更新的、有限多模态的规则预测信念；用跨状态行为签名监督模式相干性，并选择能最快区分这些规则模式的干预动作。

它与普通 ensemble disagreement 的关键差别，应通过实验体现为：粒子表示的是 **同一个新任务内的规则歧义**，而不是慢参数对陌生状态的全局 epistemic uncertainty。

## 参考对照

- [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- [Planning to Explore via Self-Supervised World Models](https://proceedings.mlr.press/v119/sekar20a.html)
- [Prediction-Oriented Bayesian Active Learning](https://proceedings.mlr.press/v206/bickfordsmith23a.html)
- [Meta reinforcement learning as task inference](https://arxiv.org/abs/1905.06424)
