# GRAM 顺序证据同化：快速猜想—观察—证伪是否成立

## 结论

这次 follow-up 支持一半、否定一半：

- **外置验证器 + 持久离散候选 + 每阶段重新读取完整历史**，可以构成可靠的快速“猜想—观察—筛选”外循环；
- 但当前 GRAM prior 不是好的规则 proposal。它在决定性强证据出现后仍很少提出此前缺失的规则，并且在相同 proposal 数下显著弱于 uniform control。

在 fold 0 的同一 48 个 held-out contexts、三个 inference seeds、每阶段 `W=32` fresh proposals 上，learned-verifier population search 的 t3 target coverage 为：

| proposal | factual | paired counterfactual | 两分支平均 |
|---|---:|---:|---:|
| raw GRAM，仅重新推理 | 33.33% | 31.25% | 32.29% |
| GRAM + verifier + carry | 38.19% | 45.83% | 42.01% |
| uniform + verifier + carry | **79.17%** | **84.72%** | **81.94%** |
| fixed K4，每阶段重新推理 | 64.58% | 72.92% | 68.75% |
| exact 64-state filter | 100% | 100% | 100% |

更直接的反证是 conditional recovery：只看 t2 仍未包含 target 的情况，GRAM 在 t3 强证据后仅恢复 factual `2/91 = 2.20%`、counterfactual `0/78 = 0%`；uniform control 分别为 `17/47 = 36.17%` 和 `10/32 = 31.25%`。

所以目前最合理的判断不是“GRAM + SMC 已经奏效”，而是：**belief update 骨架成立，GRAM proposal support 失败**。

## 这次到底测试了什么

每个 task 固定执行同一条完整证据序列，模型不选择 action：

1. `t0_support`：六条 public support，精确 version space 大小为 4；
2. `t1_partial`：追加一个 `1+3` partition probe；
3. `t2_neutral`：追加大视觉变化但不改变 version space 的 distractor；
4. `t3_strong`：追加能唯一确定规则的 `1+1+1+1` probe。

每个 factual task 都有一个 paired counterfactual：两条分支使用完全相同的 public states 和 actions，只把 observed target 分别由两个初始 support-compatible programs 生成。partial target 被强制不同，neutral target 被强制相同，strong evidence 最后分别唯一确定各自 program。

这个设计测试：

- 新观察是否改变候选规则；
- 无信息观察是否保持已有候选；
- 决定性观察能否救回此前遗漏的规则；
- belief 是否真正响应 target，而不是 task/probe identity。

它**没有测试主动 action selection**。candidate kind 只供 evaluator 预先构造固定序列，task ID、probe ID、candidate kind 和 true program 都不会进入模型 tensor。

## 为什么没有把实现称为 SMC

最初的 SMC 草案经数值审计后被否决：support-conditioned proposal 使用了同一份 evidence，却没有可计算的 proposal correction；变更后的 code 还继承旧 code 的 path weight，rejuvenation 也不是合法的 MH kernel。因此它不能声称近似 Bayesian posterior。

最终 [`prp_wm/gram_smc.py`](../prp_wm/gram_smc.py) 保留历史文件名，但实现明确叫 `GRAMVerifierPopulationSearch`：

1. 每个 stage 产生 `W` 个 fresh GRAM 或 iid-uniform codes；
2. 只携带上一阶段保留的离散 codes，不携带 recurrent state 或 path weights；
3. 合并、去重后，用独立 verifier 对**当前完整历史**一次性评分；
4. 优先保留 MAP-exact codes；若一个都没有，只保留 minimum-error stratum；
5. normalized energy 只用于当前 retained population 的启发式排序，不解释为 posterior mass。

数值回归验证了：同一最终 code 与同一完整历史，无论伪造的旧 path weights 顺序如何，结果完全相同。

## 独立 active-prefix verifier

原 support-calibrated executor 在六条 support 上能 100% 复原 version space，但 appended active transitions 完全 OOD：t1、t2、t3 的 MAP-exact bank 都是空集。因此不能直接用它评价顺序证据候选。

[`scripts/run_active_support_calibrated_executor.py`](../scripts/run_active_support_calibrated_executor.py) 从旧 checkpoint warm-start，训练一个**独立** verifier。训练仍是 privileged ceiling：factor code 与 palette-role canonicalization 都是 oracle 输入；对 diagnostic 和 t0–t3 public panels，使用 simulator 为全部 64 codes 产生 targets。

第一轮 500 steps 已接近但没有通过 gate：

| stage | neural set = symbolic set | true code MAP-exact |
|---|---:|---:|
| t0 | 100.00% | 100.00% |
| t1 | 93.75% | 97.92% |
| t2 | 93.75% | 97.92% |
| t3 | 97.92% | 97.92% |

错误全部是 false negative，且集中在 held-out trigger cases。唯一一轮 300-step low-LR continuation 后，192/192 held-out tasks 的 t0/t1/t2/t3 set equality、true-code exact、single/pair/triple diagnostic exact 全部达到 100%，所有 false positive/negative 都为 0。

最终 verifier artifact：[`runs/active_support_calibrated_executor_cont300_seed20260731/result.json`](../runs/active_support_calibrated_executor_cont300_seed20260731/result.json)，checkpoint SHA256 为 `b0227834cbcc2e3fd30c513d6dc2234446a5ae851a00608ca001ac2344a72483`。GRAM proposer 始终保留其原 checkpoint 内的旧 executor；新 verifier 只在外部评分，两个对象不共享。

## 三 seed learned-verifier 结果

下面均为相同 48 tasks 上三个 inference seeds 的平均；factual/counterfactual 是 paired branches，不当作独立数据集。

### target 是否在 belief 中

| method | branch | t0 | t1 partial | t2 neutral | t3 strong |
|---|---|---:|---:|---:|---:|
| raw GRAM | factual | 30.56% | 30.56% | 29.86% | 33.33% |
| raw GRAM | counterfactual | 34.03% | 34.72% | 30.56% | 31.25% |
| GRAM + verifier + carry | factual | 31.94% | 36.81% | 36.81% | 38.19% |
| GRAM + verifier + carry | counterfactual | 34.72% | 44.44% | 45.83% | 45.83% |
| uniform + verifier + carry | factual | 29.17% | 50.00% | 67.36% | **79.17%** |
| uniform + verifier + carry | counterfactual | 42.36% | 62.50% | 77.78% | **84.72%** |

GRAM 的外置筛选和 carry 相比 raw prior 有小幅收益，但远小于 uniform proposal。neutral stage 中，两种 population search 的 target coverage drop 都是 0，说明 carry 确实提供了持久性；其 JSD 仍包含 fresh-proposal churn，不能被解释成纯 evidence response。

### symbolic-verifier proposal ceiling

把 learned energy ranking 替换成 privileged exact consistency 后，结论不变：t3 GRAM coverage 为 factual `37.50%`、counterfactual `43.06%`；uniform 为 `89.58%`、`86.81%`。对 t2-missing cases，GRAM recovery 为 `5/95`、`0/82`，uniform 为 `13/28`、`14/33`。

这排除了“只是 learned verifier 排错了”这一解释。

## GRAM proposal 具体坏在哪里

逐 task 重放三个 symbolic seeds 后，发现不是普通 Monte Carlo 波动，而是结构性 mode collapse：

- 所有已审计的 stage、recursion、task 与 trajectory draws 中，`collision=STOP` 从未成为 argmax；
- `relation=NONE` 只占约 1.4–2.1% draws；
- 全部抽样只出现 32/64 codes；
- W32 每 task 只有约 10.3–12.3 个 unique codes，uniform 是约 25.2–25.5；
- unique codes 从递归第一层约 12–13 个收缩到末层约 10–11 个，递归在变窄而非发现新机制；
- t3 的 172 个 misses 中，119 个来自含 STOP 或 NONE 的 target，占 69.2%；排除这两种值后仍只有 67.9% coverage，说明还存在 joint-code holes。

把 factor argmax 改成 categorical sampling 不是主修复：target 为 STOP 时，最终 head 的平均 softmax probability 只有约 `2.4e-5` 到 `3.4e-5`；NONE 的约 1–2% 概率可能得到小幅改善，但救不了零支持的 STOP 和大量 joint holes。

## 对 JEPA / GRAM / 因果理解的含义

这次实验更支持一个分层设计，而不是单一大 policy：

- JEPA-like encoder 提取与 query/probe 后果有关的状态变化；
- latent mechanism hypotheses 表示多个可能规则；
- world-model verifier 用干预后的 transition 证伪 hypotheses；
- persistent population 保存仍可能的机制；
- active controller 最后才依据 hypotheses 对不同 probe outcome 的分歧选择实验。

这与因果理解的核心对应：规则不必先被翻译成人工 DSL，但 latent hypothesis 必须能产生可比较的 intervention consequences。当前结果说明“验证和保留”可以工作；失败的是 proposal 对机制空间的支持，而不是必须把所有规则先写成文字。

## Proposal 修复 follow-up：探索底座有效，soft coverage 未通过

后续实验把评测拆成三个量，避免把四阶段累计抽样误称为 evidence conditioning：

- `F_t`：stage `t` 恰好 32 个 fresh proposals 是否包含 target；
- `U_t`：从 t0 到 `t` 的所有 fresh proposals 的累计 union 是否包含 target；
- `B_t`：symbolic verifier + carry 后 retained belief 是否包含 target。

iid uniform 的解析参考为 `P(F_t)=39.59%`，`P(U_0..U_3)=39.59/63.50/77.95/86.68%`。所以 t3 retained 接近 87% 本身不证明模型响应了新证据；关键指标是 `P(F_3 | target not in U_2)`。

### matched-W32 mixture 与 structured exploration

[`scripts/run_gram_proposal_mix_ablation.py`](../scripts/run_gram_proposal_mix_ablation.py) 固定每 task/stage 总共 32 个 fresh proposals、carry cap 32，比较 nested GRAM/uniform streams 与一个 32-code pairwise-balanced Latin covering bank。三个 inference seeds、同一 48 held-out tasks 的两分支 pooled mean 为：

| fresh proposal | F3 | U3 = B3 | F3 recovery given not U2 |
|---|---:|---:|---:|
| GRAM 32 | 33.68% | 42.36% | 2.30% |
| GRAM 24 + uniform 8 | 40.28% | 62.50% | 15.09% |
| GRAM 16 + uniform 16 | 45.14% | 74.31% | 25.80% |
| GRAM 8 + uniform 24 | 46.18% | 81.60% | 28.47% |
| uniform 32 | 38.89% | 85.76% | **41.06%** |
| Latin pairwise cover 32 | **48.26%** | **92.01%** | 36.11% |

所有方法的 `P(B_t|U_t)=1`，neutral retention 也是 1；verifier/carry 没有丢掉已经发现的规则。增加 uniform 的比例会单调改善累计发现与 fresh recovery。Latin cover 的无重复、pairwise balance 得到最高累计覆盖，但 uniform 对“此前完全没见过的 target”恢复更稳。由此可见 exploration floor 是可靠保底，却没有让 GRAM 学会新 evidence 后提出 blind modes。

完整 artifact：[`runs/gram_proposal_mix_ablation_w32_fold0_3seed/result.json`](../runs/gram_proposal_mix_ablation_w32_fold0_3seed/result.json)。

### public-only full-version-space coverage loss

[`GRAMFactorizedCausalK4.coverage_losses`](../prp_wm/gram_causal_rules.py) 新增了一个严格 public-support-only objective：

1. frozen executor 对 t0 public support 评估全部 64 codes，以逐 cell MAP equality 得到恰好四个 compatible codes；
2. 四个 iid public-prior trajectories 与这四个 codes 做无序 `4!` soft permutation matching；
3. 加入由 version space 派生的 axis-marginal KL 和 invalid-mass penalty；
4. 不读取 true program、query、behavior target 或 posterior `q`。

fold0 train pool 是 36 contexts × 4 tasks，三轴四值计数完全平衡。300-step、单 training seed 的全 public-path continuation 确实把 STOP/NONE 从经验零提高到明显非零，也把跨 context 的全局 code union 从 32/64 扩到 50/64；但这是 **mass relocation，不是 per-context coverage**：

- held-out W32 version-space recall `31.25% → 18.23%`；
- mean unique codes/task `11.83 → 6.25`；
- 三-seed F3 `33.68% → 22.57%`；
- U3=B3 `42.36% → 38.54%`；
- fresh recovery 虽从 `2.30% → 5.81%`，仍远低于 uniform `41.06%`。

审计没有发现 factor-bank index、mask gather、assignment 方向或 axis KL 的实现错误。失败来自 surrogate mismatch 与 compositional overfit：soft W4 probability/marginal loss可以通过把概率搬到少数新 codes 来下降，却不保证 hard argmax joint codes 在每个 task 内不同；全路径更新又允许 encoder/core/readout 记住 36 个训练 contexts，破坏 12 个 held-out fixed-axis pairs。虽然 global empirical-zero joint codes 从 32 降到 14，实际 t3 target strata 的 zero-hit 反而从 34/57 增到 37/57。

为隔离 catastrophic forgetting，又从原 checkpoint 做了只更新 `prior_head` 7,440 个参数的同配置 continuation：

- single-seed held-out W32 recall `31.25% → 32.29%`；
- 三-seed F3 `33.68% → 34.72%`；
- U3=B3 `42.36% → 46.18%`；
- fresh recovery `2.30% → 10.92%`。

这是一个干净的弱正 control，说明直接覆盖信号能稍微改善 prior；但 STOP 仍然从未成为 argmax，unique codes 继续下降，因此不能称作修复。全参数与 prior-only artifacts 分别为 [`all-scope result`](../runs/gram_public_coverage_cont300_fold0_seed20260801/result.json)、[`all-scope paired audit`](../runs/gram_public_coverage_mix_ablation_w32_fold0_3seed/result.json)、[`prior-only result`](../runs/gram_public_coverage_prioronly_cont300_fold0_seed20260801/result.json) 和 [`prior-only paired audit`](../runs/gram_public_coverage_prioronly_mix_ablation_w32_fold0_3seed/result.json)。

### 当前停止结论与下一架构

不再给这个 soft W4 loss 增加 steps 或事后调权重。下一次若继续 learned proposal，必须改变 proposal family，而不仅是目标函数：

1. 用显式 persistent/stratified components 或 slot identity，使不同 hypotheses 在结构上承担不同 mode；
2. 直接优化 hard/unique joint-code coverage，并用旧 checkpoint anchoring 防止组合表示漂移；
3. 永久保留 uniform 或 Latin-cover proposal floor，不能让 learned prior 拥有排除整个机制区域的权力；
4. 先在 t0 达到 fresh version-space recall `>=90%`、all-four `>=75%`，再训练 partial→neutral→strong prefix curriculum；
5. 进入 action selection 前还需 t3 fresh hit `>=80%`、`P(F3|not U2)>=70%`、retention `>=99%`、neutral drop `=0`，并让相对 uniform 的 paired context-bootstrap 95% CI 下界大于 0。

当前 fold0 已用于多轮开发，因此这些只能是 engineering gates。真正的结论必须换未用于调试的 Latin fold，并覆盖多个 training seeds。TTT 和 EIG action policy 继续后置。

## 限制与复现边界

- 只用了一个训练 checkpoint / context fold，三个 seeds 只是 inference randomness；尚不能主张跨训练 seed 稳定性。
- coverage continuation 也只有一个 training seed；三 inference seeds 不能替代跨训练 seed/fold 复现。
- all-scope continuation 运行后，训练 runner 才加入 `--trainable-scope` 选项；当前默认值仍是 `all`，训练路径未变，但该 all-scope artifact 记录的是加入选项前的 runner SHA。prior-only continuation 与三个 paired proposal audits 均记录当前源码 SHA。
- axes、每轴四值、palette roles、simulator targets 与 verifier pretraining 都是 privileged；没有从 ARC 原始像素自主发现变量。
- 固定证据 schedule 不等于 active exploration policy。
- fixed K4 和原 GRAM 都没有在 appended active prefixes 上训练；它们的 t1–t3 结果是有意的 OOD stress test。
- active verifier artifact 在运行期间记录了当时的 sequential-runner source hash；随后 runner 的报告标签与 symbolic-control 实现被审计修正，因此该单项 hash 与当前文件不同。active calibration runner 自身、checkpoint/result SHA 和其余依赖一致；最终 learned/symbolic screen artifacts 都记录当前 runner SHA `811c650369bd5bdf87231d405622c925f1877ebf4bb3651dc98333e51116c792`。

核心 artifacts：

- learned verifier-guided runs：[`seed 20260729`](../runs/gram_vps_learned_fold0_seed20260729/result.json)、[`seed 20260730`](../runs/gram_vps_learned_fold0_seed20260730/result.json)、[`seed 20260731`](../runs/gram_vps_learned_fold0_seed20260731/result.json)；
- symbolic-verifier runs：[`seed 20260729`](../runs/gram_vps_symbolic_fold0_seed20260729/result.json)、[`seed 20260730`](../runs/gram_vps_symbolic_fold0_seed20260730/result.json)、[`seed 20260731`](../runs/gram_vps_symbolic_fold0_seed20260731/result.json)；
- original raw GRAM checkpoint：[`result.json`](../runs/gram_causal_screen600_fold0_seed20260728/result.json)。

复现单个 learned run：

```bash
/Users/yangzhenbang/anaconda3/bin/python3 scripts/run_gram_smc_active_screen.py \
  --output runs/gram_vps_learned_fold0_seed20260729 \
  --active-verifier-checkpoint \
    runs/active_support_calibrated_executor_cont300_seed20260731/checkpoint_last.pt \
  --verifier-mode learned --eval-tasks 48 --particles 32 \
  --seed 20260729 --device cpu
```
