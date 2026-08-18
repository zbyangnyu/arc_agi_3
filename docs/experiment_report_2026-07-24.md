# Meta-learning × JEPA 第一轮实验报告

日期：2026-07-24

## 决策摘要

本轮把研究计划中的三个近期资格测试都实际跑了起来，结论是：

1. **Track A 内部 close-out：部分成功，但总 gate 仍为 NO-GO。**
   Frozen-teacher categorical distillation 找到了一个不遗忘旧能力的 Pareto 点：
   100-step、teacher weight 10,000 的模型在 192 个 held-out tasks 的四个
   active-prefix stages、single、pair、triple 上全部为 100%。相对原模型，
   cross-axis RR reversal 从 2.083% 降至 0.521%，learned/learned B2 从
   82.64% 升至 89.06%。但预注册门槛分别是 0.1% 和 90%，P99 odds drop
   也仍有 18.35 nats，远高于 0.5 nat。
2. **Symbolic Alchemy：benchmark qualification 为 GO。**
   官方纯 Python 环境、确定性、action/observation contract 均通过；在官方固定
   1,000 episodes 上，ideal/search oracle 相对 random 和 released baseline
   都有很大的 paired return headroom。
3. **Push-T / LeWM：Tier 1 checkpoint qualification 通过，B0 尚未完成。**
   官方 LeWM checkpoint 已在 RTX 5090 上 strict-load、转为 stable-worldmodel
   object checkpoint 并完成 CUDA encoder forward。未下载 13.1 GB dataset，
   未运行 MPC，因此这不是 Push-T success reproduction。
4. **“主要因为模型和数据太小”仍不是当前最有证据的解释。**
   同一小模型通过更合适的 objective 就能大幅改善 retention、partition 和 B2；
   剩余错误又高度集中在特定 cross-factor probe。这更支持 likelihood
   factorization/calibration 问题。像素世界模型以后确实需要更大容量，但不能据此
   解释当前 RuleGrid 的结构性 failure。

## 本轮实现

### Frozen-teacher proper distillation

`scripts/run_counterfactual_locality_finetune.py` 新增：

- frozen initial executor；
- 对五个原 replay domains 的完整 categorical next-cell
  \(KL(\text{teacher}\Vert\text{student})\)；
- teacher 不参与新 geometry/locality 目标；
- teacher weight、逐域 KL、provenance 和新 checkpoint schema 的完整记录。

目标变为：

\[
L =
L_{\text{active replay}}
+ 0.1L_{\text{geometry}}
+ \lambda_{\text{loc}}L_{\text{fiber Huber}}
+ \lambda_{\text{teacher}}KL(p_T\Vert p_S).
\]

### 审计 checkpoint 边界

下游 acquisition/audit loader 现在只接受：

- 原 active-support checkpoint；或
- 明确继承该 schema 的 locality-v3 continuation。

新 continuation 还必须同时满足 model type、parent lineage、checkpoint/result
schema、SHA-256 binding 和 active-prefix gate。没有把 schema 检查改成无条件放行。

### 外部 benchmark runners

- `scripts/run_symbolic_alchemy_smoke.py`
- `scripts/run_symbolic_alchemy_headroom_qualification.py`

两者都固定官方源码 commit、依赖/runtime、资源哈希和 deterministic protocol，并拒绝
覆盖已有结果。

## Track A：内部实验

### 训练 Pareto

所有新 distillation runs 使用同一初始 checkpoint、同一 replay/geometry streams 和
同一 held-out singleton set。

| 条件 | Steps | Teacher | 旧域 formal gate | Singleton Huber ↓ | NLL/cell ↓ | Exact grid ↑ |
|---|---:|---:|---:|---:|---:|---:|
| GeomSup | 100 | 0 | Fail | 5.8366 | 0.3233 | 7.75% |
| LocReg | 100 | 0 | Fail | **2.8697** | **0.2366** | 17.84% |
| Distill | 10 | 1,000 | **Pass** | 6.2832 | 0.5548 | 21.22% |
| Distill | 100 | 1,000 | Fail | 3.0789 | 0.2845 | **25.13%** |
| Distill | 100 | 10,000 | **Pass** | 3.5981 | 0.3361 | 24.93% |

100-step/10,000 模型的 t0/t1/t2/t3 set equality 和 true-rule exact 均为
192/192；single、pair、triple 的 cell/grid/task accuracy 也全部为 1.0。

这说明 catastrophic forgetting 可以通过 proper teacher anchor 消除，但更强 anchor
与新 geometry/locality 之间仍有可测的 trade-off。100-step/1,000 虽在 singleton
上略好，却在少数 active-prefix 边界上失败，不应作为正式 checkpoint。

### Forced cross-axis likelihood

三次 audit 的 protocol、canonical panels、split、seeds 和统计单位完全 matched；
每个 branch 包含 1,536 个 deduplicated cross sequences。

| Checkpoint | RR reversal ↓ | RR P99 drop ↓ | Raw full-grid NLL ↓ | Partition F1 ↑ | PP reversal |
|---|---:|---:|---:|---:|---:|
| Base | 2.083% | 35.7118 | 4.1156 | 0.8479 | 0 |
| Distill 10 / 1,000 | **0.260%** | 19.6416 | 2.5213 | **0.9168** | 0 |
| Distill 100 / 10,000 | 0.521% | **18.3521** | **2.2063** | 0.9098 | 0 |

最坏错误仍集中在 `collision:v1` 和 `trigger:v1`。100-step/10,000 的
`collision:v1` 仍把正确的四个 outcome classes 分裂为七个；其最坏 RR case
会把 relation query 的真值概率从接近 1 降到 \(7.1\times10^{-7}\)。同一 case
用 oracle factor projection 后没有 reversal。

因此：

- locality/distillation 的方向有效；
- 更多相同步数不是单调修复；
- oracle projection 的完全 rescue 把故障定位到 global likelihood head 的
  cross-factor contamination。

### Learned two-axis B2

144 scenarios、每个 16 truths、三个 seeds、四个 selection/update conditions：

| Condition | Base B2 | Distill 10 / 1,000 | Distill 100 / 10,000 |
|---|---:|---:|---:|
| Exact select / exact update | 1.0000 | 1.0000 | 1.0000 |
| Exact select / learned update | 0.9392 | **0.9566** | 0.9531 |
| Learned select / exact update | 0.8368 | 0.9132 | **0.9757** |
| Learned select / learned update | 0.8264 | 0.8637 | **0.8906** |

100-step/10,000 的 B3 learned/learned 为 0.9340。它在 B2 learned selection +
exact update 已达到 0.9757，说明 selector 的主要结构已经可用；learned posterior
反馈到下一步 selection 时仍会放大少量 likelihood 错误。

10-step 模型拥有更高的 aggregate partition F1，却有更低的 B2/B3。这证明静态
MAP partition F1 不能替代 sequential posterior evaluation。

### Track A gate 判定

| Gate | 目标 | 当前最好 | 判定 |
|---|---:|---:|---:|
| 旧 static/active/geometry retention | 正式模型 100% | 100% | Pass |
| RR reversal | ≤ 0.1% | 0.260% | **Fail** |
| RR P99 odds drop | ≤ 0.5 nat | 18.352 | **Fail** |
| Learned/learned B2 | ≥ 90% | 89.06% | **Fail** |
| DoorGame learned selector paired RMST | CI lower bound > 0 | 尚未实现 | Pending |

表中的 reversal 与 P99 是两个 formal-gate-PASS checkpoints 各自最好的单项值；
没有一个 checkpoint 同时达到这些目标。闭环更好的 100-step/10,000 checkpoint
对应的是 0.521% reversal 和 18.352 P99。

总判定：**Track A 尚未 close-out。** 已经获得足够证据停止继续 sweep global-head
宽度、teacher weight 或相同数据步数；下一项应是 factor-local proper likelihood
architecture。

## Symbolic Alchemy qualification

### 环境 smoke

官方 `dm_alchemy` commit：
`68a26254b5c0f15e84fa0c15d66bf0c626ede8e0`。

- observation：`float32[39]`
- action：integer `0..39`
- 10 trials × 20 no-op = 200 steps，reward 0，正确到达 LAST
- 同 seed 双环境的 201 个 hashed states 逐步完全一致
- action 2 的真实 potion transition 改变四个 observation features，双环境一致

### 官方 1,000-episode headroom

使用同一批官方固定 chemistries/items。ideal/search 不重新做昂贵 planning，而是
用官方 bundled actions 经官方 replay API 重放；random-action 使用固定、按 episode
派生的 seed。

| Policy | Mean episode return |
|---|---:|
| No-op | 0.000 |
| Random action | 145.582 |
| Released baseline | 155.182 |
| Ideal observer | 284.417 |
| Search oracle | 288.529 |

| Paired gap | Mean | Episode bootstrap 95% CI |
|---|---:|---:|
| Ideal − random | 138.835 | [136.617, 141.102] |
| Search − random | 142.947 | [140.753, 145.143] |
| Ideal − baseline | 129.235 | [127.084, 131.504] |
| Search − baseline | 133.347 | [131.131, 135.595] |

四个核心差值在 1,000/1,000 paired episodes 上都为正。Ideal/search 每 episode
平均只使用约 69/61 个 effective events，而 random/baseline 约为 134/130。
因此该 headroom 不是靠更多动作换来的。

判定：

- **作为普通 learned method 到 meta-learning/oracle 水平的主外部 gate：GO。**
- search 相对 ideal 只高 4.11 return，且 56% episodes 持平；对两个 oracle-class
  方法的顶端分辨率有限。
- 本轮没有运行项目模型；这是 benchmark qualification，不是方法结果。

## Push-T / LeWM Tier 1

在既有 RTX 5090 主机的隔离环境中：

| 项目 | 结果 |
|---|---|
| LeWM repo commit | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` |
| HF model revision | `22b330c28c27ead4bfd1888615af1340e3fe9052` |
| Source weights SHA-256 | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| Parameters | 18,034,478 |
| Strict state-dict load | Pass，303 entries |
| CUDA load | 0.073 s |
| Encoder forward | Pass，0.218 s，output `[1,257,192]` |
| Runtime | PyTorch 2.8.0 + CUDA 12.8，RTX 5090 |
| Dataset / MPC | **未下载 / 未运行** |

远端通过 HF mirror 获取模型后，又以官方 Hugging Face 页面显示的 revision、byte
count 和 SHA-256 交叉核验，三者完全一致。LeWM tracked source 未改动；仓库的 dirty
标记仅来自 Python import 生成的 untracked `__pycache__/`。

安装暴露了真实的 reproducibility 风险：`env` extra 牵入需要 system SWIG 的
Box2D、当前 unconstrained Transformers major version 与 checkpoint encoder key
布局不兼容、stable-worldmodel 的 checkpoint path contract 也必须显式遵守。最终
artifact 固定了实际工作的 package versions 和每次失败记录。

判定：

- Tier 1 代码/权重/CUDA compatibility：**PASS**。
- Track B0 platform gate：**仍未判定**；至少还需 dataset alignment smoke 和官方
  MPC evaluation。

## 对原假设的更新

### 是否主要因为模型和数据规模太小？

**目前不支持把它列为主因。**

- 不增参数，只改 teacher/locality objective，就让旧域从严重遗忘恢复到全 100%，
  并把 learned/learned B2 提高 6.42 个百分点。
- 剩余 reversal 集中在两个已知 probe variants，并可被 factor projection 完全消除。
  这更像错误的条件独立结构，而不是均匀的容量不足。
- 18M 参数 LeWM 说明 raw pixels 确实需要更大 backbone；它不能反推 54k RuleGrid
  model 的当前 likelihood failure 也是 scale failure。

规模实验仍应做，但必须在 factor-local architecture 后做 matched
small/base/large scaling，而不是先扩大旧 global head。

### 是否应该追求更多 benchmark？

应追求**更正交的证据，不是更多分数**：

- RuleGrid：机制与 likelihood 单元测试；
- Symbolic Alchemy：主因果 meta-learning 外部 gate；
- Push-T/LeWM：pixels、world model、planning gate。

这三类已经足够覆盖近期研究命题。暂不增加 MetaWorld/PointMaze，除非需要定位特定
control 或 horizon failure。

### Meta-learning + JEPA / LeJEPA 是否合理？

合理，但需要保持模块边界：

- LeWM/SIGReg：共享 pixel/state representation 与 action-conditioned rollout；
- proper factor-local predictive density：posterior evidence；
- Persistent-K belief：episode-level hidden mechanism；
- query-relevant VOI：active probes。

未经校准的 latent distance 不能直接替代 likelihood。本轮 cross-axis 结果正好说明：
query MAP 很准时，错误的 full-grid evidence 仍可把真规则淘汰。

## 下一轮冻结计划

### P0：Factor-local proper likelihood head

只比较以下三项，停止 teacher-weight sweep：

1. 当前 shared/global outcome head；
2. shared encoder + acted-axis gated delta heads；
3. 相同总参数量的 wider global-head control。

训练继续保留：

- 10,000-weight frozen teacher anchor；
- 原五域 replay；
- counterfactual nuisance-fiber pairs；
- proper categorical NLL，不用 MAP-only loss。

固定复验顺序：

1. old retention gate；
2. exhaustive cross-axis reversal/P99；
3. matched three-seed B2；
4. DoorGame learned-selector paired RMST。

只有 reversal ≤ 0.1%、P99 ≤ 0.5 nat、B2 ≥ 90% 后才进行三训练流正式复现。

### P1：Symbolic Alchemy passive adaptation

先做 symbolic observation，不做 pixels：

- no-context；
- GRU/Transformer context；
- calibrated K1；
- ordinary K4 ensemble；
- Persistent-K4；
- bundled ideal/search ceilings。

主指标为 trial-prefix adaptation AUC；必须按 episode/chemistry 切分，不能把官方
evaluation chemistries 用作训练集。Passive 通过后才加入 active action selection。

### P1：Push-T B0 completion

1. 固定当前成功环境与 checkpoint；
2. 下载 dataset 前先保存 manifest、大小和 SHA；
3. 跑 1-episode reduced-budget alignment smoke；
4. 再跑官方 protocol checkpoint evaluation；
5. 只有 reproduction 落入合理置信区间后，才训练 matched no-SIGReg / LeWM。

## 主要 artifacts

- 研究计划：`docs/meta_jepa_external_benchmark_research_plan.md`
- 本报告：`docs/experiment_report_2026-07-24.md`
- Distill 正式候选：
  `runs/counterfactual_locality_distill100_w10000_seed2026072402/`
- Candidate forced audit：
  `runs/forced_cross_axis_likelihood_audit_distill100_w10000_g64_b16_seed2026072401/`
- Candidate learned bridge：
  `runs/two_axis_compositional_learned_bridge_distill100_w10000_g16_s3_20260911/`
- Alchemy smoke：
  `runs/symbolic_alchemy_smoke_seed123_20260724/`
- Alchemy headroom：
  `runs/symbolic_alchemy_headroom_qualification_20260724/`
- LeWM Tier 1：
  `runs/lewm_pusht_tier1_5090_20260724/`

## 限制

- 新训练 Pareto 目前只有一个 replay/geometry training stream；不应把点估计写成稳定
  scaling law。
- RuleGrid executor 仍使用 privileged factor code 和 oracle-canonical palette roles。
- 最终 DoorGame 仍没有 learned selector 的 paired RMST 结果。
- Alchemy 只完成环境与 benchmark headroom qualification，项目模型尚未接入。
- Push-T 只完成 checkpoint/load/forward，未完成 dataset/MPC qualification。
