# PRP-WM

当前仓库用于验证 **Persistent Rule Particle World Model** 的最小研究命题。

- [PRP-WM v0.1：反证优先的研究规格](docs/prp_wm_v0_1.md)
- [PRP-WM Stage 0-B / Stage 1 可复现实验协议 v1.0](docs/prp_wm_reproducible_experiment_protocol_v1.md)
- [RuleGrid pilot v2：真实执行记录与停止结论](docs/rulegrid_pilot_v2_report.md)
- [Latent-rule ceiling：executor、K1/K4 与下一步决策](docs/latent_rule_ceiling_report.md)
- [Causal hypothesis filter：规则猜想、证据筛选与 latent version space](docs/causal_hypothesis_filter_report.md)
- [Expected-discrete K4：把显式规则筛选蒸馏成快速抽象器](docs/amortized_discrete_causal_report.md)
- [Factorization Latin：24-run 全量对照、symbolic interchange 与随机几何下一步](docs/factorization_latin_full_report.md)
- [GRAM causal-rule screen：随机递归假设、width scaling 与 verifier-guided 下一步](docs/gram_causal_rule_screen_report.md)
- [GRAM 顺序证据同化：paired 证伪、独立 verifier 与 proposal blind spots](docs/gram_sequential_assimilation_report.md)
- [JEPA × GRAM × ARC-AGI-3：当前实验思路与进展](docs/jepa_gram_arc_agi3_status.md)
- [接下来 10 小时：cross-axis likelihood locality audit](docs/next_10h_experiment_plan.md)
- [Meta-learning × JEPA：外部 benchmark 与六周研究计划](docs/meta_jepa_external_benchmark_research_plan.md)
- [Meta-learning × JEPA：2026-07-24 第一轮实验报告](docs/experiment_report_2026-07-24.md)
- [PRP-WM 系统 Roadmap：从 factor locality 到 multi-task active inference](docs/prp_wm_system_roadmap_2026-07-24.md)
- [代码架构与 agent 上下文路由](docs/architecture.md)

v0.1 暂不追求直接解决 ARC-AGI-3；它先隔离验证“多种持续规则假设是否能支持更准确的干预预测，以及更省交互的主动辨识”。

## 已完成：Stage 0-A

`GF(2)-RuleProbe` 用四条可枚举规则验证精确 belief update、信息增益动作、数据泄漏边界和配对评测。它只是管线 smoke test，不是 PRP-WM 有效性的证据。

该 reference run 固定使用 Python `3.12.3`；`.python-version`、[`configs/stage0a.json`](configs/stage0a.json) 和 [`results/stage0a_manifest.json`](results/stage0a_manifest.json) 一起构成不可静默漂移的复现契约。

```bash
# Reference host: bypass the local pyenv shim; elsewhere use any Python 3.12.3 executable.
PYTHON=/opt/homebrew/bin/python3
"$PYTHON" --version  # Python 3.12.3
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/run_gate0.py --verify | shasum -a 256
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/verify_stage0a.py
```

第二条命令会重放冻结 config，第三条命令还会检查解释器 patch version、结果字节、config hash 与所有 runtime source hash。它们应分别输出：

```text
77dc5eaded9b3d98b03dddd79cf731e00e4d67076a1a42e12b1e155304fc7c7d
{
  "experiment_id": "prp-wm-stage0a-v1",
  "source_manifest_verified": true,
  ...
}
```

解析真值为：oracle 用 2.0 步辨识规则；budget=12 时 uniform random 的受限平均为 3.3319 步，动作数下降 39.97%。固定种子 Monte Carlo 得到 3.3354 步和 40.04%，通过预注册的 25% headroom gate。

完整固定种子输出保存在 [`results/gate0_seed0.json`](results/gate0_seed0.json)。若需要试验不同参数，可传入 `scripts/run_gate0.py` 的覆盖参数；这类输出不能称为冻结的 Stage 0-A reference run。

## 已完成：Stage 0-B RuleGrid headroom gate

`RuleGrid` 将隐藏规则拆成 collision / trigger / relation 三个四值 axis（共 64 个程序），并把 controller 可见输入与 simulator-only target 分开。完整 Gate 0-B 已在 4,800 个 task、budget=4 上执行，结果保存在 [`results/gate0b_seed0.json`](results/gate0b_seed0.json)：

- exact oracle `RMST_4 = 1.0`；
- exact uniform-without-replacement `RMST_4 = 2.428571`；
- 相对下降 `58.82%`，超过预注册的 `25%` headroom 门槛；
- calibration、neutral support、candidate partitions 与 24-query diagnostic signatures 的全部结构检查均通过。

该结果只证明这个实验台存在主动辨识的 oracle headroom；它不证明神经粒子模型、持久性或主动策略有效。完整 artifact 的 SHA256 为 `836d03a98709b006d74416b39737577ff3cde96837e0c9736d7682979c934729`。

## Bounded K=4 pilot：已停止扩张

真实 GPU 的 500-step Persistent-K4 pilot 可以完整复现，但其 composition holdout 的四行为类审计为 **Coverage@4 = 0.0**：192 个 task 的 768 个兼容行为类中，没有任一 particle 对完整三帧面板达到 coherent MAP exact 与 `≤0.05` nats/cell。该结论比单一 posterior-mode exact 更强，因为审计逐一检查了全部四个 particle，且不向模型输入真实 target。

因此项目不会把更多训练步数、五个 seeds、JEPA 或外部预训练当作修复。下一步是先做可证伪的 rule-executor ceiling 与结构化规则假设空间对照；详见 [pilot report](docs/rulegrid_pilot_v2_report.md) 和 [Coverage artifact](runs/pilot_v2_seed1103_steps500/triple_coverage_audit.json)。

## Latent-rule ceiling：executor 通过，抽象 gate 未通过

后续受控实验发现 raw palette 中存在不可辨识的输出色绑定；在显式标记为 privileged 的 palette-role canonicalization 下，给定三个正确 rule factors 的 pooled 与 spatial executor 都在新 split 的 single / pair / held-out triple 上达到 **100% exact**。

冻结该 executor、去掉 program/factor label 和真实 query target 后，只用 support-derived 无序行为集合训练 latent hypotheses：Persistent-K4 的 held-out Coverage 为 **43.75%（336/768）**，同容量 tied-K1 为 **3.125%（24/768）**；K4 平均产生 `3.83` 个不同 MAP signatures，但 all-four-covered task rate 只有 `10.42%`。这说明多猜想有明确作用，但仍未达到 `90%` 静态 gate。

因此当前决策是：先修 public palette 可辨识性和重复 pair/triple fixtures，再比较连续 latent、受限 codebook 与共享循环深度；静态 coverage 通过前不进入 learned active policy。完整方法、artifact 与 JEPA / AdaJEPA / GRAM / LoopWM 的对应关系见 [latent-rule ceiling report](docs/latent_rule_ceiling_report.md)。

## Causal hypothesis filter：privileged 静态 gate 通过

进一步实验发现，原冻结 executor 虽能 100% 执行 held-out diagnostic queries，却没有在六条 support transition 上校准；真实兼容规则的 support likelihood 排名接近随机。因此旧 causal-slot 训练中的 support-consistency 实际是反信号，四个离散 slot 最终塌缩到约一个规则，Coverage@4 只有 `10.94%`。

新实验把每条 support 同时配给全部四个 compatible factor tuples，训练一个 support-calibrated executor；它在独立 split 上保持 single / pair / triple 100% exact，并使 neural support version space 与 symbolic version space 100% 一致。随后显式枚举 64 个 latent 机制、仅凭 public support 选 top-4，在 192 tasks / 768 held-out behavior classes 上达到 **Coverage@4 = 100%**，打乱 support target 后降到 **7.68%**。

这证明在给定三轴机制空间、oracle palette role 和 privileged executor pretraining 时，“多个规则猜想 → 世界模型验证 → 保留 version space”是可行的；它不证明能从 raw pixels 自主发现这些 axes。下一步应把显式 filter 当 teacher，训练不经 straight-through decoder backward 的 amortized hypothesis generator，再做 public palette binding 与真正的主动 EIG probes。详见 [causal hypothesis filter report](docs/causal_hypothesis_filter_report.md)。

## Expected-discrete K4：显式 filter 已成功摊销

使用完整 64-code detached cost、hard `4!` matching、calibrated support validity 与 full-tuple diversity 后，support-only amortized K4 不再通过 frozen decoder 的 straight-through 切向反传。为排除原 split 的 48-context lookup 捷径，本轮只在 36 个 observed-factor contexts 上训练，在另外 12 个从未见过的 factor-pair contexts 上评估。

累计 600 steps / 4,800 tasks 后，未见 contexts 的 Coverage@4、四规则完整恢复、support consistency 和三个 heldout axes 都达到 **100%**；shuffled-support control 为 `13.02%`。同为 100% exact 时，amortized inference 在当前 CPU 实现上为 `0.237 ms/task`，优化后的 64-code exhaustive filter 为 `5.621 ms/task`，约快 **23.7×**。

这支持“显式猜想—验证可以蒸馏成快速 latent 规则抽象器”，但仍不等于自主发现因果变量：axes、codebook、palette role 和 behavior-set teacher 都是 privileged。进入主动 EIG 前还必须修复 public task/probe ID 泄漏、未启用的 geometry nuisance 与重复 triple fixtures。详见 [expected-discrete report](docs/amortized_discrete_causal_report.md)。

## Factorization Latin：结构优势成立，稳定抽象 gate 未通过

后续 24-run Latin suite 对 `factorized-3x4` 与近参数量的 unstructured rank-5 head 做了 4 folds × 3 seeds 的完整配对。factorized held-out `Coverage@4` 平均为 **75.00%**，unstructured 为 **6.42%**，逐 fold/seed **12/12 胜出**；两者参数仅差 25（`0.0713%`）。但 factorized 的四指标 `≥90%` 静态 gate 只有 **2/12** 次通过，不能称为稳定的规则抽象器。rank-9 与 direct-linear 共 12 个容量补充 run 仍只有 `7.55%` 与 `7.29%` coverage，未支持“只是 unstructured head 太小”的解释。

修正后的 symbolic-code interchange v2 在固定 canonical 几何上执行 exact 为 **100%**，support exact version-space 只有 **49.31%**、code recall 为 **75%**、source/donor support compatibility 约 **82%**；换到 512 个随机几何 execution cases 后 exact 降至 **28.32%**。因此当前首要下一步不是继续扩大旧固定模板或立即加入 TTT，而是按已通过审计的 random-geometry protocol，在 singleton/pair 上训练、冻结后只评估 disjoint triple 几何。完整数值、SHA、限制与实验协议见 [Factorization Latin full report](docs/factorization_latin_full_report.md)。

## GRAM causal-rule + 顺序证据同化：外循环成立，proposal 未通过

在同一 fold、seed 和 600-step 预算下，GRAM 式 posterior/prior、随机 guidance 与共享递归深度产生了平均 `3.40` 个不同候选，但 held-out Coverage@4 只有 **13.54%**、valid particle rate 只有 **18.75%**；配对固定四槽 K4 分别为 **91.67%** 与 **93.75%**。将总 trajectory width 从 4 增至 32 只把 compatible-rule recall 提高到 **28.13%**，四规则完整覆盖仍为零。

因此失败点不是缺少表面多样性，而是随机分支没有覆盖 public evidence 仍允许的机制。后续实验先否决了不具 proposal correction / MH kernel 的“SMC”表述，改为路径无关的 verifier-guided population search，并训练了一个与 GRAM 内部 executor 分离的 active-prefix verifier；后者在 192 个 held-out tasks 的 t0–t3 上都能 100% 复原 symbolic version space。

在同一 48 tasks、三个 inference seeds、每阶段 W32 的 paired factual/counterfactual 顺序证据实验中，GRAM + verifier + carry 的 t3 target coverage 为 **38.19% / 45.83%**，而 matched uniform-proposal control 为 **79.17% / 84.72%**。若 t2 仍缺 target，GRAM 经 strong evidence 的恢复率只有 **2.20% / 0%**，uniform 为 **36.17% / 31.25%**。逐 proposal 审计发现 `collision=STOP` 从未被提出、`relation=NONE` 仅占约 1–2%，并存在大量 joint-code holes。

所以快速“观察—验证—保留”骨架有价值，但当前 GRAM proposal 是瓶颈。后续 matched-W32 实验进一步发现：GRAM 的 fresh strong-evidence recovery 只有 **2.30%**，加入更多 uniform proposals 会单调改善，纯 uniform 为 **41.06%**；pairwise-balanced Latin bank 的累计 retained coverage 达 **92.01%**。public-only full-version-space coverage continuation 也已实测：全路径更新把 held-out W32 recall 从 **31.25% 降到 18.23%**，只更新 prior head 可小幅提高到 **32.29%**，但 STOP 仍为零、fresh recovery 仅 **10.92%**。

因此不再给当前 iid-Gaussian proposal / soft-W4 coverage loss 增加训练步数。下一架构需要显式 persistent/stratified hypotheses、hard unique joint coverage 和永久 uniform/Latin exploration floor；通过 fresh recovery gate 后才进入 EIG action selection，TTT 继续后置。原始静态实验见 [GRAM causal-rule screen report](docs/gram_causal_rule_screen_report.md)，顺序证据、混合 proposal 与 coverage continuation 的完整结果见 [GRAM sequential assimilation report](docs/gram_sequential_assimilation_report.md)。

## 实验性 baseline：RL from scratch

[`prp_wm/rl.py`](prp_wm/rl.py) 提供一个与 PRP-WM 解耦的稀疏奖励
actor-critic baseline。控制器只读取公开的原始 `8×8` 网格、动作、已观察转移和
候选可用 mask；它不读取 rule ID、version space、candidate kind、diagnostic target
或 oracle EIG，也不使用 reconstruction loss。simulator 只在奖励边界判断行为类是否
已经辨识。

```bash
/Users/yangzhenbang/anaconda3/bin/python3 scripts/train_rulegrid_rl.py \
  --steps 2000 --batch-size 32 --device cpu \
  --output runs/rl_scratch_seed20260716
```

该 baseline 用于检验“原始网格上的纯 RL 是否能直接学会选择诊断动作”，不能单独
证明内部形成了对象表示、世界模型或多假设 belief。正式比较必须使用独立 split，
并同时报告 uniform-without-replacement baseline。首个 smoke run 虽达到 100%，但
删除 calibration history 后成绩不变，确认了 candidate-only shortcut；详见
[RL-from-scratch smoke report](docs/rulegrid_rl_scratch_smoke_report.md)。

## 终局获胜奖励：RuleGame

[`prp_wm/rulegame.py`](prp_wm/rulegame.py) 新增了一个与规则辨识奖励解耦的最小游戏。
四种隐藏规则共享逐像素相同的初始与终局画面，只有中间实验结果不同；前两步奖励恒为
零，只有选对终局门才获得 `1`。[`prp_wm/rulegame_rl.py`](prp_wm/rulegame_rl.py)
实现 recurrent PPO 与 GRPO。

单 seed smoke 中，PPO 用 19,200 条轨迹达到 98.05% validation win；GRPO 在相同轨迹
预算下为 25%，增加到 76,800 条轨迹后达到 91.41%。两者在终局前清空 memory 后均降到
25%，说明成功策略确实依赖视觉历史。详见
[RuleGame win-only RL report](docs/rulegame_win_only_rl_report.md)。
