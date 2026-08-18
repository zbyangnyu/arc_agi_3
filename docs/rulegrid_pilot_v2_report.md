# RuleGrid pilot v2：真实执行记录与停止结论

> 日期：2026-07-16  
> 状态：**bounded pilot，非预注册 Stage 1 结果**  
> 决策：Coverage@4 已完成且为零；不进入 5-seed 正式训练，也不靠增加训练步数继续扩张。下一步必须是有明确可证伪假设的机制修订。

这次运行的作用是验证真实的 RuleGrid → Persistent-K4 → checkpoint → triple holdout 链路，并尽早寻找反证，而不是证明 H1、H2 或 H3。

## 可复现 artifact

- 训练配置与最终指标：[train_summary.json](../runs/pilot_v2_seed1103_steps500/train_summary.json)
- 逐步训练日志：[progress.jsonl](../runs/pilot_v2_seed1103_steps500/progress.jsonl)
- 三因素 holdout：[composition_eval.json](../runs/pilot_v2_seed1103_steps500/composition_eval.json)
- 四行为类覆盖审计：[triple_coverage_audit.json](../runs/pilot_v2_seed1103_steps500/triple_coverage_audit.json)，reference GPU artifact SHA256 `0343d436111895c43d68cb80364595404b5852859ab45689f5baf7d8a8005caf`
- checkpoint：[`checkpoint_last.pt`](../runs/pilot_v2_seed1103_steps500/checkpoint_last.pt)，SHA256 `53245ea30b28b4ffe49a8fec6f60f5ecec8aba1eff4aaac4901b8232ff1eafe5`
- 稀疏网格 copy 诊断：[pilot_v2_seed1103_copy_baseline.json](../results/pilot_v2_seed1103_copy_baseline.json)
- 两次同 seed、2-step CUDA smoke 的 checkpoint SHA256 完全相同：`7a7c0dbfeeab8b5089aca96d6c9e3198e9da4ce6fb1006cf23ea44e028f478c1`。
- 在同一 reference GPU 与输出路径下重跑 Coverage 审计，JSON SHA256 也完全相同：`0343d436111895c43d68cb80364595404b5852859ab45689f5baf7d8a8005caf`。

主运行使用 RTX 4090、CUDA 12.8、PyTorch 2.8.0、Python 3.12.3；训练代码与 evaluator 的 SHA256 已写入 checkpoint 和两个 JSON artifact。训练的 `data_master_seed=2026071601` 与 `model_seed=1103` 分离，且 evaluator 强制其数据 seed 与 checkpoint 一致。

在具有匹配 PyTorch/CUDA 环境的干净 checkout 中，可用下列命令重跑同一 bounded pilot：

```bash
python scripts/train_rulegrid_pilot.py \
  --device cuda --steps 500 --batch-size 16 --seed 1103 \
  --log-every 25 --checkpoint-every 100 \
  --output runs/pilot_v2_seed1103_steps500

python scripts/eval_rulegrid_pilot.py \
  --checkpoint runs/pilot_v2_seed1103_steps500/checkpoint_last.pt \
  --device cuda --tasks 192 --batch-size 16 --seed 20260717 \
  --data-master-seed 2026071601 \
  --output runs/pilot_v2_seed1103_steps500/composition_eval.json

python scripts/eval_rulegrid_copy_baseline.py \
  --split pilot-composition --tasks 192 --diagnostic-indices 21 22 23 \
  --data-master-seed 2026071601 \
  --output results/pilot_v2_seed1103_copy_baseline.json

python scripts/eval_rulegrid_coverage_audit.py \
  --checkpoint runs/pilot_v2_seed1103_steps500/checkpoint_last.pt \
  --device cuda --tasks 192 --batch-size 16 --seed 20260718 \
  --data-master-seed 2026071601 --split pilot-composition \
  --output runs/pilot_v2_seed1103_steps500/triple_coverage_audit.json
```

## 训练与 holdout 边界

训练共 500 optimizer steps、batch size 16、8,000 task bundles、193,377 参数的紧凑 K=4 profile，耗时 228.92 秒。训练时只构造并读取 diagnostic indices `0..20`（single/pair）；三因素 indices `21..23` 没有被模拟或存储在训练 task 中。这个边界由构造次数单测、subset tensor test、checkpoint metadata 和 evaluator audit 共同约束。

holdout 使用独立 `pilot-composition` stream 的 192 tasks（完整的 64 program × 3 held-out-axis nuisance group），只构造和读取 triple targets `21..23`。因此它不是把训练实例的 target 重新拿来评分。

## 结果

| 项目 | 数值 |
|---|---:|
| 训练最终 total loss | 0.13933 |
| Triple joint predictive NLL / cell | 0.18347 |
| Triple per-query mixture NLL / cell | 0.18347 |
| coherent exact-grid accuracy | 0.00000 |
| coherent-mode cell accuracy（辅助） | 0.93913 |
| copy cell accuracy | 0.92969 |
| copy exact-grid accuracy | 0.00000 |
| 真实变化格比例 | 0.07031 |
| mode entropy | 1.38629 nats（约为 `ln(4)`） |
| Coverage@4（质量加权） | 0.00000 |
| Coverage@4（不加权行为类） | 0.00000（0 / 768） |
| 覆盖全部四类的 task rate | 0.00000（0 / 192） |
| 任一 mode 三帧 MAP 全 exact（辅助） | 0.00000（0 / 768 类） |
| 任一 mode 达到 `≤0.05` nats/cell（辅助） | 0.00000（0 / 768 类） |

“coherent exact-grid”先仅依据 support posterior 选定一个 mode，再用该 mode 对整帧取 MAP；它不会把不同 particle 的逐格 argmax 拼成并不存在的 frame。copy baseline 没有定义可比较的有限 NLL（它在任一变化格上赋零概率），所以这里只把 cell accuracy 作为稀疏背景检查，不把两者的 NLL 混为一谈。

Coverage 审计则比这个单一 posterior-mode 指标更强：它从可见的 six-transition support 重新枚举四个兼容程序，按三个公开 triple probe 构造四个替代行为面板，再对 **全部四个** 学习到的 particle 逐个评分。模型输入中没有 query target 或 behavior target；真程序和真实 target sidecar 不参与 inference 或 mode 选择。192 个 task 都严格有四个兼容程序、四个行为类，且每类质量都是 0.25；checkpoint 的核心训练源码 hash 也逐项验证通过。

## 解释与停止条件

这个小模型相对 copy 只多了约 **0.94 个百分点**的 cell accuracy，且 576 个真实 triple frame 中没有一帧 exact。更关键的是，Coverage 审计已排除“只是 support posterior 挑错 particle”的解释：四个 particle 都被逐一尝试，仍没有任何一个 particle 对任一兼容的完整三帧行为面板做到 MAP exact，也没有一个达到冻结的 `0.05` nats/cell 门槛。它尚未形成可执行、完整的三机制规则预测；低训练 loss 不能抵消这个事实。

因此当前允许的结论只有：

- RuleGrid benchmark 的 oracle headroom 已通过；
- K=4 的真实 GPU 训练、严格 composition holdout、确定性 checkpoint 和 coherent evaluator 都可以运行；
- 这个 500-step 紧凑 pilot **没有**提供多假设、持久性、组合归纳或主动探索有效的证据；其当前 `Coverage@4=0` 直接不满足正式 Gate 1 的先决条件。

不应由这个结果推出“增加步数、JEPA 或预训练就会解决问题”。这里失败的是离散、已知调色板规则上的行为覆盖，而不是视觉表征瓶颈；此时加入外部预训练会改变变量、掩盖机制问题。v1 正式实验仍保持 `external_pretrained = null`。

## 下一次可证伪改进

Coverage gate 已经执行且失败，所以不能把 20k/50k step 或 five-seed sweep 当作“下一步”。应先以新版本记录一个更窄的机制假设，并按以下顺序检验：

1. **rule-executor ceiling**：用仅供诊断的 oracle factor/rule code 训练同一个 decoder，确认它是否能对三机制面板达到 near-exact。若这个 ceiling 也失败，先改状态/动作编码或 decoder，而不是讨论 particle。
2. **结构化假设空间**：把纯连续、交换对称的 particle latent 改为可组合的 collision / trigger / relation 因子（或受限离散 codebook），并用一对一的完整行为面板分配训练。该实验检验的是“规则归纳 bias”而非单纯扩大容量。
3. 只有修订模型先在静态 six-transition support 上取得非零、稳定的 Coverage@4，才加入真实 strong / partial / neutral active prefixes `7..10`，再比较 `K1-capacity-matched`、`SingleBelief-4Head` 与 `Reinfer-K4`。
4. 只有 H1 的 coverage 先通过，才冻结 checkpoint 并评估 uniform / coverage / change-seeking / mode-info 策略及 `RMST_4`；否则 H3 没有可解释的基础。

这条顺序保留了当前反证：先验证粒子是否真的覆盖行为，再讨论 persistence 或探索是否带来额外收益。此时不加入 JEPA 或外部预训练；输入已是离散、已知调色板，当前失败不是视觉表征瓶颈。
