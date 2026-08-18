# PRP-WM Stage 0-B / Stage 1 可复现实验协议 v1.0

> 协议日期：2026-07-16  
> 状态：Stage 0-A 与 Stage 0-B 已实现并有固定 reference artifact；一个 bounded pilot 已执行，且其冻结的 `Coverage@4` 审计为零，正式 Stage 1（基线、五个 seed、主动闭环和统计）尚待实现。  
> 核心原则：任何人在只拿到代码、锁定环境、数据 manifest 和本文命令后，应能重新生成逐任务结果、汇总表、置信区间和相同的 Gate 判定。

## 0. 先区分“已复现”与“已设计”

Stage 0-A 可以实际执行并达到字节级复现；Stage 0-B 的 exact oracle gate 也已有完整 fixed-seed artifact。以下 Stage 0-A 命令已在 Python 3.12.3、Darwin arm64 上验证：

```bash
cd /path/to/arc-agi-3
# Reference host: use /opt/homebrew/bin/python3 (Python 3.12.3), not its pyenv shim.
PYTHON=/opt/homebrew/bin/python3
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/run_gate0.py --verify | shasum -a 256
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/verify_stage0a.py
```

预期结果：

```text
source_manifest_verified: true
77dc5eaded9b3d98b03dddd79cf731e00e4d67076a1a42e12b1e155304fc7c7d  -
```

该 hash 应与 `results/gate0_seed0.json` 的 SHA256 完全一致。`verify_stage0a.py` 还必须报告 `source_manifest_verified: true`；它检查 Python patch version、冻结 config、参考结果和所有 runtime source file 的 SHA256。

Stage 0-B 的 minimal exact implementation 已覆盖 generator、version-space oracle、uniform subset-DP 与结构性 hard-fail tests；其 reference artifact 位于 [`../results/gate0b_seed0.json`](../results/gate0b_seed0.json)。正式 Stage 1 仍未完成，因此：

- 可以声称 Stage 0-B oracle headroom 已复现，但不能将其写成 learned-model result；
- 在 Stage 1 的数据物化、基线、五 seed、主动闭环与统计都完成前，不能声称 H1/H2/H3 已复现；
- 任何数据定义、阈值、主要指标或主要比较发生修改，都必须提升协议版本，不能覆盖 v1.0。

### 0.1 已执行的 Stage 0-B reference gate

执行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_gate0b.py \
  --repeats 25 --budget 4 --bootstrap-resamples 2000 \
  --seed 2026071701 --output results/gate0b_seed0.json
```

固定结果的 SHA256 为 `836d03a98709b006d74416b39737577ff3cde96837e0c9736d7682979c934729`。在 4,800 tasks 上，exact oracle `RMST_4=1.0`，exact uniform `RMST_4=2.4285714285714284`，相对下降 `58.8235%`，并通过 25% Gate 0-B。所有 calibration / neutral / candidate partition / diagnostic-signature 检查均为真。

这一步仅验证 benchmark 的可辨识 headroom；不涉及神经训练，也不构成 H1、H2 或 H3 的证据。

### 0.2 复现等级

| 等级 | 定义 | 本项目要求 |
|---|---|---|
| R1 字节复现 | 相同源码、环境和硬件栈下，输出及 hash 相同 | Stage 0-A 必须达到 |
| R2 数值复现 | 相同 CUDA/GPU 栈下，指标在冻结容差内相同 | 单个神经训练 run 必须达到 |
| R3 统计复现 | 跨同等级 GPU/操作系统复跑 5 seeds，效应方向、CI 与 Gate 判定一致 | Stage 1 最终结论必须达到 |

## 1. 研究问题与唯一允许的结论

本协议不验证“模型能否通关 ARC-AGI-3”，只验证三个可证伪命题。

### H1：多模态规则表示

在支持历史仍兼容多个行为规则时，`Persistent-K4` 是否比计算量匹配的 single-belief 多峰模型更好地覆盖完整、相干的反事实结果？

主要比较：

```text
Persistent-K4 vs validation 预选的最强 single-belief baseline
```

### H2：跨时间持久性

在两者都能输出四个相干模式时，递归保留 mode lineage 是否比每个 prefix 从完整历史重新推断四个无序 mode 更稳定，并带来预测收益？

主要比较：

```text
Persistent-K4 vs Reinfer-K4
```

### H3：主动证伪

冻结同一个 `Persistent-K4` checkpoint 后，full-state mode information 策略是否比最强非 oracle 策略用更少真实动作达到行为预测等价？

主要比较：

```text
同一 checkpoint：mode-info policy vs validation 预选的最强非-oracle policy
```

`test-Composition` 是三个假设唯一的主要 split。`test-ID`、`test-Shape` 与 `test-Both` 是支持性结果，不能替代主要 split。

## 2. 整体执行顺序

```text
Stage 0-A：验证评测和统计管线
        ↓ 通过
Stage 0-B：验证规则 benchmark 有 oracle headroom
        ↓ Gate 0-B 通过
超参数调优：只使用 validation
        ↓ 冻结每个方法配置
Stage 1：5 个全新训练 seed
        ↓
表示评估：H1、H2
        ↓ 冻结 checkpoint 与 calibration
主动闭环评估：H3
        ↓
分层配对 bootstrap + Holm 校正
        ↓
发布逐任务数据、预测、manifest 和最终 Gate 表
```

如果某个 Gate 失败，停止其下游主张：

- Gate 0-B 失败：不训练神经模型；
- Gate 1 失败：不能主张多假设表示有效；
- Gate 2 失败：可以报告多峰预测，但不能主张 persistence 有独立价值；
- Gate 3 失败：可以报告预测模型，但不能主张主动证伪更省交互。

## 3. Stage 0-B / Stage 1 共用的 RuleGrid 实验台

### 3.1 为什么使用“可重置实验台”

Stage 1 暂不测试导航或到达 probe state 的规划。每个候选 probe 自带一个输入网格和一个动作，执行后返回一个输出网格；不同 probe 之间重置画面，但隐藏规则和 belief 跨 probe 持续。这样唯一被操纵的变量是规则归纳与实验选择。

| 项目 | 固定值 |
|---|---:|
| benchmark version | `prp-rulegrid-v0.2.0` |
| 网格 | `uint8[8,8]` |
| 颜色 | `0..15`，`0` 是背景 |
| 转移 | deterministic |
| 隐藏程序 | 3 个 axis、每 axis 4 个 mode，共 64 个程序 |
| 初始 support | 2 个 calibration + 4 个 neutral transitions |
| active candidate bank | 8 个 probes，无放回选择 |
| active budget | `B=4` |
| diagnostic panel | 24 个 probes |
| 外部预训练 | `null` |

输入 controller 的 `InferenceView` 只能包含已观察转移、候选 probe 的输入网格/动作和公开 candidate ID。规则程序、probe kind、候选 target、诊断 target、version space 和 oracle EIG 均属于 `PrivilegedTargets`。controller 尝试访问它们必须抛异常。

### 3.2 隐藏程序与编号

```text
rule = (collision, trigger, relation)
collision ∈ {STOP, BOUNCE, PASS, PUSH}
trigger   ∈ {TOGGLE, DELETE, SPAWN, RECOLOR}
relation  ∈ {SWAP, FOLLOW, REPEL, NONE}

program_id = 16 * collision_id + 4 * trigger_id + relation_id
```

ID 按表中顺序从 0 开始。颜色角色在一个 task 内固定，跨 task 随机置换到 `1..15`；动作只包含坐标、方向和公开 action type，不能包含对象角色或隐藏 mode 名称。

### 3.3 Collision 的精确语义

沿动作方向 `d` 的标准局部布局：

```text
p-d: empty
p:   actor
p+d: blocker
p+2d: empty
```

| mode | 原子结果 |
|---|---|
| STOP | 不变 |
| BOUNCE | actor 从 `p` 移到 `p-d` |
| PASS | actor 从 `p` 移到 `p+2d`，blocker 不变 |
| PUSH | actor 从 `p` 移到 `p+d`，blocker 从 `p+d` 移到 `p+2d` |

任一目标格越界或被非规定对象占用时，该 mode 的整个局部操作为 no-op，不执行部分写入。

### 3.4 Trigger 的精确语义

状态包含 trigger、payload `P0` 和 spawn socket，动作是 `ACTIVATE(trigger_coord)`：

| mode | 原子结果 |
|---|---|
| TOGGLE | payload `P0↔P1` |
| DELETE | payload 变背景 |
| SPAWN | 原 payload 保持，socket 变 `P0` |
| RECOLOR | payload 变 `P2` |

trigger 本身不变；缺少该 mode 所需的 payload 或 socket 时，该局部操作为 no-op。

### 3.5 Relation 的精确语义

沿 `d` 的标准布局：

```text
p-d: empty
p:   object A
p+d: object B
p+2d: empty
```

动作是 `MOVE(A,d)`：

| mode | 原子结果 |
|---|---|
| SWAP | A、B 交换位置 |
| FOLLOW | A 从 `p` 到 `p+d`，B 从 `p+d` 到 `p+2d` |
| REPEL | A 不动，B 从 `p+d` 到 `p+2d` |
| NONE | 不变 |

对象按刚体平移；任一目标像素非法时，整个局部操作为 no-op。

### 3.6 多机制组合与干扰变化

一个 diagnostic query 可以包含多个互不重叠的局部事件。每个事件先独立计算 delta，再取写集合并集；生成器必须保证写集合不重叠，否则该布局非法，不能靠任意执行顺序解冲突。

每个 probe 还可以携带一个 rule-independent pulse region。执行动作时，该区域的 `D0/D1` 像素互换；它不参与任何隐藏规则。neutral-large-change probes 用 pulse 保证变化像素不少于 strong probes 的中位数，但四个候选 mode 的完整结果仍逐格相同。pulse 的大小、位置和颜色不能依赖真实 mode。

### 3.7 对象形状和几何变换

```text
S0 = {(0,0)}
S1 = {(0,0),(0,1)}
S2 = {(0,0),(1,0),(1,1)}
```

| split | shape |
|---|---|
| train / validation-ID / test-ID | `S0`、`S1` 严格各半 |
| Shape-OOD | `S2` |
| Composition-OOD | `S0`、`S1` 严格各半 |
| Both-OOD | `S2` |

每个 template 在 D4 的 8 个旋转/镜像上严格均衡。生成器枚举所有合法平移，按 `(row,column)` 排序后由固定 RNG 选择；再从剩余空格放 distractor。任何 rejection 或布局选择都不能读取真实 mode。

### 3.8 Episode 的固定组成

每个 task 先均衡选择一个 `heldout_axis ∈ {collision,trigger,relation}`：

| 部分 | 数量 | 规范 |
|---|---:|---|
| calibration support | 2 | 非 heldout 的两个 axis 各一个 strong probe |
| neutral support | 4 | `C/T/R/heldout-axis` 各一个 neutral probe，顺序打乱 |
| active bank | 8 | 2 strong、2 partial、4 neutral-large-change |
| active budget | 4 | 无放回选择 |
| diagnostic panel | 24 | 12 single、9 pair、3 triple |

读取两个 calibration 后，精确 version space 必须恰好包含 heldout axis 的 4 个 modes。四个 neutral support 不得缩小 version space。训练与身份评估保留 prefixes `[2,3,4,5,6]`；主动阶段从完整 6-transition support 开始。

active bank 的 outcome partitions 固定为：

- strong ×2：`1+1+1+1`；
- partial ×2：`1+3`，singleton mode 在 nuisance packs 中严格均衡；
- neutral ×4：`4`，且 visible change 不小于 strong 的中位数。

probe template 通过有限枚举构造：对每个 axis、shape 和 D4 变换枚举合法局部占用，按 outcome partition 过滤，再按未置换的 `state_bytes, action_bytes` 字典序取 canonical template。probe kind 仅写 privileged sidecar，绝不进入模型输入。

训练任务额外固定一个与真实 mode 独立的 candidate permutation；取前 4 个 probe 的真实结果形成训练 active prefixes `[7,8,9,10]`。每次训练样本从 `[2,3,4,5,6,7,8,9,10]` 均匀选 prefix，使 updater 同时看到歧义、无信息证据和证伪证据。

### 3.9 Diagnostic panel 与 Composition-OOD

24 个诊断输入的 canonical 顺序：

```text
0..11   single: C/T/R 各 4 个
12..20  pair: CT/CR/TR 各 3 个
21..23  triple: CTR 共 3 个
```

train、validation-ID 和 test-ID 的训练/主评分只使用 `0..20`；`21..23` 的 targets 不进入任何训练 loss、模型选择或 beta calibration。`Composition-OOD` 的主要评分只使用三个 triple queries，full-24 指标作为补充。因而模型见过所有 primitive 和 pair composition，但从未被监督过 triple composition。

## 4. 固定数据、随机流与文件契约

### 4.1 数量

| split | programs | heldout axes | repeats/stratum | tasks |
|---|---:|---:|---:|---:|
| train | 64 | 3 | 500 | 96,000 |
| validation-ID | 64 | 3 | 25 | 4,800 |
| test-ID | 64 | 3 | 50 | 9,600 |
| test-Shape | 64 | 3 | 50 | 9,600 |
| test-Composition | 64 | 3 | 50 | 9,600 |
| test-Both | 64 | 3 | 50 | 9,600 |
| gate0b | 64 | 3 | 25 | 4,800 |

核心实验使用全因子 64-program 训练，避免 split 本身泄露 rule prior。未见 mode-tuple 泛化必须另开 `Tuple-OOD` 协议，不能混进本实验的 Composition-OOD 结论。

### 4.2 固定 seeds 与派生算法

```text
benchmark_version = "prp-rulegrid-v0.2.0"
master_seed        = 2026071601
tuning_seeds       = [101, 211, 307]
final_model_seeds  = [1103, 2207, 3301, 4409, 5519]
policy_seeds       = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
bootstrap_seed     = 2026071701
```

每个 nuisance 随机流：

```text
seed64 = little_endian_uint64(
  SHA256(
    "prp-rulegrid-v0.2.0|2026071601|"
    + split + "|" + heldout_axis + "|"
    + replicate + "|" + stream_name
  )[0:8]
)
rng = numpy.random.Generator(numpy.random.PCG64DXSM(seed64))
```

`stream_name` 只能是：

```text
palette, geometry, support_order, candidate_order, diagnostic, rollout
```

关键约束：nuisance seed 不包含 `program_id`。同一个 `(split, heldout_axis, replicate)` 对全部 64 个程序复用相同布局、palette 和 candidate order，从结构上保证 nuisance 与隐藏规则独立。

### 4.3 Dataset 文件

公开 inference JSONL 与 privileged JSONL 必须分文件、分 dataclass 和分 API。稳定 ID：

```text
task_id = split + "/P" + zero_pad(program_id, 2)
          + "/H" + heldout_axis_id
          + "/N" + zero_pad(replicate, 4)
```

manifest 必须记录 benchmark version、生成器 source hash、Python/NumPy 版本、完整 config hash、每个未压缩 JSONL 的 SHA256，以及 program/axis/shape/D4 频数。validation/test/gate0b 必须物化，评估时禁止重新采样。

### 4.4 生成器 hard-fail tests

1. 同 seed 两个独立进程生成的未压缩 JSONL SHA256 完全一致；
2. 各 split 的 task ID 和 inference hash 无禁止重叠；
3. calibration 后 `|V|=4`；
4. neutral support 后 version space 不变；
5. strong probe 恰有 4 个 distinct outcomes；
6. partial probe partition 恰为 `1+3`；
7. neutral probe 对 4 modes 的完整结果逐格相同；
8. neutral change-cell 数不小于 strong 中位数；
9. full-24 diagnostic signature 对 64 programs 唯一；
10. nuisance pack 对同一 heldout axis 的 4 个 true modes 字节级相同；
11. 只看 state/action、不看 support target 的 mode classifier accuracy 不超过 27%，且 Wilson CI 包含 25%；
12. future target 扰动不改变任何更早 prefix 的 prediction、weight 或 action；
13. controller 访问 privileged 字段必须抛异常；
14. simulator 重放每个公开 transition 后逐格等于存储 target；
15. train 不含 triple diagnostic target 的 loss 或 calibration 访问路径。

## 5. 精确 oracle 与 Gate 0-B

历史 `D` 后：

```text
V(D) = 对全部已见 transition 给出相同结果的所有 64 programs
```

停止条件不是 true `program_id` 唯一，而是剩余程序在 full-24 diagnostic panel 上的行为签名唯一。对候选 probe `a`：

\[
EIG(a)=H(Z\mid D)-\sum_y p(y\mid D,a)H(Z\mid D,a,y),
\]

其中 `Z` 是 diagnostic behavior class，信息单位为 bits。并列按公开 candidate ID 最小者决定。

在 4,800 个独立 `gate0b` tasks、budget 4 上比较 exact oracle 与 uniform-without-replacement。uniform 不做 Monte Carlo；对 8 个候选使用最多 `2^8=256` 个子集的动态规划计算 exact `RMST_4`：

\[
RMST_4=\sum_{t=0}^{3}P(T_{id}>t).
\]

Gate 0-B 同时要求：

1. 每个 task 在 calibration 后 `|V|=4`；
2. 每个 task 恰有 2 strong、2 partial、4 neutral-large-change probes；
3. oracle 相比 exact uniform 的 `RMST_4` 至少下降 25%；
4. 配对、按 `heldout_axis × true_mode` 分层 bootstrap 的 `uniform-oracle` 95% CI 下界大于 0。

任一条件失败，修改生成器并提升协议版本，不进入神经训练。

## 6. 主模型：Persistent-K4

### 6.1 冻结架构

```text
grid_size             = 8
num_colors            = 16
color_embedding       = 64
row_embedding         = 64
column_embedding      = 64
encoder_channels      = 64
encoder_resblocks     = 4
kernel_size           = 3
normalization         = GroupNorm(groups=8)
activation            = SiLU
downsampling          = false
action_embedding      = 32
K                     = 4
rule_dim              = 128
updater               = shared GRUCell(128)
particle_interaction  = 1 set-attention layer, 4 heads, FFN 256
decoder               = shared FiLM-ResNet, 4 blocks, 64 channels
dropout               = 0
external_pretrained   = null
```

输入 embedding 使用 `color + row + column` 相加。四个 modes 共享 encoder、updater 和 decoder；`[rule_mode, action_embedding]` 为每个 decoder block 产生 FiLM 参数。不同 K 不能拥有独立 decoder。

目标规模约 1.2M–1.8M 参数。实际参数量、训练 FLOPs、每动作推理 FLOPs和峰值显存必须写入 run manifest；估算值不能代替实测值。

### 6.2 Proper outcome distribution

每格输出一个 change Bernoulli 和一个排除原颜色的 new-color distribution：

\[
p(y_{ij}\mid x_{ij},r_k,a)=
\begin{cases}
1-c_{ij}^k,&y_{ij}=x_{ij},\\
c_{ij}^kq_{ij}^k(y_{ij}),&y_{ij}\ne x_{ij}.
\end{cases}
\]

`q` 中原颜色 logit 必须设为 `-inf`。用于 mode weight 的 score 不能使用真实 change mask 决定的类别加权。

### 6.3 Mode 更新与评分顺序

每个真实转移必须严格按下列顺序执行：

```text
1. 用 D_<t 的 modes 预测 y_t
2. 保存 prequential loss
3. 用 y_t 形成每个 mode 的 spatial residual
4. updater 提议新的 modes
5. 从任务初始 mode 重新 replay D_≤t，为移动后的 modes 计算完整历史 evidence weight
```

历史重放权重：

\[
s_t^k=\log(1/K)+\beta
\sum_{i<t}\frac{1}{HW}S_{ik},\qquad
w_t=\operatorname{softmax}(s_t).
\]

不能同时使用包含旧证据的 `w_{t-1}` 作为 prior 后再重放全部历史，否则会重复计数。

## 7. 冻结训练目标

mode `k` 对一帧的 proper log score：

\[
S_k(x,a,y)=\sum_{ij}\log p_k(y_{ij}\mid x,a).
\]

### 7.1 Support prequential mixture loss

\[
L_{support}=
-\frac1t\sum_{i<t}\frac1{HW}
\log\sum_k w_i^k\exp S_{ik}.
\]

必须先计算该项，才能把 `y_i` 交给 updater。

### 7.2 Joint diagnostic-query loss

\[
L_{joint}=
-\frac1{QHW}\log\sum_k w_k
\exp\left(\sum_{q=1}^{Q}S_{qk}\right).
\]

同一 mode 必须联合解释当前评分集合中的全部 query，不能逐 query 重新匹配。训练、validation-ID 与 test-ID 使用 21 个 single/pair queries；Composition-OOD 的主要分数使用 3 个 triple queries；full-24 只作行为签名与补充指标。

### 7.3 行为类 assignment loss

\[
C_{km}=-\frac1{QHW}\sum_q\log p_k(Y_m^q\mid x_q,a_q).
\]

对 `K=4` 直接枚举最多 `M^K≤256` 个 row-hard assignments：

\[
L_{assign}=\min_z\left[
\sum_kw_kC_{k,z_k}
+0.25D_{KL}\left(
\mu\middle\Vert
\frac{\hat\mu(z)+10^{-4}}{1+M10^{-4}}
\right)
\right].
\]

总损失：

\[
L=L_{support}+1.0L_{joint}+0.5L_{assign}.
\]

v1.0 明确禁止加入：

- diversity loss；
- latent persistence regularizer；
- future/hindsight weight target；
- JEPA loss；
- value、risk、policy loss；
- rule-ID classification loss；
- 测试时梯度更新。

训练时 `β=1`。训练完成后只在 validation 上从

```text
[0.25, 0.5, 1, 2, 4, 8]
```

选择 diagnostic mixture NLL 最低的 `β`，随后冻结到全部 test。

## 8. 优化器、调参与最终训练

### 8.1 每个方法相同的调参预算

```text
learning_rate ∈ [1e-4, 3e-4]
lambda_assign ∈ [0.25, 0.5, 1.0]
tuning seeds = [101, 211, 307]
steps per tuning run = 20,000
```

共 6 个配置 × 3 个 seed。选择规则按顺序执行：

1. 最小化 4,800 个 validation-ID tasks 上的 21-query joint NLL；
2. 差异小于 `0.005 nats/cell` 时选参数更少的配置；
3. 参数量相同时选较小的 `lambda_assign`；
4. 禁止查看 test。

### 8.2 最终训练配置

```text
optimizer                  = AdamW
betas                      = [0.9, 0.95]
eps                        = 1e-8
weight_decay               = 0.01
no_decay                   = bias, norm, embedding
micro_batch_size           = 32 task bundles
gradient_accumulation      = 4
effective_batch_size       = 128
optimizer_steps            = 50,000
warmup_steps               = 2,000
lr_schedule                = cosine
minimum_lr                 = 0.1 × peak_lr
gradient_clip_norm         = 1.0
precision                  = fp32
TF32                       = false
EMA                        = false
early_stopping             = false
selected_checkpoint        = final step 50,000
final seeds                = [1103, 2207, 3301, 4409, 5519]
```

每个最终 run 恰好消费 6.4M 个 task bundles。对 96,000 个训练 task，每个 epoch 使用由 `SHA256(model_seed, "train_order", epoch)` 派生的 PCG64DXSM permutation；每个 batch 的 prefix 从 `[2,3,4,5,6,7,8,9,10]` 用独立的 `prefix` 随机流均匀选择。相同 model seed 的所有方法看到完全相同的 task/prefix 序列。中途 checkpoint 只用于故障恢复，不能根据 test 或最好看的 validation 点挑 checkpoint。

确定性设置：

```text
PYTHONHASHSEED=<train_seed>
CUBLAS_WORKSPACE_CONFIG=:4096:8
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
DataLoader(num_workers=0)
```

checkpoint 必须包含模型、optimizer、scheduler、step、data-stream position 和所有 RNG state。CI 必须验证“连续 N 步”与“N/2 步保存恢复后继续”数值一致。

## 9. 必须实现的表示基线

### 9.1 主要基线

1. `SingleBelief-4Head`：一个递归 belief latent，输出 4 个 coherent heads；
2. `Reinfer-K4`：每个 prefix 从完整历史重新推断 4 个无序 modes；
3. `Persistent-K4`：4 个递归 persistent modes。

这三者必须满足：

- 同一训练 task keys、同一 inference inputs、同一 privileged losses；
- encoder 和输出分布相同；
- 每个 query 都执行 4 次共享 decoder；
- 参数量差异不超过 5%；
- 单 prefix 训练 FLOPs 差异不超过 10%。

若不能同时满足，必须额外报告 parameter-matched 与 FLOP-matched 两组，不能只选有利匹配。

### 9.2 诊断基线

4. `K1-capacity-matched`：只有一个 coherent mode，但增宽到参数匹配；
5. `Independent-Ensemble-4`：共享 encoder、独立 updater/decoder，使用 80% deterministic bootstrap support mask；
6. `Categorical-q(rule|D)`：可使用 rule-ID 监督，但必须标记为 privileged structured upper baseline，不参加无特权主胜负。

## 10. 冻结模型后的主动策略

所有策略必须使用同一个 `Persistent-K4` checkpoint 和同一个 calibration：

1. `uniform`；
2. `coverage`；
3. `change-seeking`；
4. `weighted-cell-JS`；
5. `full-state-mode-info`；
6. `exact-oracle-EIG`，仅作上界。

对 stochastic mode prediction，`full-state-mode-info` 每个动作采样 128 个完整下一状态；所有动作使用 common random numbers。不能把逐格 disagreement 相加后称为严格 full-state information。

并列规则：

```text
information score
→ least-used action
→ canonical action ID
```

在 test 前，只用 validation-ID 的最低 `RMST_4` 选出“最强非 oracle 策略”，并永久冻结。

闭环评估：

- 每个策略从相同的 6-transition support 开始，在 8 个候选中无放回执行满 4 步；
- 自主 stop 单独报告，不截断主要曲线；
- stochastic/random 策略每任务使用 16 个固定 policy seeds；
- 方法间共享 task、初态、动作集合和 rollout seed。

## 11. 指标的机器可执行定义

### 11.1 识别时刻

行为类在 prefix `t` 被识别，当且仅当：

1. 聚合到真实行为签名的 mode 权重至少 0.95；
2. 最高权重 coherent mode 在 full-24 diagnostic panel 上全部整帧 exact-match；
3. 以上条件在后续所有 prefix 持续成立。

第一次满足的步数为 `T_id`；预算内从未满足记为右删失。

### 11.2 首要指标

1. 每个主动交互步 `t=0..4` 的 diagnostic mixture NLL；
2. diagnostic NLL 的梯形 AULC：

\[
AULC=\frac{0.5L_0+L_1+L_2+L_3+0.5L_4}{4};
\]

3. `RMST_4`；
4. 全诊断面板 coherent exact transition accuracy；
5. joint-query NLL；
6. behavior-signature `Coverage@4`。

行为类 `m` 被覆盖，当存在同一个 mode 同时满足：

```text
当前主要 query set 的 MAP frame 全部 exact
且 C_km ≤ 0.05 nats/cell
```

主 Coverage 为 `Σ_m μ_m × covered_m`，同时报告 all-classes-covered task rate。

整帧准确率必须来自一个 coherent mode，禁止逐格 marginal argmax 拼成不存在的 “Frankenstein frame”。

### 11.3 诊断指标

- identity-switch rate；
- 兼容行为类 survival rate；
- 被证伪模式权重下降到 `<0.05` 所需步数；
- effective particle count `exp(H(w))`；
- 行为类 Brier、NLL 与 10 个 equal-mass bins 的 ECE；
- change/no-change 分层 NLL；
- oracle EIG regret；
- oracle headroom recovery；
- 参数量、训练 FLOPs、每动作推理 FLOPs、峰值显存。

identity-switch 使用整个 prefix 序列上的一次全局最小-cost lineage assignment；只对相邻 prefix 均仍兼容的目标类计分，被证伪类不进入分母。

## 12. 统计协议

统计单位不能是单个格子、transition 或 rollout。

主要比较使用配对 hierarchical bootstrap，共 20,000 次：

1. 重采样 5 个 training seeds；
2. 每个 seed 内按 12 个 `heldout_axis × true_mode` strata 重采样 task；
3. task 内重采样 rollout seed；
4. 所有方法始终使用相同的抽样索引。

报告：

- 五个逐 training-seed 效应；
- 配对均值差与相对差；
- 95% percentile CI；
- H1/H2/H3 的原始 p-value 与 Holm 校正结果；
- 样本数、删失数和统计单位。

family-wise `α=0.05`。最终 NLL 非劣界冻结为：

```text
delta_NI = 0.02 nats/cell
```

## 13. 预注册 Gate

### Gate 1：多模态表示成立

`Composition-OOD test` 同时满足：

1. `Persistent-K4 Coverage@4 ≥ 0.90`；
2. 相比 validation 预选的最强 single-belief baseline，AULC 至少降低 `0.02 nats/cell`；
3. Holm 校正后的差值 95% CI 不跨 0。

### Gate 2：持久性有独立价值

相比 `Reinfer-K4` 同时满足：

1. `Persistent-K4 identity-switch rate ≤ 0.05`；
2. identity-switch 相对下降至少 50%，校正后 CI 不跨 0；
3. 第 2 次主动交互后的 diagnostic NLL 至少降低 `0.01 nats/cell`，CI 不跨 0。

如果只有 latent 更稳定而预测不改善，Gate 2 不通过。

### Gate 3：主动证伪成立

冻结同一个 `Persistent-K4` 后，相比 validation 预选的最强非 oracle 策略同时满足：

1. `RMST_4` 点估计至少下降 20%；
2. `RMST_baseline - RMST_mode-info` 的校正 95% CI 下界大于 0；
3. 第 4 步 diagnostic NLL 差值 CI 上界不超过 `0.02 nats/cell`；
4. oracle headroom recovery 至少 40%。

test-ID 成功不能替代任何 test-Composition Gate。

## 14. 失败定位，不允许事后改故事

| 结果 | 允许的结论 | 下一步 |
|---|---|---|
| oracle 不优于 random | benchmark 无 headroom | 重做生成器并提升协议版本 |
| oracle 好，Coverage@4 差 | mode 推断或监督失败 | 查 assignment、容量、数据捷径 |
| Coverage 好，权重校准差 | evidence weighting 失败 | 只在新版本修改 proper score/calibration |
| Persistent 与 Reinfer 相同 | persistence 无独立价值 | 接受否定结果 |
| active 优于 random，不优于 change | 任务奖励变化而非证伪 | 增加无信息大变化动作，新版本重测 |
| ID 好、Composition-OOD 差 | 学到模板而非组合推理 | 加强组合 split，不能宣称 rule induction |
| 外部预训练才有效 | 表征先验提供收益 | 另建预训练协议，不能归因于 v1 核心机制 |

## 15. 环境、配置和 artifact 契约

进入 Stage 0-B 前必须增加：

```text
.python-version
pyproject.toml
uv.lock
configs/stage0a/gate0_full.json
configs/stage0b/gate0b_full.json
configs/stage1/*.json
schemas/config.schema.json
schemas/dataset_manifest.schema.json
schemas/run_manifest.schema.json
```

GPU 正式结果还必须发布 Dockerfile、基础镜像 digest、PyTorch/CUDA/cuDNN/驱动版本和 GPU 型号。未来若使用 pretrained checkpoint，必须记录本地文件 SHA256；v1 固定为 `pretrained: null`。

每个 run：

```text
runs/<experiment>/<method>/seed=<n>/<run_id>/
  config.resolved.json
  provenance.json
  environment.json
  dataset-manifest.json
  stdout.log
  stderr.log
  checkpoints/final.safetensors
  checkpoints/training-state.pt
  task_metrics.jsonl
  predictions.jsonl
  metrics.json
  artifact-manifest.json
  artifact-manifest.sha256
  COMPLETED
```

`provenance.json` 必须记录：

- Git commit、dirty 状态和 diff hash；若无 Git，记录确定性 source snapshot hash；
- source、config、lock、dataset 和输入 checkpoint hash；
- 完整 argv 与 cwd；
- 软件/硬件环境；
- 所有 namespace-derived seeds。

只有 schema、hash、任务数和指标检查全部通过后才写 `COMPLETED`。

## 16. 统一结果表

所有汇总必须从逐任务记录自动生成，不允许手工改表。

`run_index.csv`：

```text
experiment_id,run_id,method,policy,config_hash,code_hash,dataset_hash,
environment_hash,train_seed,status,artifact_manifest_hash
```

`task_metrics.jsonl` 的最小字段：

```text
run_id,method,policy,split,rule_family,composition_id,train_seed,
eval_seed,task_id,rollout_id,prefix_t,metric,value,identified,censored,budget
```

`summary_metrics.csv`：

```text
experiment_id,method,policy,split,metric,estimate,ci95_low,ci95_high,
n_train_seeds,n_tasks,n_rollouts,statistical_unit
```

`comparisons.csv`：

```text
hypothesis_id,treatment,control,split,metric,paired_delta,ci95_low,
ci95_high,p_raw,p_holm,noninferiority_margin,passes
```

预测记录必须足以离线重算 Coverage@4、AULC、identity switch、calibration、RMST 和 acquisition regret。

## 17. CLI 契约

以下是必须实现的接口，不是当前已经存在的命令：

```bash
uv sync --frozen --extra dev
uv run prp-wm doctor --strict
uv run prp-wm generate --config configs/stage0b/gate0b_full.json
uv run prp-wm verify-data --manifest data/manifests/prp-stage1-v1.json --strict
uv run prp-wm run --config configs/stage0b/gate0b_full.json
uv run prp-wm sweep --manifest experiments/prp_stage1_v1/tuning.json
uv run prp-wm select-config --experiment prp_stage1_v1
uv run prp-wm train --config configs/stage1/persistent_k4.json --seed 101
uv run prp-wm calibrate --experiment prp_stage1_v1 --method persistent_k4 --seed 1103 --split validation-ID
uv run prp-wm eval-representation --experiment prp_stage1_v1
uv run prp-wm eval-active --experiment prp_stage1_v1
uv run prp-wm aggregate --experiment prp_stage1_v1 --bootstrap 20000 --holm
uv run prp-wm verify --release results/releases/prp_stage1_v1 --strict
```

正式 release 的验收标准：一个不了解项目内部实现的人，从干净的 release tag/snapshot 开始，只执行 README 中的命令，就能：

1. 验证源码、环境、数据和 checkpoint hash；
2. 重跑 smoke、单 seed 或完整 5-seed 实验；
3. 从逐任务记录重建全部表格；
4. 得到相同 Gate 判定；
5. 明确看到该运行属于 smoke、R2 还是 R3，不能把小样本结果误报成正式结论。

## 18. 实施优先级

### 进入 Stage 0-B 前

1. 实现规则 DSL、逐步语义和 golden transition fixtures；
2. 加入环境锁、严格 JSON config、manifest schema 与统一 CLI；
3. 把当前 Gate 0-A hash 变成 CI golden test；
4. 实现固定 validation/test 数据生成与 split 泄漏审计；
5. 实现 exact version space、oracle EIG 和 Gate 0-B。

### 进入神经训练前

1. 实现三种主要表示方法并检查参数/FLOP 公平性；
2. 实现 checkpoint-resume 等价测试和同 seed 双运行测试；
3. 实现 task-level prediction/metric 导出；
4. 实现 calibration、active policy 和 hierarchical bootstrap；
5. 冻结 v1.0 release candidate 后再启动 tuning 与 5-seed final run。

这份协议的重点不是保证 PRP-WM 成功，而是保证它无论成功还是失败，第三方都能得到同样的证据和同样的结论边界。
