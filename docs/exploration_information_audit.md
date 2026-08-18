# 主动探索的信息获取审计

## 结论

当前 selector 的方向是对的：它优先获取**能改善当前 query 决策**的信息，而不是
无条件最大化关于整个环境的信息。当前主要瓶颈不是 utility，而是 learned world
model 对“一个 probe 会怎样划分规则假设”的预测不够准。

selector 对候选 \(a\) 的主分数为一步后的期望最优决策增益：

\[
U_q(a)=
\sum_y \max_d P(q=d,y\mid a)-\max_d P(q=d).
\]

只有当 \(U_q\) 并列时，才用 outcome entropy \(H(Y_a)\) 作次级键。因此它追求
query-relevant value of information，而不是把 nuisance information 也当成同等目标。

## Two-axis exact control

实验把两个相关 rule axes 各限制为两个可能值，组合成四个 equiprobable doors；第三
轴仍有四个 nuisance values。候选集为每轴两个 atomic probes 加两个 neutral probes。
正式运行覆盖 144 个 scenario、每个 policy 2,304 条配对 hidden-program paths。

| policy | B0 | B1 | B2 | B3 | B2 前两步覆盖互补轴 |
|---|---:|---:|---:|---:|---:|
| current query-success greedy | 0.25 | 0.50 | **1.00** | 1.00 | **100%** |
| query mutual information | 0.25 | 0.50 | **1.00** | 1.00 | **100%** |
| depth-2 exact DP | 0.25 | 0.50 | **1.00** | 1.00 | **100%** |
| global information gain | 0.25 | 0.25 | **0.50** | 1.00 | 0% |
| uniform | 0.25 | 0.382 | **0.538** | 0.703 | 16.7% |

global-EIG 第一步选择四值 nuisance，获得 2 bits 全局信息，但 query success 仍为
0.25。到 B2，它累计获得 3 bits，却只有 0.50 query success；当前 query selector
只获取 2 bits，但已经达到 1.00。这直接说明“信息越多”不等于“对任务越有用”。

## Learned 2×2 bridge

随后冻结当前 executor，独立替换 action-selection partition 与 posterior-update
likelihood。四个条件共享初始 16-code exact posterior、候选顺序、反馈和预算：

| B2 condition | terminal accuracy | 互补轴覆盖 | query entropy |
|---|---:|---:|---:|
| exact select / exact update | **1.000** | **1.000** | 0 |
| exact select / learned update | 0.939 | 0.938 | 0.093 bit |
| learned select / exact update | 0.837 | 0.694 | 0.347 bit |
| learned select / learned update | **0.826** | 0.694 | 0.116 bit |

learned outcome partition 的 pairwise F1 为 0.848，只有 50% candidate panels 与
exact partition 完全一致。learned selector 第一步有 16.7% 选择 nuisance；第二步
有 9.7% 选择 neutral。最弱的 query pair 是 collision–relation，learned/learned
B2 仅 0.660；collision–trigger 为 0.969，trigger–relation 为 0.850。

因此当前误差以 **selection model error** 为主：只替换 learned selection 带来约
16.3 个百分点损失，只替换 learned likelihood update 带来约 6.1 个百分点损失。
放宽到 B3 后 learned/learned 回升到 0.918，说明额外探索能部分补救，但没有消除
world-model bias。

## Locality continuation 的初步结果

在相同随机 singleton geometry、64 codes、训练顺序和初始 checkpoint 下比较
GeomSup 与 LocReg。100-step LocReg 把 held-out fiber Huber 从 5.84 降到 2.87，
proper NLL 从 0.323 降到 0.237，MAP exact 从 7.7% 提到 17.8%。这说明 locality
loss 本身有信号。

但两组都破坏了旧 active-prefix gate；LocReg 的 held-out triple exact 只有 63.5%，
GeomSup 为 85.4%。所以这些 checkpoint 都不能作为改进模型使用。问题是新 geometry
gradient 远大于已拟合 replay gradient，普通 replay 没有形成足够强的 trust region。

## 下一步

1. selection 不再只使用 MAP outcome partition；比较完整 predictive query-MI、
   ensemble disagreement 和 conservative lower-bound value of information。
2. query 已识别时允许 early stop，避免固定预算下继续观察 nuisance 并被错误
   likelihood 反向污染。
3. LocReg 加 frozen-teacher distillation、parameter anchoring 或显式 factor heads，
   先满足旧 gate，再评价 locality 与 two-axis learned B2。
4. 设计真正的 2+2 partial probes。当前 atomic probe 会识别完整四值 axis，只是
   初始 posterior 的二值限制让它对 query 等价于一 bit。
5. 最终再去掉预制 probe bank、显式 query axes、known codebook 与 oracle palette，
   加入动作生成、reachability、代价和多步 lookahead。

Artifacts:

- [exact result](../runs/two_axis_compositional_acquisition_g16_s3_20260911/result.json)
- [learned bridge](../runs/two_axis_compositional_learned_bridge_g16_s3_20260911/result.json)
- [GeomSup control](../runs/counterfactual_locality_geomsup100_seed2026072402/result.json)
- [LocReg control](../runs/counterfactual_locality_locreg100_seed2026072402/result.json)
