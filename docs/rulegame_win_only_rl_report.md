# RuleGame terminal-win-only PPO / GRPO smoke report

> 日期：2026-07-16  
> 状态：单 seed 机制 smoke test，非正式算法排名

## 环境

`RuleGame` 是一个三步、四种隐藏 collision rule 的最小游戏：

1. policy 选择 `PROBE` 或跳过；
2. 若实验，观察 mode-specific 图像结果，然后继续；
3. 在逐像素相同的终局画面中选择四扇门之一。

只有与隐藏规则对应的门获胜。前两步 reward 恒为 0；终局获胜 reward 为 1，
失败为 0。Policy 不读取 rule label、identification reward、diagnostic target 或
reconstruction loss。

四种 mode 的初始画面和终局画面在相同 nuisance group 内完全相同；只有历史中的
实验结果不同。因此终局动作必须依赖 recurrent memory。

## 结果

| 算法 | 训练轨迹 | validation win | 清空终局记忆 | probe rate |
|---|---:|---:|---:|---:|
| PPO | 19,200 | 98.05% | 25.00% | 100% |
| GRPO，等轨迹预算 | 19,200 | 25.00% | 25.00% | 100% |
| GRPO，完整 smoke | 76,800 | 91.41% | 25.00% | 100% |

Artifacts：

- [PPO summary](../runs/rulegame_ppo_smoke_seed20260716/summary.json)
- [GRPO equal-trajectory summary](../runs/rulegame_grpo_equal_trajectories_seed20260716/summary.json)
- [GRPO full-smoke summary](../runs/rulegame_grpo_smoke_seed20260716/summary.json)

## 允许的解释

1. 只使用终局胜利奖励，recurrent PPO 和 GRPO 都能够学会先实验；
2. 足够训练后，两者能把视觉实验结果保存在 memory 中，并据此选择终局动作；
3. 清空终局记忆后均回到四选一的 25%，为历史依赖提供了因果消融证据；
4. 在该单 seed smoke 配置下，PPO 的 trajectory efficiency 明显优于 GRPO；GRPO
   在 19,200 轨迹时只学会 `PROBE`，尚未学会四规则到门的映射。

这不能证明 PPO 普遍优于 GRPO，也不能证明模型形成了显式对象或世界模型。当前游戏
只有四种原子规则、固定三步 horizon 和固定门语义。正式结论需要多 seed、相同调参预算、
更丰富的规则组合以及 history-swap / held-out-composition 测试。
