#!/usr/bin/env python3
"""Train PPO or GRPO on RuleGame using terminal win reward only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("ppo", "grpo"), required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--data-master-seed", type=int, default=2026071601)
    parser.add_argument("--eval-tasks", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_tasks <= 0:
        raise SystemExit("steps, batch-size and eval-tasks must be positive")
    if args.group_size < 2 or args.log_every <= 0:
        raise SystemExit("group-size must be >=2 and log-every positive")
    try:
        import torch
        from prp_wm.rulegame import make_rulegame_specs
        from prp_wm.rulegame_rl import (
            RuleGamePolicy,
            RuleGamePolicyConfig,
            evaluate_rulegame_policy,
            train_grpo_update,
            train_ppo_update,
        )
    except ImportError as error:
        raise SystemExit(f"RuleGame RL requires PyTorch: {error}") from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    config = RuleGamePolicyConfig()
    policy = RuleGamePolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    output = (
        args.output
        if args.output is not None
        else Path(f"runs/rulegame_{args.algorithm}_seed{args.seed}")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    summary_path = output / "summary.json"
    checkpoint_path = output / "checkpoint_last.pt"
    eval_specs = make_rulegame_specs(
        split="rulegame-validation",
        master_seed=args.data_master_seed,
        start=0,
        count=args.eval_tasks,
    )

    latest: dict[str, float] = {}
    with progress_path.open("w", encoding="utf-8") as progress:
        for step in range(args.steps):
            specs = make_rulegame_specs(
                split="rulegame-train",
                master_seed=args.data_master_seed,
                start=step * args.batch_size,
                count=args.batch_size,
            )
            if args.algorithm == "ppo":
                metrics = train_ppo_update(policy, optimizer, specs, device=device)
                sampled_trajectories = args.batch_size
            else:
                metrics = train_grpo_update(
                    policy,
                    optimizer,
                    specs,
                    group_size=args.group_size,
                    device=device,
                )
                sampled_trajectories = args.batch_size * args.group_size
            latest = asdict(metrics)
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0:
                record = {
                    "step": completed,
                    "base_tasks_seen": completed * args.batch_size,
                    "trajectories_seen": completed * sampled_trajectories,
                    **latest,
                }
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress.write(encoded + "\n")
                progress.flush()
                print(encoded, flush=True)

    normal = evaluate_rulegame_policy(policy, eval_specs, device=device)
    memory_reset = evaluate_rulegame_policy(
        policy, eval_specs, device=device, reset_memory_at_decision=True
    )
    torch.save(
        {
            "algorithm": args.algorithm,
            "model_config": asdict(config),
            "model_state_dict": {
                name: value.detach().cpu() for name, value in policy.state_dict().items()
            },
            "training": vars(args) | {"output": str(output)},
        },
        checkpoint_path,
    )
    summary = {
        "algorithm": args.algorithm,
        "reward": "terminal_win_only",
        "intermediate_reward": False,
        "uses_rule_label": False,
        "uses_identification_reward": False,
        "uses_reconstruction_loss": False,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size if args.algorithm == "grpo" else None,
        "base_tasks_seen": args.steps * args.batch_size,
        "trajectories_seen": args.steps * args.batch_size * (
            args.group_size if args.algorithm == "grpo" else 1
        ),
        "model_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "final_train_batch": latest,
        "validation": asdict(normal),
        "memory_reset_validation": asdict(memory_reset),
        "checkpoint_path": str(checkpoint_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
