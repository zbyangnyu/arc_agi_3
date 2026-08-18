#!/usr/bin/env python3
"""Train the sparse-reward RuleGrid RL-from-scratch baseline."""

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
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--data-master-seed", type=int, default=2026071601)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--eval-tasks", type=int, default=192)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/rl_scratch_seed0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_tasks <= 0:
        raise SystemExit("steps, batch-size, and eval-tasks must be positive")
    if args.budget <= 0 or args.log_every <= 0:
        raise SystemExit("budget and log-every must be positive")
    try:
        import torch
        from prp_wm.pilot import make_pilot_tasks
        from prp_wm.rl import (
            RLPolicyConfig,
            RuleGridActorCritic,
            evaluate_rl_policy,
            remove_calibration_history,
            train_actor_critic_batch,
        )
    except ImportError as error:
        raise SystemExit(f"RL baseline requires the optional PyTorch dependency: {error}") from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    policy_config = RLPolicyConfig()
    policy = RuleGridActorCritic(policy_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    checkpoint_path = output / "checkpoint_last.pt"
    summary_path = output / "summary.json"
    eval_tasks = make_pilot_tasks(
        split="rl-validation",
        master_seed=args.data_master_seed,
        start=0,
        count=args.eval_tasks,
        diagnostic_indices=(0,),
    )
    random_baseline = evaluate_rl_policy(
        None, eval_tasks, budget=args.budget, random_seed=args.seed
    )

    latest: dict[str, float] = {}
    with progress_path.open("w", encoding="utf-8") as progress:
        for step in range(args.steps):
            policy.train()
            tasks = make_pilot_tasks(
                split="rl-train",
                master_seed=args.data_master_seed,
                start=step * args.batch_size,
                count=args.batch_size,
                diagnostic_indices=(0,),
            )
            metrics = train_actor_critic_batch(
                policy,
                tasks,
                optimizer,
                device=device,
                budget=args.budget,
            )
            latest = asdict(metrics)
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0:
                record = {
                    "step": completed,
                    "tasks_seen": completed * args.batch_size,
                    **latest,
                }
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                progress.write(encoded + "\n")
                progress.flush()
                print(encoded, flush=True)

    learned = evaluate_rl_policy(policy, eval_tasks, device=device, budget=args.budget)
    calibration_ablated = evaluate_rl_policy(
        policy,
        eval_tasks,
        device=device,
        budget=args.budget,
        observation_transform=remove_calibration_history,
    )
    checkpoint = {
        "model_type": "RuleGridActorCritic",
        "model_config": asdict(policy_config),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in policy.state_dict().items()
        },
        "training": vars(args) | {"output": str(output)},
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "algorithm": "monte_carlo_actor_critic",
        "reward": "sparse_behavior_identification_success",
        "uses_reconstruction_loss": False,
        "uses_rule_id_as_input": False,
        "uses_version_space_as_input": False,
        "uses_oracle_eig": False,
        "uses_diagnostic_targets": False,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "tasks_seen": args.steps * args.batch_size,
        "budget": args.budget,
        "seed": args.seed,
        "model_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "final_train_batch": latest,
        "validation": asdict(learned),
        "calibration_ablated_validation": asdict(calibration_ablated),
        "random_validation": asdict(random_baseline),
        "checkpoint_path": str(checkpoint_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
