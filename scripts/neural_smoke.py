#!/usr/bin/env python3
"""Run a tiny CPU/GPU train path for the Stage-1 neural PRP scaffold.

This uses a deliberately trivial synthetic hidden-rule batch.  It verifies the
same pieces an eventual RuleGrid training run needs (forward, prequential
replay, assignment loss, backward, optimizer step, and device transfer), but
it is *not* a benchmark result and must not be used in a paper table.

Example:

    python scripts/neural_smoke.py --device cuda --steps 3 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise SystemExit("--steps and --batch-size must be positive")
    try:
        import torch
        from prp_wm.neural import NeuralPRPConfig, PersistentK4, make_toy_rulegrid_batch
    except ImportError as error:
        raise SystemExit(f"neural smoke requires PyTorch: {error}") from error

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    # Keep defaults aligned with the architecture spec rather than shrinking
    # the model for smoke coverage.  This is not a full protocol validation.
    config = NeuralPRPConfig()
    model = PersistentK4(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    random = torch.Generator(device=device).manual_seed(args.seed ^ 0x5A17)
    started = time.perf_counter()
    metrics: dict[str, float] = {}
    for step in range(args.steps):
        batch = make_toy_rulegrid_batch(
            batch_size=args.batch_size,
            config=config,
            device=device,
            generator=random,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.losses(batch)
        if not torch.isfinite(loss.total):
            raise RuntimeError(f"non-finite loss at smoke step {step}: {loss.total}")
        loss.total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        optimizer.step()
        metrics = loss.detached_metrics() | {"gradient_norm": grad_norm}

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    payload: dict[str, object] = {
        "batch_size": args.batch_size,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seed": args.seed,
        "smoke": "synthetic_rulegrid_shaped_only_not_benchmark",
        "steps": args.steps,
        "torch_version": torch.__version__,
        **metrics,
    }
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
