"""Train a minimal affine calibration head on synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.training.fine_tune import FineTuneConfig, fine_tune_affine_head, save_fine_tune_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a simple affine calibration head")
    parser.add_argument("--output", type=Path, default=Path("fine_tune_result.json"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    relative = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    target = 3.5 * relative + 12.0

    result = fine_tune_affine_head(
        relative,
        target,
        config=FineTuneConfig(epochs=args.epochs, lr=args.lr, init_scale=0.0, init_offset=0.0, seed=args.seed),
    )
    save_fine_tune_result(result, args.output)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
