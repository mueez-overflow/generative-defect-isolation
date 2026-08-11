#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = ("efficientnetv2_l", "vit_small", "vit_large")
DEFAULT_SEEDS = (24, 42, 67, 76)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 20% baseline/GDI multi-seed robustness experiment."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default="runs/multiseed_20pct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--conditions", nargs="+", choices=("baseline", "gdi"), default=["baseline", "gdi"]
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    output_root = Path(args.output_dir)

    for model in args.models:
        for condition in args.conditions:
            inpainted_pct = 0 if condition == "baseline" else 100
            for seed in args.seeds:
                run_dir = output_root / model / condition / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                train_cmd = [
                    sys.executable,
                    str(here / "train.py"),
                    "--train-csv", args.train_csv,
                    "--train-dir", args.train_dir,
                    "--test-csv", args.test_csv,
                    "--test-dir", args.test_dir,
                    "--output-dir", str(run_dir),
                    "--model", model,
                    "--seed", str(seed),
                    "--original-pct", "20",
                    "--inpainted-pct", str(inpainted_pct),
                    "--device", args.device,
                    "--num-workers", str(args.num_workers),
                ]
                run(train_cmd)

                eval_cmd = [
                    sys.executable,
                    str(here / "evaluate.py"),
                    "--checkpoint", str(run_dir / "best_f1_model.pth"),
                    "--test-csv", args.test_csv,
                    "--test-dir", args.test_dir,
                    "--output", str(run_dir / "fixed_threshold_results.json"),
                    "--model", model,
                    "--condition", condition,
                    "--seed", str(seed),
                    "--device", args.device,
                    "--num-workers", str(args.num_workers),
                ]
                run(eval_cmd)

    run(
        [
            sys.executable,
            str(here / "summarize.py"),
            "--runs-dir", str(output_root),
            "--output-dir", str(output_root / "summary"),
        ]
    )


if __name__ == "__main__":
    main()
