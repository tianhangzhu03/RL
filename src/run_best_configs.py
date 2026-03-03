"""Run unified test report using best configs selected by validation-only tuning.

Usage:
  scripts/py.sh -m src.run_best_configs --tuning-root runs/tuning_budget --output-root runs/tuned_best
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from src.pipeline import prepare_dataset_split
from src.train import load_config, run_training


ALGOS = ["q_learning", "sg_sarsa", "nstep_sarsa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified test suite using tuned best configs")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tuning-root", default="runs/tuning_budget")
    parser.add_argument("--output-root", default="runs/tuned_best")
    parser.add_argument("--algo", default="all", choices=["all", "q_learning", "sg_sarsa", "nstep_sarsa"])
    return parser.parse_args()


def find_best_config(tuning_root: Path, algo: str) -> Path:
    latest_ptr = tuning_root / algo / "LATEST.txt"
    if latest_ptr.exists():
        p = Path(latest_ptr.read_text(encoding="utf-8").strip()) / "best_config.yaml"
        if p.exists():
            return p

    candidates = sorted((tuning_root / algo).glob("*/best_config.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No best_config.yaml found for {algo} under {tuning_root/algo}")
    return candidates[-1]


def main() -> None:
    args = parse_args()

    base_cfg = load_config(args.config)
    seeds: List[int] = [int(s) for s in base_cfg["training"]["seeds"]]

    algo_list = ALGOS if args.algo == "all" else [args.algo]

    split = prepare_dataset_split(base_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    tuning_root = Path(args.tuning_root)

    for algo in algo_list:
        best_cfg_path = find_best_config(tuning_root, algo)
        with open(best_cfg_path, "r", encoding="utf-8") as f:
            best_cfg = yaml.safe_load(f)
        tag = "with_cvar" if float(best_cfg["env"].get("lambda_cvar", 0.0)) > 0.0 else "no_cvar"

        for seed in seeds:
            run_dir = out_root / algo / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            metrics = run_training(
                algo=algo,
                config=best_cfg,
                train_df=split.train,
                val_df=split.val,
                test_df=split.test,
                seed=seed,
                output_dir=run_dir,
                evaluate_test=True,
            )

            rows.append(
                {
                    "algo": algo,
                    "tag": tag,
                    "seed": seed,
                    "train_seconds": metrics["training"]["total_train_seconds"],
                    "time_per_100k_steps": metrics["training"]["time_per_100k_steps"],
                    "test_total_reward": metrics["test"]["total_reward"],
                    "test_cumulative_return": metrics["test"]["cumulative_return"],
                    "test_max_drawdown": metrics["test"]["max_drawdown"],
                    "test_cvar": metrics["test"]["cvar"],
                    "best_config_path": str(best_cfg_path),
                }
            )
            print(f"Completed tuned test: {algo} seed={seed}")

    summary = pd.DataFrame(rows).sort_values(["algo", "seed"]).reset_index(drop=True)
    summary.to_csv(out_root / "suite_summary.csv", index=False)

    agg = (
        summary.groupby("algo")
        .agg(
            n_runs=("seed", "count"),
            train_seconds_mean=("train_seconds", "mean"),
            test_total_reward_mean=("test_total_reward", "mean"),
            test_cumulative_return_mean=("test_cumulative_return", "mean"),
            test_max_drawdown_mean=("test_max_drawdown", "mean"),
            test_cvar_mean=("test_cvar", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(out_root / "suite_summary_agg.csv", index=False)

    (Path(args.output_root) / "LATEST.txt").write_text(str(out_root), encoding="utf-8")

    print(f"Saved tuned unified report to: {out_root}")


if __name__ == "__main__":
    main()
