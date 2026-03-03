"""Run the full experiment suite across algorithms, seeds, and risk settings.

Usage:
    python -m src.run_suite --config configs/base.yaml --output-root runs
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from src.pipeline import prepare_dataset_split
from src.train import load_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full experiment suite")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--include-no-cvar", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    split = prepare_dataset_split(config)

    algorithms = ["q_learning", "sg_sarsa", "nstep_sarsa"]
    seeds: List[int] = [int(s) for s in config["training"]["seeds"]]
    tags = ["with_cvar", "no_cvar"] if args.include_no_cvar else ["with_cvar"]

    output_root = Path(args.output_root)
    summary_rows = []

    for algo in algorithms:
        for tag in tags:
            run_cfg: Dict[str, Any] = copy.deepcopy(config)
            if tag == "no_cvar":
                run_cfg["env"]["lambda_cvar"] = 0.0

            for seed in seeds:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = output_root / algo / tag / f"seed_{seed}_{timestamp}"
                run_dir.mkdir(parents=True, exist_ok=True)

                with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(run_cfg, f)

                metrics = run_training(
                    algo=algo,
                    config=run_cfg,
                    train_df=split.train,
                    val_df=split.val,
                    test_df=split.test,
                    seed=seed,
                    output_dir=run_dir,
                )

                summary_rows.append(
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
                    }
                )

                print(f"Completed {algo} / {tag} / seed={seed}: {run_dir}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "suite_summary.csv", index=False)
    print(f"Saved suite summary to: {output_root / 'suite_summary.csv'}")


if __name__ == "__main__":
    main()
