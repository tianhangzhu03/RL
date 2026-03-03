"""Evaluation and reporting entrypoint.

Usage:
    python -m src.eval --run_dir runs_xlf_baseline/q_learning/with_cvar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_run_dirs(run_dir: Path) -> List[Path]:
    if (run_dir / "test_metrics.json").exists():
        return [run_dir]

    run_dirs: List[Path] = []
    for child in sorted(run_dir.glob("**/test_metrics.json")):
        run_dirs.append(child.parent)

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {run_dir}")
    return run_dirs


def parse_tag(run_path: Path) -> str:
    for part in run_path.parts:
        if part in {"with_cvar", "no_cvar"}:
            return part
    return "unknown"


def collect_tables(run_dirs: List[Path]) -> Dict[str, pd.DataFrame]:
    perf_rows = []
    risk_rows = []
    timing_rows = []
    history_frames = []

    for run in run_dirs:
        test_metrics = load_json(run / "test_metrics.json")
        train_metrics = load_json(run / "training_metrics.json")
        baseline_metrics = load_json(run / "baseline_metrics.json")

        run_name = run.name
        tag = parse_tag(run)

        perf_rows.append(
            {
                "run": run_name,
                "tag": tag,
                "cumulative_return": test_metrics["cumulative_return"],
                "mean_return": test_metrics["mean_return"],
                "volatility": test_metrics["volatility"],
                "sharpe": test_metrics["sharpe"],
                "max_drawdown": test_metrics["max_drawdown"],
                "total_reward": test_metrics["total_reward"],
            }
        )

        risk_rows.append(
            {
                "run": run_name,
                "tag": tag,
                "var_95": test_metrics["var"],
                "cvar_95": test_metrics["cvar"],
                "buy_hold_cvar_95": baseline_metrics["buy_hold"]["cvar"],
                "random_cvar_95": baseline_metrics["random"]["cvar"],
            }
        )

        timing_rows.append(
            {
                "run": run_name,
                "tag": tag,
                "total_train_seconds": train_metrics["total_train_seconds"],
                "total_steps": train_metrics["total_steps"],
                "time_per_100k_steps": train_metrics["time_per_100k_steps"],
                "convergence_episode": train_metrics.get("convergence_episode"),
            }
        )

        history = pd.read_csv(run / "train_history.csv")
        history["run"] = run_name
        history["tag"] = tag
        history_frames.append(history)

    performance_table = pd.DataFrame(perf_rows)
    risk_table = pd.DataFrame(risk_rows)
    timing_table = pd.DataFrame(timing_rows)
    history_table = pd.concat(history_frames, ignore_index=True)

    return {
        "performance": performance_table,
        "risk": risk_table,
        "timing": timing_table,
        "history": history_table,
    }


def expected_results_table(performance: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if {"with_cvar", "no_cvar"}.issubset(set(performance["tag"].unique())):
        perf_group = performance.groupby("tag").mean(numeric_only=True)
        risk_group = risk.groupby("tag").mean(numeric_only=True)

        rows.append(
            {
                "check": "CVaR should improve with cvar penalty",
                "expected_direction": "with_cvar < no_cvar",
                "observed_delta": risk_group.loc["with_cvar", "cvar_95"] - risk_group.loc["no_cvar", "cvar_95"],
                "matches_expectation": bool(risk_group.loc["with_cvar", "cvar_95"] < risk_group.loc["no_cvar", "cvar_95"]),
            }
        )
        rows.append(
            {
                "check": "Max drawdown should improve with cvar penalty",
                "expected_direction": "with_cvar > no_cvar (less negative)",
                "observed_delta": perf_group.loc["with_cvar", "max_drawdown"] - perf_group.loc["no_cvar", "max_drawdown"],
                "matches_expectation": bool(
                    perf_group.loc["with_cvar", "max_drawdown"] > perf_group.loc["no_cvar", "max_drawdown"]
                ),
            }
        )
        rows.append(
            {
                "check": "Return may decrease with cvar penalty",
                "expected_direction": "with_cvar <= no_cvar",
                "observed_delta": perf_group.loc["with_cvar", "cumulative_return"]
                - perf_group.loc["no_cvar", "cumulative_return"],
                "matches_expectation": bool(
                    perf_group.loc["with_cvar", "cumulative_return"] <= perf_group.loc["no_cvar", "cumulative_return"]
                ),
            }
        )
    else:
        rows.append(
            {
                "check": "Risk-ablation direction",
                "expected_direction": "requires both with_cvar and no_cvar runs",
                "observed_delta": np.nan,
                "matches_expectation": False,
            }
        )

    return pd.DataFrame(rows)


def plot_learning_curve(history: pd.DataFrame, output_path: Path) -> None:
    summary = (
        history.groupby(["episode", "tag"])["episode_reward"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "reward_mean", "std": "reward_std"})
    )

    plt.figure(figsize=(10, 6))
    for tag in sorted(summary["tag"].unique()):
        subset = summary[summary["tag"] == tag]
        x = subset["episode"].to_numpy()
        y = subset["reward_mean"].to_numpy()
        std = subset["reward_std"].fillna(0.0).to_numpy()

        plt.plot(x, y, label=f"{tag} mean reward")
        plt.fill_between(x, y - std, y + std, alpha=0.2)

    plt.title("Training Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate run artifacts")
    parser.add_argument("--run_dir", required=True, help="Single run folder or parent folder")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dirs = discover_run_dirs(run_dir)

    tables = collect_tables(run_dirs)
    exp_table = expected_results_table(tables["performance"], tables["risk"])

    output_dir = run_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    tables["performance"].to_csv(output_dir / "performance_table.csv", index=False)
    tables["risk"].to_csv(output_dir / "risk_table.csv", index=False)
    tables["timing"].to_csv(output_dir / "training_time_table.csv", index=False)
    tables["history"].to_csv(output_dir / "history_table.csv", index=False)
    exp_table.to_csv(output_dir / "expected_results_table.csv", index=False)

    plot_learning_curve(tables["history"], output_dir / "learning_curve.png")

    print(f"Saved evaluation artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
