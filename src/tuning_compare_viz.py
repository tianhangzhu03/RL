"""Compare tuned test results between two tuning rounds (e.g., v2 vs v3).

Usage:
  scripts/py.sh -m src.tuning_compare_viz \
    --v2-root runs/tuned_best_v2 \
    --v3-root runs/tuned_best_v3 \
    --out-dir report/compare_v2_v3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.plot_style import finish_figure, set_report_theme, unique_legend

METRIC_DIR = {
    "test_total_reward": 1.0,
    "test_cumulative_return": 1.0,
    "test_cvar": -1.0,
    "test_max_drawdown": 1.0,
    "train_seconds": -1.0,
}

ALGO_ORDER = ["q_learning", "sg_sarsa", "nstep_sarsa"]
ALGO_LABELS = {
    "q_learning": "Q-learning",
    "sg_sarsa": "SG-SARSA",
    "nstep_sarsa": "n-step SARSA",
}


@dataclass(frozen=True)
class ConstraintSpec:
    max_cvar: float
    min_max_drawdown: float
    weight_cvar: float
    weight_mdd: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tuned results from two rounds")
    parser.add_argument("--v2-root", default="runs/tuned_best_v2")
    parser.add_argument("--v3-root", default="runs/tuned_best_v3")
    parser.add_argument("--out-dir", default="report/compare_v2_v3")
    parser.add_argument("--constraint-max-cvar", type=float, default=0.065)
    parser.add_argument("--constraint-min-max-drawdown", type=float, default=-0.60)
    parser.add_argument("--constraint-weight-cvar", type=float, default=1.0)
    parser.add_argument("--constraint-weight-mdd", type=float, default=1.0)
    return parser.parse_args()


def parse_timestamp_from_dir(path: Path) -> datetime:
    try:
        return datetime.strptime(path.name, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.min


def choose_latest_by_algo(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*/suite_summary.csv"))
    if not files:
        raise FileNotFoundError(f"No suite_summary.csv found under {root}")

    rows: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_csv(file)
        if df.empty or "algo" not in df.columns:
            continue
        df["source_file"] = str(file)
        df["source_timestamp"] = parse_timestamp_from_dir(file.parent)
        rows.append(df)

    if not rows:
        raise ValueError(f"Found summary files under {root}, but no valid rows were loaded.")

    merged = pd.concat(rows, ignore_index=True)
    merged = merged.sort_values(["algo", "source_timestamp"])

    selected: List[pd.DataFrame] = []
    for algo, g in merged.groupby("algo", sort=False):
        latest_ts = g["source_timestamp"].max()
        chosen = g[g["source_timestamp"] == latest_ts].copy()
        selected.append(chosen)

    return pd.concat(selected, ignore_index=True)


def add_constraint_columns(df: pd.DataFrame, spec: ConstraintSpec) -> pd.DataFrame:
    out = df.copy()
    out["cvar_violation"] = (out["test_cvar"] - spec.max_cvar).clip(lower=0.0)
    out["mdd_violation"] = (spec.min_max_drawdown - out["test_max_drawdown"]).clip(lower=0.0)
    out["total_violation"] = spec.weight_cvar * out["cvar_violation"] + spec.weight_mdd * out["mdd_violation"]
    out["is_feasible"] = (
        (out["test_cvar"] <= spec.max_cvar) & (out["test_max_drawdown"] >= spec.min_max_drawdown)
    ).astype(int)
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["algo", "version"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            reward_mean=("test_total_reward", "mean"),
            reward_std=("test_total_reward", "std"),
            return_mean=("test_cumulative_return", "mean"),
            return_std=("test_cumulative_return", "std"),
            cvar_mean=("test_cvar", "mean"),
            cvar_std=("test_cvar", "std"),
            mdd_mean=("test_max_drawdown", "mean"),
            mdd_std=("test_max_drawdown", "std"),
            train_seconds_mean=("train_seconds", "mean"),
            train_seconds_std=("train_seconds", "std"),
            feasible_rate=("is_feasible", "mean"),
            violation_mean=("total_violation", "mean"),
        )
        .sort_values(["algo", "version"])
        .reset_index(drop=True)
    )
    return agg


def paired_delta(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for algo, g_algo in df.groupby("algo", sort=False):
        g_v2 = g_algo[g_algo["version"] == "v2"].set_index("seed")
        g_v3 = g_algo[g_algo["version"] == "v3"].set_index("seed")
        common = sorted(set(g_v2.index).intersection(set(g_v3.index)))
        if not common:
            continue

        for seed in common:
            a = g_v2.loc[seed]
            b = g_v3.loc[seed]
            rows.append(
                {
                    "algo": algo,
                    "seed": seed,
                    "delta_reward_v3_minus_v2": float(b["test_total_reward"] - a["test_total_reward"]),
                    "delta_return_v3_minus_v2": float(b["test_cumulative_return"] - a["test_cumulative_return"]),
                    "delta_cvar_v3_minus_v2": float(b["test_cvar"] - a["test_cvar"]),
                    "delta_mdd_v3_minus_v2": float(b["test_max_drawdown"] - a["test_max_drawdown"]),
                    "delta_train_seconds_v3_minus_v2": float(b["train_seconds"] - a["train_seconds"]),
                    "delta_feasible_v3_minus_v2": int(b["is_feasible"] - a["is_feasible"]),
                    "delta_violation_v3_minus_v2": float(b["total_violation"] - a["total_violation"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    mean_row = (
        out.groupby("algo", as_index=False)
        .agg(
            delta_reward_v3_minus_v2=("delta_reward_v3_minus_v2", "mean"),
            delta_return_v3_minus_v2=("delta_return_v3_minus_v2", "mean"),
            delta_cvar_v3_minus_v2=("delta_cvar_v3_minus_v2", "mean"),
            delta_mdd_v3_minus_v2=("delta_mdd_v3_minus_v2", "mean"),
            delta_train_seconds_v3_minus_v2=("delta_train_seconds_v3_minus_v2", "mean"),
            delta_feasible_v3_minus_v2=("delta_feasible_v3_minus_v2", "mean"),
            delta_violation_v3_minus_v2=("delta_violation_v3_minus_v2", "mean"),
        )
        .sort_values("algo")
        .reset_index(drop=True)
    )
    return mean_row


def evaluate_value(row: pd.Series) -> str:
    risk_better = row["delta_cvar_v3_minus_v2"] < 0 and row["delta_mdd_v3_minus_v2"] > 0
    feasible_better = row["delta_feasible_v3_minus_v2"] > 0
    reward_better = row["delta_reward_v3_minus_v2"] > 0

    if risk_better and feasible_better and reward_better:
        return "High value: both risk constraints and reward improved."
    if risk_better and feasible_better and not reward_better:
        return "Valuable for risk-sensitive objective: safer policy, lower reward."
    if reward_better and not feasible_better:
        return "Conditional value: reward rose, but risk-constraint performance worsened."
    if row["delta_violation_v3_minus_v2"] < 0:
        return "Moderate value: constraint violations decreased overall."
    return "Low value: no clear gain under risk-sensitive goal."


def make_metric_barplots(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.copy()
    plot_df["algo_label"] = plot_df["algo"].map(ALGO_LABELS)
    algo_label_order = [ALGO_LABELS[a] for a in ALGO_ORDER if a in set(plot_df["algo"])]

    plot_metrics = [
        ("test_total_reward", "Total Reward"),
        ("test_cumulative_return", "Cumulative Return"),
        ("test_cvar", "CVaR (95%, lower better)"),
        ("test_max_drawdown", "Max Drawdown (higher better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
    axes = axes.flatten()
    palette = {"v2": "#4C72B0", "v3": "#DD8452"}

    for ax, (metric, label) in zip(axes, plot_metrics):
        sns.barplot(
            data=plot_df,
            x="algo_label",
            y=metric,
            hue="version",
            order=algo_label_order,
            hue_order=["v2", "v3"],
            errorbar="sd",
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    handles, labels = unique_legend(*axes[0].get_legend_handles_labels())
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=True, title="Tuning Round")
    fig.suptitle("v2 vs v3 Tuning Comparison", y=0.955, fontsize=14)
    finish_figure(fig, out_path, tight_rect=(0, 0, 1, 0.86))


def make_constraint_scatter(df: pd.DataFrame, out_path: Path, spec: ConstraintSpec) -> None:
    algos = [a for a in ALGO_ORDER if a in set(df["algo"])]
    fig, axes = plt.subplots(1, len(algos), figsize=(6.0 * max(1, len(algos)), 5.3), squeeze=False)
    palette = {"v2": "#4C72B0", "v3": "#DD8452"}
    markers = {"v2": "o", "v3": "s"}

    for i, algo in enumerate(algos):
        ax = axes[0, i]
        g = df[df["algo"] == algo].copy()
        for version in ["v2", "v3"]:
            gv = g[g["version"] == version]
            ax.scatter(
                gv["test_cvar"],
                gv["test_max_drawdown"],
                s=58,
                alpha=0.85,
                c=palette[version],
                marker=markers[version],
                edgecolors="black",
                linewidths=0.4,
                label=version,
            )

        ax.axvline(spec.max_cvar, color="#8a2d2d", linestyle="--", linewidth=1.2, label="CVaR threshold")
        ax.axhline(spec.min_max_drawdown, color="#8a2d2d", linestyle="-.", linewidth=1.2, label="MDD threshold")
        ax.fill_betweenx(
            [g["test_max_drawdown"].min() - 0.01, g["test_max_drawdown"].max() + 0.01],
            x1=g["test_cvar"].min() - 0.005,
            x2=spec.max_cvar,
            color="#d9ead3",
            alpha=0.22,
        )

        ax.set_title(ALGO_LABELS.get(algo, algo))
        ax.set_xlabel("CVaR")
        if i == 0:
            ax.set_ylabel("Max Drawdown")
        ax.grid(True, alpha=0.22)

    handles, labels = unique_legend(*axes[0, 0].get_legend_handles_labels())
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4, frameon=True, fontsize=9)
    fig.suptitle("Risk-Constraint Feasibility Shift (v2 -> v3)", y=0.955, fontsize=14)
    finish_figure(fig, out_path, tight_rect=(0, 0, 1, 0.86))


def make_feasible_rate_plot(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary.copy()
    plot_df["algo_label"] = plot_df["algo"].map(ALGO_LABELS)
    algo_label_order = [ALGO_LABELS[a] for a in ALGO_ORDER if a in set(plot_df["algo"])]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    palette = {"v2": "#4C72B0", "v3": "#DD8452"}

    sns.barplot(
        data=plot_df,
        x="algo_label",
        y="feasible_rate",
        hue="version",
        order=algo_label_order,
        hue_order=["v2", "v3"],
        palette=palette,
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Feasible Rate")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Constraint Feasible Rate by Algorithm")
    finish_figure(fig, out_path)


def save_markdown(df: pd.DataFrame, path: Path) -> None:
    try:
        text = df.to_markdown(index=False)
    except Exception:
        text = df.to_csv(index=False)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_report_theme()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    spec = ConstraintSpec(
        max_cvar=float(args.constraint_max_cvar),
        min_max_drawdown=float(args.constraint_min_max_drawdown),
        weight_cvar=float(args.constraint_weight_cvar),
        weight_mdd=float(args.constraint_weight_mdd),
    )

    v2 = choose_latest_by_algo(Path(args.v2_root))
    v3 = choose_latest_by_algo(Path(args.v3_root))
    v2["version"] = "v2"
    v3["version"] = "v3"

    data = pd.concat([v2, v3], ignore_index=True)
    data = add_constraint_columns(data, spec)
    data["algo"] = pd.Categorical(data["algo"], categories=ALGO_ORDER, ordered=True)
    data = data.sort_values(["algo", "version", "seed"]).reset_index(drop=True)

    summary = summarize(data)
    delta = paired_delta(data)
    if not delta.empty:
        delta["value_assessment"] = delta.apply(evaluate_value, axis=1)

    summary.to_csv(table_dir / "v2_v3_summary.csv", index=False)
    delta.to_csv(table_dir / "v2_v3_paired_delta.csv", index=False)
    save_markdown(summary, table_dir / "v2_v3_summary.md")
    save_markdown(delta, table_dir / "v2_v3_paired_delta.md")

    make_metric_barplots(data, fig_dir / "fig_v2_v3_metric_bars.png")
    make_constraint_scatter(data, fig_dir / "fig_v2_v3_constraint_scatter.png", spec)
    make_feasible_rate_plot(summary, fig_dir / "fig_v2_v3_feasible_rate.png")

    lines: List[str] = []
    lines.append("# v2 vs v3 Value Assessment")
    lines.append("")
    lines.append(f"- Constraint: CVaR <= {spec.max_cvar:.4f}, Max Drawdown >= {spec.min_max_drawdown:.4f}")
    lines.append("")
    if delta.empty:
        lines.append("No paired seed overlap found, cannot compute value assessment.")
    else:
        for _, row in delta.iterrows():
            lines.append(f"## {ALGO_LABELS.get(str(row['algo']), str(row['algo']))}")
            lines.append(f"- Delta reward (v3-v2): {row['delta_reward_v3_minus_v2']:.4f}")
            lines.append(f"- Delta return (v3-v2): {row['delta_return_v3_minus_v2']:.4f}")
            lines.append(f"- Delta CVaR (v3-v2): {row['delta_cvar_v3_minus_v2']:.5f}")
            lines.append(f"- Delta max drawdown (v3-v2): {row['delta_mdd_v3_minus_v2']:.4f}")
            lines.append(f"- Delta feasible rate (v3-v2): {row['delta_feasible_v3_minus_v2']:.3f}")
            lines.append(f"- Delta mean violation (v3-v2): {row['delta_violation_v3_minus_v2']:.5f}")
            lines.append(f"- Assessment: {row['value_assessment']}")
            lines.append("")

    (out_dir / "assessment.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Saved comparison artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
