"""Generate research-style figures and tables for algorithm comparison.

Usage:
    scripts/py.sh -m src.research_viz \
        --summary runs/suite_summary.csv \
        --runs-root runs_v4_full \
        --fig-dir report/figures \
        --table-dir report/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from src.plot_style import finish_figure, set_report_theme

ALGO_ORDER = ["q_learning", "sg_sarsa", "nstep_sarsa"]
TAG_ORDER = ["no_cvar", "with_cvar"]
METRIC_COLS = [
    "test_cumulative_return",
    "test_cvar",
    "test_max_drawdown",
    "test_total_reward",
    "train_seconds",
]

METRIC_LABELS = {
    "test_cumulative_return": "Cumulative Return",
    "test_cvar": "CVaR (95%)",
    "test_max_drawdown": "Max Drawdown",
    "test_total_reward": "Total Reward",
    "train_seconds": "Train Time (s)",
}

ALGO_LABELS = {
    "q_learning": "Q-learning",
    "sg_sarsa": "SG-SARSA",
    "nstep_sarsa": "n-step SARSA",
}

ALGO_SHORT = {
    "q_learning": "QL",
    "sg_sarsa": "SG",
    "nstep_sarsa": "NS",
}

TAG_LABELS = {
    "no_cvar": "No CVaR",
    "with_cvar": "With CVaR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate research visualizations and tables")
    parser.add_argument("--summary", default="runs/suite_summary.csv")
    parser.add_argument("--runs-root", default="runs_v4_full")
    parser.add_argument("--fig-dir", default="report/figures")
    parser.add_argument("--table-dir", default="report/tables")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["algo"] = pd.Categorical(out["algo"], categories=ALGO_ORDER, ordered=True)
    out["tag"] = pd.Categorical(out["tag"], categories=TAG_ORDER, ordered=True)
    return out.sort_values(["algo", "tag", "seed"]).reset_index(drop=True)


def bootstrap_ci_mean(
    values: Iterable[float],
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])

    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = np.mean(sample)

    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def table_mean_std_ci(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (algo, tag), g in df.groupby(["algo", "tag"], sort=False):
        row: Dict[str, object] = {
            "algo": str(algo),
            "tag": str(tag),
            "n_runs": int(len(g)),
        }
        for metric in METRIC_COLS:
            vals = g[metric].to_numpy(dtype=float)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_low, ci_high = bootstrap_ci_mean(vals, n_bootstrap=n_bootstrap, rng=rng)
            row[f"{metric}_mean"] = mean_v
            row[f"{metric}_std"] = std_v
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["algo", "tag"]).reset_index(drop=True)


def table_paired_deltas(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for algo, g_algo in df.groupby("algo", sort=False):
        g_no = g_algo[g_algo["tag"] == "no_cvar"].set_index("seed")
        g_with = g_algo[g_algo["tag"] == "with_cvar"].set_index("seed")
        common_seeds = sorted(set(g_no.index).intersection(set(g_with.index)))
        if not common_seeds:
            continue

        for metric in METRIC_COLS:
            diffs = (
                g_with.loc[common_seeds, metric].to_numpy(dtype=float)
                - g_no.loc[common_seeds, metric].to_numpy(dtype=float)
            )
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            ci_low, ci_high = bootstrap_ci_mean(diffs, n_bootstrap=n_bootstrap, rng=rng)

            # Small-sample robust test; fallback if all diffs are identical.
            try:
                _, p_value = wilcoxon(diffs)
                p_value = float(p_value)
            except ValueError:
                p_value = 1.0

            rows.append(
                {
                    "algo": str(algo),
                    "metric": metric,
                    "n_pairs": len(common_seeds),
                    "delta_mean_with_minus_no": mean_diff,
                    "delta_std": std_diff,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "wilcoxon_p": p_value,
                }
            )

    return pd.DataFrame(rows).sort_values(["algo", "metric"]).reset_index(drop=True)


def make_metric_boxplots(df: pd.DataFrame, fig_path: Path) -> None:
    metrics = [
        "test_cumulative_return",
        "test_cvar",
        "test_max_drawdown",
        "train_seconds",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    axes = axes.flatten()
    palette = {"no_cvar": "#4C72B0", "with_cvar": "#DD8452"}

    for ax, metric in zip(axes, metrics):
        sns.boxplot(
            data=df,
            x="algo",
            y=metric,
            hue="tag",
            order=ALGO_ORDER,
            hue_order=TAG_ORDER,
            ax=ax,
            palette=palette,
            width=0.62,
            showfliers=False,
            linewidth=1.1,
        )
        # Add clean mean markers per box to make key differences visible at a glance.
        mean_df = (
            df.groupby(["algo", "tag"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "metric_mean"})
        )
        offset = {"no_cvar": -0.17, "with_cvar": 0.17}
        for _, row in mean_df.iterrows():
            x = ALGO_ORDER.index(str(row["algo"])) + offset[str(row["tag"])]
            y = float(row["metric_mean"])
            ax.scatter(x, y, marker="D", s=24, c="black", alpha=0.9, zorder=4)
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(range(len(ALGO_ORDER)))
        ax.set_xticklabels([ALGO_SHORT[a] for a in ALGO_ORDER], rotation=0)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(True, axis="y", alpha=0.22)

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    legend_handles = [Patch(facecolor=palette[t], edgecolor="black", label=TAG_LABELS[t]) for t in TAG_ORDER]
    fig.legend(
        handles=legend_handles,
        labels=[TAG_LABELS[t] for t in TAG_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=2,
        frameon=True,
        fontsize=9,
        title="Setting",
        title_fontsize=9,
    )

    fig.suptitle("Algorithm Comparison Across Risk Settings", y=0.955, fontsize=13.5)
    finish_figure(fig, fig_path, tight_rect=(0, 0, 1, 0.84))


def make_risk_return_scatter(df: pd.DataFrame, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    algo_palette = {"q_learning": "#4C72B0", "sg_sarsa": "#55A868", "nstep_sarsa": "#C44E52"}
    marker_map = {"no_cvar": "o", "with_cvar": "s"}

    # 1) Light seed-level scatter as context.
    sns.scatterplot(
        data=df,
        x="test_cvar",
        y="test_cumulative_return",
        hue="algo",
        style="tag",
        hue_order=ALGO_ORDER,
        style_order=TAG_ORDER,
        palette=algo_palette,
        markers=marker_map,
        s=28,
        alpha=0.24,
        legend=False,
        ax=ax,
    )

    # 2) Primary signal: group mean +- std with labels.
    grouped = (
        df.groupby(["algo", "tag"])[["test_cvar", "test_cumulative_return"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    centers: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for _, row in grouped.iterrows():
        algo = row[("algo", "")]
        tag = row[("tag", "")]
        x_mean = float(row[("test_cvar", "mean")])
        y_mean = float(row[("test_cumulative_return", "mean")])
        x_std = float(row[("test_cvar", "std")])
        y_std = float(row[("test_cumulative_return", "std")])
        centers[(algo, tag)] = (x_mean, y_mean)

        ax.errorbar(
            x_mean,
            y_mean,
            xerr=x_std,
            yerr=y_std,
            fmt=marker_map[tag],
            markersize=8,
            markeredgecolor="black",
            markerfacecolor=algo_palette[algo],
            ecolor=algo_palette[algo],
            elinewidth=1.1,
            capsize=3,
            alpha=0.95,
        )
        ax.annotate(
            f"{ALGO_SHORT[algo]}-{('R' if tag == 'with_cvar' else 'N')}",
            (x_mean, y_mean),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8.5,
            alpha=0.9,
        )

    # 3) Direction arrow: no_cvar -> with_cvar for each algorithm.
    for algo in ALGO_ORDER:
        src = centers.get((algo, "no_cvar"))
        dst = centers.get((algo, "with_cvar"))
        if src is None or dst is None:
            continue
        ax.annotate(
            "",
            xy=dst,
            xytext=src,
            arrowprops=dict(arrowstyle="->", lw=1.4, color=algo_palette[algo], alpha=0.85),
        )

    ax.set_xlabel("Test CVaR (95%, lower is better)")
    ax.set_ylabel("Test Cumulative Return (higher is better)")
    ax.set_title("Risk-Return Frontier (Means, Dispersion, and Direction)")
    ax.grid(True, alpha=0.25)

    # Compact legend by algorithm and setting.
    algo_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=algo_palette[a], markersize=8, label=ALGO_SHORT[a])
        for a in ALGO_ORDER
    ]
    tag_handles = [
        plt.Line2D([0], [0], marker=marker_map[t], color="black", linestyle="None", markersize=7, label=TAG_LABELS[t])
        for t in TAG_ORDER
    ]
    legend1 = ax.legend(
        handles=algo_handles,
        title="Algorithm",
        fontsize=8.5,
        title_fontsize=9,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        borderaxespad=0.2,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=tag_handles,
        title="Setting",
        fontsize=8.5,
        title_fontsize=9,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
        borderaxespad=0.2,
    )

    finish_figure(fig, fig_path)


def make_paired_delta_heatmap(delta_df: pd.DataFrame, fig_path: Path) -> None:
    plot_metrics = [
        "test_cumulative_return",
        "test_cvar",
        "test_max_drawdown",
        "test_total_reward",
    ]
    view = delta_df[delta_df["metric"].isin(plot_metrics)].copy()

    mat = view.pivot(index="algo", columns="metric", values="delta_mean_with_minus_no")
    mat = mat.reindex(index=ALGO_ORDER, columns=plot_metrics)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.0,
        linewidths=0.5,
        cbar_kws={"label": "Mean Delta (with_cvar - no_cvar)"},
        ax=ax,
    )

    ax.set_yticklabels([ALGO_LABELS.get(a, a) for a in mat.index], rotation=0)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in mat.columns], rotation=20, ha="right")
    ax.set_title("Paired Effect of CVaR Reward on Metrics")

    finish_figure(fig, fig_path)


def make_learning_curves(runs_root: Path, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=False)

    for i, algo in enumerate(ALGO_ORDER):
        ax = axes[i]
        history_path = runs_root / algo / "eval" / "history_table.csv"
        if not history_path.exists():
            ax.text(0.5, 0.5, f"Missing: {history_path}", ha="center", va="center")
            ax.axis("off")
            continue

        hist = pd.read_csv(history_path)
        if hist.empty:
            ax.text(0.5, 0.5, "No history", ha="center", va="center")
            ax.axis("off")
            continue

        summary = (
            hist.groupby(["episode", "tag"])["episode_reward"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "reward_mean", "std": "reward_std"})
        )

        for tag in TAG_ORDER:
            subset = summary[summary["tag"] == tag].sort_values("episode")
            if subset.empty:
                continue
            x = subset["episode"].to_numpy()
            y = subset["reward_mean"].to_numpy()
            std = subset["reward_std"].fillna(0.0).to_numpy()
            label = TAG_LABELS.get(tag, str(tag))

            ax.plot(x, y, label=label, linewidth=1.9)
            ax.fill_between(x, y - std, y + std, alpha=0.2)

        ax.set_title(ALGO_LABELS.get(algo, algo))
        ax.set_xlabel("Episode")
        if i == 0:
            ax.set_ylabel("Episode Reward")
        ax.grid(True, alpha=0.25)

        if i == 2:
            ax.legend(frameon=True, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.2)

    fig.suptitle("Learning Curves (Mean ± Std over Seeds)", y=0.96, fontsize=14)
    finish_figure(fig, fig_path, tight_rect=(0, 0, 1, 0.91))


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_csv(index=False)
    path.write_text(md + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    runs_root = Path(args.runs_root)
    fig_dir = Path(args.fig_dir)
    table_dir = Path(args.table_dir)

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    set_report_theme()
    rng = np.random.default_rng(args.bootstrap_seed)

    summary = pd.read_csv(summary_path)
    summary = format_df(summary)

    table_a = table_mean_std_ci(summary, n_bootstrap=args.bootstrap_samples, rng=rng)
    table_b = table_paired_deltas(summary, n_bootstrap=args.bootstrap_samples, rng=rng)

    table_a.to_csv(table_dir / "table_mean_std_ci.csv", index=False)
    table_b.to_csv(table_dir / "table_paired_deltas.csv", index=False)
    save_markdown_table(table_a, table_dir / "table_mean_std_ci.md")
    save_markdown_table(table_b, table_dir / "table_paired_deltas.md")

    make_metric_boxplots(summary, fig_dir / "fig_metric_boxplots.png")
    make_risk_return_scatter(summary, fig_dir / "fig_risk_return_frontier.png")
    make_paired_delta_heatmap(table_b, fig_dir / "fig_paired_delta_heatmap.png")
    make_learning_curves(runs_root, fig_dir / "fig_learning_curves.png")

    print(f"Saved tables to: {table_dir}")
    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
