"""Summarize QQQ design ablation experiments (D0/D1/D2) into report tables and figures.

Designs:
- D0: original target-position action space
- D1: delta action space
- D2: delta action space + lower opportunity-cost coefficient
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.plot_style import REPORT_COLORS, finish_figure, set_report_theme


METRIC_ALIASES = {
    "total_reward": "reward",
    "cumulative_return": "return",
    "cvar": "cvar",
    "max_drawdown": "mdd",
    "turnover_per_step": "turnover_per_step",
    "full_position_ratio": "full_position_ratio",
}


@dataclass(frozen=True)
class AblationSpec:
    code: str
    label: str
    config_path: Path
    tuning_root: Path
    eval_root: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate D0/D1/D2 design-ablation report for QQQ")
    p.add_argument("--d0-config", default="configs/ablations/qqq_d0_target_v4aligned.yaml")
    p.add_argument("--d1-config", default="configs/ablations/qqq_d1_delta_v4aligned.yaml")
    p.add_argument("--d2-config", default="configs/ablations/qqq_d2_delta_lowopp_v4aligned.yaml")
    p.add_argument("--e-config", default="")
    p.add_argument("--d0-tuning-root", default="runs/design_ablation_qqq_d0_tuning")
    p.add_argument("--d1-tuning-root", default="runs/design_ablation_qqq_d1_tuning")
    p.add_argument("--d2-tuning-root", default="runs/design_ablation_qqq_d2_tuning")
    p.add_argument("--e-tuning-root", default="")
    p.add_argument("--d0-eval-root", default="runs/design_ablation_qqq_d0_eval")
    p.add_argument("--d1-eval-root", default="runs/design_ablation_qqq_d1_eval")
    p.add_argument("--d2-eval-root", default="runs/design_ablation_qqq_d2_eval")
    p.add_argument("--e-eval-root", default="")
    p.add_argument("--out-dir", default="report/qqq/design_ablation_d0d1d2")
    return p.parse_args()


def _resolve_latest_run_dir(root: Path) -> Path:
    latest_ptr = root / "LATEST.txt"
    if latest_ptr.exists():
        text = latest_ptr.read_text(encoding="utf-8").strip()
        p = Path(text)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists():
            return p
    candidates = sorted([p for p in root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return candidates[-1]


def _resolve_latest_tuning_dir(root: Path) -> Path:
    root = root.resolve()
    algo_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not algo_dirs:
        raise FileNotFoundError(f"No algo directories found under {root}")
    if len(algo_dirs) == 1:
        algo_dir = algo_dirs[0]
    else:
        ns = [p for p in algo_dirs if p.name == "nstep_sarsa"]
        algo_dir = ns[0] if ns else sorted(algo_dirs)[-1]
    runs = sorted([p for p in algo_dir.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No tuning run directories found under {algo_dir}")
    return runs[-1]


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_design_spec_rows(specs: list[AblationSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        cfg = _read_yaml(spec.config_path)
        env_cfg = cfg.get("env", {})
        actions = env_cfg.get("actions", cfg.get("actions", cfg.get("training", {}).get("actions", [])))
        rows.append(
            {
                "design": spec.code,
                "label": spec.label,
                "config_path": str(spec.config_path),
                "action_mode": env_cfg.get("action_mode", "target"),
                "actions": json.dumps(actions),
                "opportunity_cost_coeff": float(env_cfg.get("opportunity_cost_coeff", np.nan)),
                "lambda_cvar": float(env_cfg.get("lambda_cvar", np.nan)),
                "lambda_vol": float(env_cfg.get("lambda_vol", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def design_order(designs: list[str]) -> list[str]:
    rank = {"D0": 0, "D1": 1, "D2": 2, "E": 3}
    return sorted(designs, key=lambda d: (rank.get(d, 999), d))


def load_best_trial_rows(specs: list[AblationSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        tuning_dir = _resolve_latest_tuning_dir(spec.tuning_root)
        best_path = tuning_dir / "best_trial.json"
        if not best_path.exists():
            raise FileNotFoundError(f"Missing best_trial.json in {tuning_dir}")
        d = json.load(open(best_path, "r", encoding="utf-8"))
        row = {
            "design": spec.code,
            "label": spec.label,
            "tuning_run_dir": str(tuning_dir),
            "candidate_id": int(d.get("candidate_id", -1)),
            "phase": int(d.get("phase", -1)),
            "is_feasible": int(d.get("is_feasible", 0)),
            "risk_feasible": int(d.get("risk_feasible", 0)),
            "profit_feasible": int(d.get("profit_feasible", 0)),
            "active_feasible": int(d.get("active_feasible", 0)),
            "feasible_rate": float(d.get("feasible_rate", np.nan)),
            "risk_feasible_rate": float(d.get("risk_feasible_rate", np.nan)),
            "train_seconds_val_tuning": float(d.get("train_seconds", np.nan)),
            "val_total_reward": float(d.get("val_total_reward", np.nan)),
            "val_total_reward_std": float(d.get("val_total_reward_std", np.nan)),
            "val_cumulative_return": float(d.get("val_cumulative_return", np.nan)),
            "val_cumulative_return_std": float(d.get("val_cumulative_return_std", np.nan)),
            "val_cvar": float(d.get("val_cvar", np.nan)),
            "val_max_drawdown": float(d.get("val_max_drawdown", np.nan)),
            "val_excess_reward_vs_buyhold": float(d.get("val_excess_reward_vs_buyhold", np.nan)),
            "val_excess_return_vs_buyhold": float(d.get("val_excess_return_vs_buyhold", np.nan)),
            "val_excess_cvar_vs_buyhold": float(d.get("val_excess_cvar_vs_buyhold", np.nan)),
            "val_excess_mdd_vs_buyhold": float(d.get("val_excess_mdd_vs_buyhold", np.nan)),
            "val_turnover_gap_vs_buyhold": float(d.get("val_turnover_gap_vs_buyhold", np.nan)),
            "val_full_position_ratio_gap_vs_buyhold": float(d.get("val_full_position_ratio_gap_vs_buyhold", np.nan)),
            "effective_constraint_max_cvar": float(d.get("effective_constraint_max_cvar", np.nan)),
            "effective_constraint_min_max_drawdown": float(d.get("effective_constraint_min_max_drawdown", np.nan)),
            "effective_return_floor": float(d.get("effective_return_floor", np.nan)),
            "effective_reward_floor": float(d.get("effective_reward_floor", np.nan)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _seed_num_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed_"):
            suffix = part.split("_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
    raise ValueError(f"Could not parse seed from {path}")


def load_seed_level_metrics(spec: AblationSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_dir = _resolve_latest_run_dir(spec.eval_root)
    seed_dirs = sorted((eval_dir / "nstep_sarsa").glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories under {eval_dir / 'nstep_sarsa'}")

    rows_abs: list[dict[str, Any]] = []
    rows_delta: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        seed = _seed_num_from_path(seed_dir)
        test = json.load(open(seed_dir / "test_metrics.json", "r", encoding="utf-8"))
        baselines = json.load(open(seed_dir / "baseline_metrics.json", "r", encoding="utf-8"))
        b = baselines["buy_hold"]

        abs_row = {
            "design": spec.code,
            "label": spec.label,
            "seed": seed,
            "reward": float(test["total_reward"]),
            "return": float(test["cumulative_return"]),
            "cvar": float(test["cvar"]),
            "mdd": float(test["max_drawdown"]),
            "turnover_per_step": float(test["turnover_per_step"]),
            "full_position_ratio": float(test["full_position_ratio"]),
            "buyhold_reward": float(b["total_reward"]),
            "buyhold_return": float(b["cumulative_return"]),
            "buyhold_cvar": float(b["cvar"]),
            "buyhold_mdd": float(b["max_drawdown"]),
            "buyhold_turnover_per_step": float(b["turnover_per_step"]),
            "buyhold_full_position_ratio": float(b["full_position_ratio"]),
        }
        rows_abs.append(abs_row)

        rows_delta.append(
            {
                "design": spec.code,
                "label": spec.label,
                "seed": seed,
                "d_reward": abs_row["reward"] - abs_row["buyhold_reward"],
                "d_return": abs_row["return"] - abs_row["buyhold_return"],
                "d_cvar": abs_row["cvar"] - abs_row["buyhold_cvar"],
                "d_mdd": abs_row["mdd"] - abs_row["buyhold_mdd"],
                "d_turnover_per_step": abs_row["turnover_per_step"] - abs_row["buyhold_turnover_per_step"],
                "d_full_position_ratio": abs_row["full_position_ratio"] - abs_row["buyhold_full_position_ratio"],
            }
        )

    abs_df = pd.DataFrame(rows_abs)
    delta_df = pd.DataFrame(rows_delta)

    # attach train time/test agg metrics from suite summary for consistent reporting
    suite = pd.read_csv(eval_dir / "suite_summary.csv")
    suite = suite[suite["algo"] == "nstep_sarsa"].copy()
    suite["seed"] = suite["seed"].astype(int)
    merged = abs_df.merge(
        suite[["seed", "train_seconds", "test_total_reward", "test_cumulative_return", "test_cvar", "test_max_drawdown"]],
        on="seed",
        how="left",
    )
    return merged, delta_df


def load_test_agg_rows(specs: list[AblationSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    abs_all: list[pd.DataFrame] = []
    delta_all: list[pd.DataFrame] = []
    agg_rows: list[dict[str, Any]] = []

    for spec in specs:
        abs_df, delta_df = load_seed_level_metrics(spec)
        abs_all.append(abs_df)
        delta_all.append(delta_df)

        agg_rows.append(
            {
                "design": spec.code,
                "label": spec.label,
                "n_runs": int(len(abs_df)),
                "train_seconds_mean": float(abs_df["train_seconds"].mean()),
                "reward_mean": float(abs_df["reward"].mean()),
                "reward_std": float(abs_df["reward"].std(ddof=1)),
                "return_mean": float(abs_df["return"].mean()),
                "return_std": float(abs_df["return"].std(ddof=1)),
                "cvar_mean": float(abs_df["cvar"].mean()),
                "cvar_std": float(abs_df["cvar"].std(ddof=1)),
                "mdd_mean": float(abs_df["mdd"].mean()),
                "mdd_std": float(abs_df["mdd"].std(ddof=1)),
                "turnover_per_step_mean": float(abs_df["turnover_per_step"].mean()),
                "full_position_ratio_mean": float(abs_df["full_position_ratio"].mean()),
                "buyhold_reward_mean": float(abs_df["buyhold_reward"].mean()),
                "buyhold_return_mean": float(abs_df["buyhold_return"].mean()),
                "buyhold_cvar_mean": float(abs_df["buyhold_cvar"].mean()),
                "buyhold_mdd_mean": float(abs_df["buyhold_mdd"].mean()),
            }
        )

    return pd.DataFrame(agg_rows), pd.concat(abs_all, ignore_index=True), pd.concat(delta_all, ignore_index=True)


def summarize_delta_means(delta_seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in delta_seed_df.columns if c.startswith("d_")]
    group = delta_seed_df.groupby(["design", "label"], as_index=False)[metric_cols].mean()
    return group


def make_test_metric_bars(test_agg_df: pd.DataFrame, out_path: Path) -> None:
    order = design_order(test_agg_df["design"].astype(str).unique().tolist())
    df = test_agg_df.copy()
    df["design"] = pd.Categorical(df["design"], categories=order, ordered=True)
    df = df.sort_values("design").reset_index(drop=True)
    color_map = {
        "D0": REPORT_COLORS["blue_light"],
        "D1": REPORT_COLORS["orange"],
        "D2": REPORT_COLORS["green"],
        "E": REPORT_COLORS["blue"],
    }
    colors = [color_map.get(d, "#4c72b0") for d in df["design"].astype(str).tolist()]

    panels = [
        ("reward_mean", "Reward"),
        ("return_mean", "Return"),
        ("cvar_mean", "CVaR (95%)"),
        ("mdd_mean", "Max Drawdown"),
        ("train_seconds_mean", "Train Time (s)"),
        ("turnover_per_step_mean", "Turnover / Step"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for ax, (col, title) in zip(axes.flatten(), panels):
        vals = df[col].to_numpy(dtype=float)
        x = np.arange(len(df))
        bars = ax.bar(x, vals, color=colors, edgecolor=REPORT_COLORS["ink"], linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        if col.endswith("_mean") and col.replace("_mean", "_std") in df.columns and "train_seconds" not in col and "turnover" not in col:
            err = df[col.replace("_mean", "_std")].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(x, vals, yerr=err, fmt="none", ecolor="#6B7A8C", elinewidth=1.0, capsize=3, zorder=5)
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-12
        for b in bars:
            y = b.get_height()
            offset = 0.02 * yspan
            ax.text(
                b.get_x() + b.get_width() / 2,
                y + (offset if y >= 0 else -offset),
                f"{y:.3f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=8,
            )

    fig.suptitle("QQQ Design Ablation (D0/D1/D2): Test Metrics After Fixed Tuning Protocol", y=0.98, fontsize=13)
    finish_figure(fig, out_path, tight_rect=(0, 0, 1, 0.94))


def make_buyhold_delta_bars(delta_mean_df: pd.DataFrame, out_path: Path) -> None:
    order = design_order(delta_mean_df["design"].astype(str).unique().tolist())
    df = delta_mean_df.copy()
    df["design"] = pd.Categorical(df["design"], categories=order, ordered=True)
    df = df.sort_values("design").reset_index(drop=True)
    color_map = {
        "D0": REPORT_COLORS["blue_light"],
        "D1": REPORT_COLORS["orange"],
        "D2": REPORT_COLORS["green"],
        "E": REPORT_COLORS["blue"],
    }
    colors = [color_map.get(d, "#4c72b0") for d in df["design"].astype(str).tolist()]

    panels = [
        ("d_reward", "Delta Reward vs buy_hold"),
        ("d_return", "Delta Return vs buy_hold"),
        ("d_cvar", "Delta CVaR vs buy_hold (lower is better)"),
        ("d_mdd", "Delta MDD vs buy_hold (higher is better)"),
        ("d_turnover_per_step", "Delta Turnover / Step"),
        ("d_full_position_ratio", "Delta Full-Position Ratio"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4))
    for ax, (col, title) in zip(axes.flatten(), panels):
        vals = df[col].to_numpy(dtype=float)
        x = np.arange(len(df))
        bars = ax.bar(x, vals, color=colors, edgecolor=REPORT_COLORS["ink"], linewidth=0.5)
        ax.axhline(0.0, color="#B8C4D3", linewidth=1.0)
        ax.set_title(title, fontsize=9.8)
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-12
        for b in bars:
            y = b.get_height()
            offset = 0.03 * yspan
            ax.text(
                b.get_x() + b.get_width() / 2,
                y + (offset if y >= 0 else -offset),
                f"{y:.3f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=8,
            )

    fig.suptitle("QQQ Design Ablation: Test Delta vs buy_hold (Same Tuned Eval Environment)", y=0.98, fontsize=13)
    finish_figure(fig, out_path, tight_rect=(0, 0, 1, 0.94))


def make_validation_excess_figure(best_df: pd.DataFrame, out_path: Path) -> None:
    order = design_order(best_df["design"].astype(str).unique().tolist())
    df = best_df.copy()
    df["design"] = pd.Categorical(df["design"], categories=order, ordered=True)
    df = df.sort_values("design").reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    color_map = {
        "D0": REPORT_COLORS["blue_light"],
        "D1": REPORT_COLORS["orange"],
        "D2": REPORT_COLORS["green"],
        "E": REPORT_COLORS["blue"],
    }
    colors = [color_map.get(d, "#4c72b0") for d in df["design"].astype(str).tolist()]
    x = np.arange(len(df))

    # validation excess reward/return
    for ax, col, title in [
        (axes[0], "val_excess_reward_vs_buyhold", "Validation Excess Reward vs buy_hold"),
        (axes[1], "val_excess_return_vs_buyhold", "Validation Excess Return vs buy_hold"),
    ]:
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(x, vals, color=colors, edgecolor=REPORT_COLORS["ink"], linewidth=0.5)
        ax.axhline(0.0, color="#B8C4D3", linewidth=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=9)
        for b in bars:
            y = b.get_height()
            yspan = ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-12
            off = 0.04 * yspan
            ax.text(b.get_x() + b.get_width()/2, y + (off if y >= 0 else -off), f"{y:.3f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=8)

    # feasibility flags heatmap-like table
    flag_cols = ["risk_feasible", "profit_feasible", "active_feasible", "is_feasible"]
    mat = df[flag_cols].to_numpy(dtype=float)
    im = axes[2].imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[2].set_title("Best-Trial Feasibility Flags", fontsize=10)
    axes[2].set_xticks(np.arange(len(flag_cols)))
    axes[2].set_xticklabels(["Risk", "Profit", "Active", "All"], fontsize=8)
    axes[2].set_yticks(np.arange(len(df)))
    axes[2].set_yticklabels(order, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axes[2].text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=9, color="#243447")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("QQQ Design Ablation: Validation Selection Signals (Best Trial)", y=0.99, fontsize=12.5)
    finish_figure(fig, out_path, tight_rect=(0, 0, 1, 0.90))


def write_assessment_md(
    *,
    out_path: Path,
    specs_df: pd.DataFrame,
    best_df: pd.DataFrame,
    test_agg_df: pd.DataFrame,
    delta_mean_df: pd.DataFrame,
) -> None:
    test_by = test_agg_df.set_index("design")
    delta_by = delta_mean_df.set_index("design")
    best_by = best_df.set_index("design")

    def f(v: Any) -> str:
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    lines: list[str] = []
    lines.append("# QQQ Design Ablation Assessment (D0/D1/D2)")
    lines.append("")
    lines.append("Fixed tuning protocol across all designs; only environment/action design changed.")
    lines.append("")
    lines.append("## Design Specs")
    lines.append("")
    lines.append(specs_df.to_csv(index=False))
    lines.append("")
    lines.append("## Core Findings")
    lines.append("")
    for d in design_order(test_agg_df["design"].astype(str).unique().tolist()):
        if d not in test_by.index:
            continue
        t = test_by.loc[d]
        dm = delta_by.loc[d]
        b = best_by.loc[d]
        lines.append(f"- **{d}**: test reward={f(t['reward_mean'])}, return={f(t['return_mean'])}, CVaR={f(t['cvar_mean'])}, MDD={f(t['mdd_mean'])}; "
                     f"vs buy_hold delta return={f(dm['d_return'])}, delta reward={f(dm['d_reward'])}; "
                     f"best-trial val excess return={f(b['val_excess_return_vs_buyhold'])}, excess reward={f(b['val_excess_reward_vs_buyhold'])}.")
    lines.append("")
    lines.append("## Diagnostic Interpretation (Template)")
    lines.append("")
    lines.append("- If D1 improves validation excess candidates but final test still collapses to ~buy_hold, action-space flexibility alone is not sufficient under current reward/penalty mix.")
    lines.append("- If D2 shows stronger positive excess (or less negative excess) than D1, opportunity-cost pressure is a major cause of the buy_hold attractor.")
    lines.append("- If D0/D1/D2 all remain near buy_hold despite fixed protocol and active-selection ranking, the limitation is primarily task design/data regime rather than TD algorithm choice.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown_fallback(df: pd.DataFrame) -> str:
    """Render markdown if tabulate exists; otherwise fall back to fenced CSV text."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "```"


def main() -> None:
    args = parse_args()
    set_report_theme()

    specs = [
        AblationSpec("D0", "Original target actions (V4-aligned)", Path(args.d0_config), Path(args.d0_tuning_root), Path(args.d0_eval_root)),
        AblationSpec("D1", "Delta actions", Path(args.d1_config), Path(args.d1_tuning_root), Path(args.d1_eval_root)),
        AblationSpec("D2", "Delta actions + lower opportunity cost", Path(args.d2_config), Path(args.d2_tuning_root), Path(args.d2_eval_root)),
    ]
    if args.e_config and args.e_tuning_root and args.e_eval_root:
        specs.append(
            AblationSpec(
                "E",
                "Core+overlay (robust)",
                Path(args.e_config),
                Path(args.e_tuning_root),
                Path(args.e_eval_root),
            )
        )

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    specs_df = load_design_spec_rows(specs)
    best_df = load_best_trial_rows(specs)
    test_agg_df, seed_abs_df, delta_seed_df = load_test_agg_rows(specs)
    delta_mean_df = summarize_delta_means(delta_seed_df)

    # Save tables
    specs_df.to_csv(table_dir / "table_design_specs.csv", index=False)
    best_df.to_csv(table_dir / "table_best_trial_validation_summary.csv", index=False)
    test_agg_df.to_csv(table_dir / "table_test_aggregate_metrics.csv", index=False)
    seed_abs_df.to_csv(table_dir / "table_test_seed_metrics_and_buyhold.csv", index=False)
    delta_seed_df.to_csv(table_dir / "table_test_seed_deltas_vs_buyhold.csv", index=False)
    delta_mean_df.to_csv(table_dir / "table_test_mean_deltas_vs_buyhold.csv", index=False)

    # Markdown convenience exports
    for name, df in [
        ("table_design_specs.md", specs_df),
        ("table_best_trial_validation_summary.md", best_df),
        ("table_test_aggregate_metrics.md", test_agg_df),
        ("table_test_mean_deltas_vs_buyhold.md", delta_mean_df),
    ]:
        (table_dir / name).write_text(dataframe_to_markdown_fallback(df), encoding="utf-8")

    # Figures
    make_test_metric_bars(test_agg_df, fig_dir / "fig_test_metrics_d0_d1_d2.png")
    make_buyhold_delta_bars(delta_mean_df, fig_dir / "fig_test_deltas_vs_buyhold_d0_d1_d2.png")
    make_validation_excess_figure(best_df, fig_dir / "fig_validation_selection_signals_d0_d1_d2.png")

    # Lightweight narrative scaffold
    write_assessment_md(
        out_path=out_dir / "design_ablation_assessment.md",
        specs_df=specs_df,
        best_df=best_df,
        test_agg_df=test_agg_df,
        delta_mean_df=delta_mean_df,
    )

    # Alias-friendly readme
    present_designs = ",".join(design_order([s.code for s in specs]))
    readme = [
        f"# QQQ Design Ablation ({present_designs})",
        "",
        "This report compares environment designs under the same tuning protocol (n-step SARSA + successive halving).",
        "",
        "## Tables",
        "- `tables/table_design_specs.csv`",
        "- `tables/table_best_trial_validation_summary.csv`",
        "- `tables/table_test_aggregate_metrics.csv`",
        "- `tables/table_test_seed_deltas_vs_buyhold.csv`",
        "- `tables/table_test_mean_deltas_vs_buyhold.csv`",
        "",
        "## Figures",
        "- `figures/fig_test_metrics_d0_d1_d2.png`",
        "- `figures/fig_test_deltas_vs_buyhold_d0_d1_d2.png`",
        "- `figures/fig_validation_selection_signals_d0_d1_d2.png`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Saved design ablation report to: {out_dir}")


if __name__ == "__main__":
    main()
