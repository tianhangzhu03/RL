"""Generate final report figures focused on algorithm effects and result interpretation.

Usage:
    scripts/py.sh -m src.final_viz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.plot_style import REPORT_COLORS, finish_figure, set_report_theme

TITLE_PAD = 10
LABEL_SIZE = 9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate final report figures")
    p.add_argument("--out-dir", default="report/final/figures")
    p.add_argument("--sync-gate-csv", default="report/xlf/core_conclusion/table_xlf_three_methods_sync_gate.csv")
    p.add_argument("--with-no-csv", default="report/xlf/core_conclusion/table_xlf_with_vs_no_deltas.csv")
    p.add_argument("--sg-budget-csv", default="report/xlf/core_conclusion/table_sg_controlled_learning_budget.csv")
    p.add_argument("--nstep-gate-json", default="report/xlf/iteration_gate_v2/promotion_decision.json")
    p.add_argument("--main-conclusion-csv", default="report/xlf/core_conclusion/table_xlf_main_conclusion.csv")
    return p.parse_args()


def _bool01(x: Any) -> int:
    return 1 if bool(x) else 0


def fig_sync_gate_deltas(sync: pd.DataFrame, out: Path) -> None:
    metric_map = {
        "reward_delta": "Reward Δ",
        "return_delta": "Return Δ",
        "CVaR_delta": "CVaR Δ",
        "MDD_delta": "MDD Δ",
    }
    long = sync.melt(
        id_vars=["algo", "promoted"],
        value_vars=list(metric_map.keys()),
        var_name="metric",
        value_name="delta",
    )
    long["metric"] = long["metric"].map(metric_map)
    long["algo_label"] = long["algo"].map(
        {"q_learning": "Q-learning", "sg_sarsa": "SG-Sarsa", "nstep_sarsa": "n-step Sarsa"}
    )

    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    palette = {
        "Q-learning": REPORT_COLORS["blue_light"],
        "SG-Sarsa": REPORT_COLORS["orange"],
        "n-step Sarsa": REPORT_COLORS["green"],
    }
    sns.barplot(data=long, x="metric", y="delta", hue="algo_label", palette=palette, ax=ax)
    ax.axhline(0.0, color=REPORT_COLORS["ink"], lw=1.0, alpha=0.8)
    ax.set_title("Synchronized Gate: Candidate Delta vs Baseline", pad=TITLE_PAD)
    ax.set_xlabel("")
    ax.set_ylabel("Delta Value", fontsize=LABEL_SIZE)
    ax.legend(title="Algorithm", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8, title_fontsize=9)
    finish_figure(fig, out, tight_rect=(0, 0, 0.85, 1))


def fig_sync_gate_checks(sync: pd.DataFrame, out: Path) -> None:
    checks = ["check_primary", "check_secondary", "check_cvar", "check_mdd", "check_seed_consistency"]
    m = sync.set_index("algo")[checks].copy()
    m = m.rename(
        index={"q_learning": "Q-learning", "sg_sarsa": "SG-Sarsa", "nstep_sarsa": "n-step Sarsa"},
        columns={
            "check_primary": "Primary",
            "check_secondary": "Secondary",
            "check_cvar": "CVaR",
            "check_mdd": "MDD",
            "check_seed_consistency": "Seed Consistency",
        },
    )
    m = m.apply(lambda s: s.map(_bool01))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    sns.heatmap(
        m,
        annot=True,
        fmt="d",
        cmap=sns.color_palette(["#d9584a", "#3f8f5d"]),
        cbar=False,
        linewidths=0.8,
        linecolor="#e3e9f2",
        ax=ax,
    )
    ax.set_title("Synchronized Gate Check Matrix (1=Pass, 0=Fail)", pad=TITLE_PAD)
    ax.set_xlabel("")
    ax.set_ylabel("")
    finish_figure(fig, out)


def fig_risk_return_time(sync: pd.DataFrame, out: Path) -> None:
    plot = sync.copy()
    plot["algo_label"] = plot["algo"].map(
        {"q_learning": "Q-learning", "sg_sarsa": "SG-Sarsa", "nstep_sarsa": "n-step Sarsa"}
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    palette = {
        "Q-learning": REPORT_COLORS["blue_light"],
        "SG-Sarsa": REPORT_COLORS["orange"],
        "n-step Sarsa": REPORT_COLORS["green"],
    }
    tmin = float(plot["train_time_candidate_s"].min())
    tmax = float(plot["train_time_candidate_s"].max())

    def bubble_size(t: float) -> float:
        if abs(tmax - tmin) < 1e-9:
            return 220.0
        return 160.0 + 280.0 * ((t - tmin) / (tmax - tmin))

    x_vals = plot["CVaR_candidate"].astype(float).to_numpy()
    y_vals = plot["return_candidate"].astype(float).to_numpy()
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    for _, r in plot.iterrows():
        x = float(r["CVaR_candidate"])
        y = float(r["return_candidate"])
        status_marker = "o" if bool(r["promoted"]) else "X"
        s = bubble_size(float(r["train_time_candidate_s"]))
        ax.scatter(
            x,
            y,
            s=s,
            marker=status_marker,
            color=palette[str(r["algo_label"])],
            edgecolor=REPORT_COLORS["ink"],
            linewidth=0.8,
            alpha=0.88,
            zorder=3,
        )
        near_right = x > (x_max - 0.18 * x_span)
        near_top = y > (y_max - 0.16 * y_span)
        dx = (-0.00022 if near_right else 0.00022)
        dy = (-0.010 if near_top else 0.008)
        ha = ("right" if near_right else "left")
        va = ("top" if near_top else "bottom")

        ax.text(
            x + dx,
            y + dy,
            str(r["algo_label"]),
            fontsize=8,
            color=REPORT_COLORS["ink"],
            ha=ha,
            va=va,
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
        )

    x_pad = 0.12 * x_span
    y_pad = 0.16 * y_span
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Two compact legends: algorithm color and promotion marker.
    algo_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=v, markeredgecolor=REPORT_COLORS["ink"], label=k)
        for k, v in palette.items()
    ]
    status_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=REPORT_COLORS["ink"], label="Promoted"),
        plt.Line2D([0], [0], marker="X", linestyle="", color=REPORT_COLORS["ink"], label="Not Promoted"),
    ]
    legend1 = ax.legend(
        handles=algo_handles, title="Algorithm", loc="upper left", bbox_to_anchor=(1.01, 1.00), borderaxespad=0.0, fontsize=8, title_fontsize=9
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=status_handles, title="Status", loc="upper left", bbox_to_anchor=(1.01, 0.60), borderaxespad=0.0, fontsize=8, title_fontsize=9
    )

    ax.set_title("Risk-Return-Time View (Synchronized Candidate Means)", pad=TITLE_PAD)
    ax.set_xlabel("CVaR (lower is safer)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Cumulative Return", fontsize=LABEL_SIZE)
    finish_figure(fig, out, tight_rect=(0, 0, 0.80, 1))


def fig_with_no_heatmap(deltas: pd.DataFrame, out: Path) -> None:
    m = deltas.set_index("algo")[
        ["delta_reward_with_minus_no", "delta_return_with_minus_no", "delta_CVaR_with_minus_no", "delta_MDD_with_minus_no"]
    ]
    m.index = m.index.map({"q_learning": "Q-learning", "sg_sarsa": "SG-Sarsa", "nstep_sarsa": "n-step Sarsa"})
    m = m.rename(
        columns={
            "delta_reward_with_minus_no": "Reward Δ (with-no)",
            "delta_return_with_minus_no": "Return Δ (with-no)",
            "delta_CVaR_with_minus_no": "CVaR Δ (with-no)",
            "delta_MDD_with_minus_no": "MDD Δ (with-no)",
        }
    )

    vmax = float(np.nanmax(np.abs(m.to_numpy(dtype=float))))
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    sns.heatmap(
        m,
        annot=True,
        fmt=".4f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.8,
        linecolor="#e3e9f2",
        cbar_kws={"shrink": 0.9},
        ax=ax,
    )
    ax.set_title("Risk-Sensitive Direction Check: with_CVaR - no_CVaR", pad=TITLE_PAD)
    ax.set_xlabel("")
    ax.set_ylabel("")
    finish_figure(fig, out)


def fig_sg_budget_tradeoff(sg: pd.DataFrame, out: Path) -> None:
    order = ["sg_ctrl_lrn", "sg_ctrl_lrn_e200", "sg_ctrl_final_guard_e200"]
    labels = {
        "sg_ctrl_lrn": "120ep",
        "sg_ctrl_lrn_e200": "200ep (base)",
        "sg_ctrl_final_guard_e200": "200ep (guard)",
    }
    x = [labels[k] for k in order]
    sg2 = sg.set_index("experiment").loc[order].reset_index()
    promoted = sg2["promoted"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))

    ax = axes[0]
    ax.plot(x, sg2["return"], marker="o", lw=2, color=REPORT_COLORS["blue"])
    ax.plot(x, sg2["reward"], marker="s", lw=2, color=REPORT_COLORS["orange"])
    for i, ok in enumerate(promoted):
        if ok:
            ax.scatter([x[i]], [float(sg2.loc[i, "return"])], s=90, marker="*", color=REPORT_COLORS["green"], zorder=5)
    ax.set_title("SG Budget Study: Return and Reward", pad=TITLE_PAD)
    ax.set_xlabel("Round")
    ax.set_ylabel("Value", fontsize=LABEL_SIZE)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(["Return", "Reward", "Promoted"], loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=8, frameon=True)

    ax = axes[1]
    ax.plot(x, sg2["cvar"], marker="o", lw=2, color=REPORT_COLORS["red"])
    ax.plot(x, sg2["train_seconds"], marker="s", lw=2, color=REPORT_COLORS["ink"])
    ax.set_title("SG Budget Study: Risk and Time", pad=TITLE_PAD)
    ax.set_xlabel("Round")
    ax.set_ylabel("CVaR / Seconds", fontsize=LABEL_SIZE)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(["CVaR", "Train Seconds"], loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=8, frameon=True)

    finish_figure(fig, out)


def fig_nstep_seed_deltas(gate_json: dict[str, Any], out: Path) -> None:
    seed_rows = gate_json.get("seed_deltas", [])
    if not seed_rows:
        return
    df = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    ax.bar(df["seed"].astype(str), df["delta_return"], color=REPORT_COLORS["blue_light"])
    ax.axhline(0.0, color=REPORT_COLORS["ink"], lw=1.0)
    ax.set_title("n-step Promotion Round: Seed-level Return Delta", pad=TITLE_PAD)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Candidate - Baseline")

    ax = axes[1]
    ax.bar(df["seed"].astype(str), df["delta_cvar"], color=REPORT_COLORS["red"])
    ax.axhline(0.0, color=REPORT_COLORS["ink"], lw=1.0)
    ax.set_title("n-step Promotion Round: Seed-level CVaR Delta", pad=TITLE_PAD)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Candidate - Baseline")

    finish_figure(fig, out)


def fig_td_methods_schema(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (
            0.03,
            REPORT_COLORS["blue_light"],
            "Tabular Q-learning",
            "Off-policy TD control\n"
            "Update:\n"
            "Q(s,a) <- Q(s,a) + a[r + g max_a' Q(s',a') - Q(s,a)]\n"
            "State handling: discretized bins",
        ),
        (
            0.36,
            REPORT_COLORS["orange"],
            "Semi-gradient Sarsa",
            "On-policy TD with linear FA\n"
            "q_hat(s,a,w) = w^T x(s,a)\n"
            "w <- w + a[r + g q_hat(s',a',w) - q_hat(s,a,w)] x(s,a)",
        ),
        (
            0.69,
            REPORT_COLORS["green"],
            "n-step Sarsa",
            "Multi-step return (n=3/5/7)\n"
            "G_t:t+n = sum_k g^k r_{t+k+1} + g^n q_hat(s_{t+n},a_{t+n})\n"
            "w <- w + a[G_t:t+n - q_hat(s_t,a_t,w)] x(s_t,a_t)",
        ),
    ]

    for x, color, title, body in boxes:
        rect = FancyBboxPatch(
            (x, 0.12),
            0.28,
            0.75,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=color,
            edgecolor=REPORT_COLORS["ink"],
            linewidth=1.0,
            alpha=0.18,
        )
        ax.add_patch(rect)
        ax.text(x + 0.01, 0.82, title, fontsize=10, fontweight="bold", color=REPORT_COLORS["ink"], va="top")
        ax.text(x + 0.01, 0.76, body, fontsize=8.6, color=REPORT_COLORS["ink"], va="top")

    for x1, x2 in [(0.31, 0.36), (0.64, 0.69)]:
        arrow = FancyArrowPatch(
            (x1, 0.50),
            (x2, 0.50),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=REPORT_COLORS["ink"],
            alpha=0.8,
        )
        ax.add_patch(arrow)

    ax.set_title("Algorithm Principles Compared: Update Mechanisms and Representation", pad=TITLE_PAD)
    finish_figure(fig, out)


def save_figure_notes(out_dir: Path) -> None:
    notes = [
        ("fig_td_methods_schema.png", "Conceptual schematic of Q-learning, SG-Sarsa, and n-step Sarsa update mechanics."),
        ("fig_xlf_sync_gate_deltas.png", "Synchronized 3-method round deltas vs per-method incumbent baselines."),
        ("fig_xlf_sync_gate_checks.png", "Pass/fail matrix for primary/secondary/risk/seed checks."),
        ("fig_xlf_risk_return_time.png", "Candidate means in risk-return plane with training-time bubble size."),
        ("fig_xlf_with_no_heatmap.png", "with_CVaR minus no_CVaR directional effects by method."),
        ("fig_sg_budget_tradeoff1.png", "SG controlled rounds showing budget sensitivity and non-monotonic outcomes."),
        ("fig_nstep_seed_deltas.png", "Seed-level deltas for promoted n-step round."),
    ]
    md = ["# Figure Notes", ""]
    for fname, desc in notes:
        md.append(f"- `{fname}`: {desc}")
    (out_dir.parent / "figure_notes.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sync = pd.read_csv(args.sync_gate_csv)
    with_no = pd.read_csv(args.with_no_csv)
    sg = pd.read_csv(args.sg_budget_csv)
    with open(args.nstep_gate_json, "r", encoding="utf-8") as f:
        nstep_gate = json.load(f)

    set_report_theme()

    fig_sync_gate_deltas(sync, out_dir / "fig_xlf_sync_gate_deltas.png")
    fig_sync_gate_checks(sync, out_dir / "fig_xlf_sync_gate_checks.png")
    fig_risk_return_time(sync, out_dir / "fig_xlf_risk_return_time.png")
    fig_with_no_heatmap(with_no, out_dir / "fig_xlf_with_no_heatmap.png")
    fig_sg_budget_tradeoff(sg, out_dir / "fig_sg_budget_tradeoff1.png")
    fig_nstep_seed_deltas(nstep_gate, out_dir / "fig_nstep_seed_deltas.png")
    fig_td_methods_schema(out_dir / "fig_td_methods_schema.png")
    save_figure_notes(out_dir)

    print(f"Saved final figures to: {out_dir}")


if __name__ == "__main__":
    main()
