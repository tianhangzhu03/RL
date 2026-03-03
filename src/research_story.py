"""Generate non-chart research artifacts: scorecards and casebook narratives.

Usage:
    scripts/py.sh -m src.research_story \
      --summary runs/suite_summary.csv \
      --out-dir report/insights
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ALGO_ORDER = ["q_learning", "sg_sarsa", "nstep_sarsa"]
TAG_ORDER = ["no_cvar", "with_cvar"]

ALGO_LABELS = {
    "q_learning": "Q-learning",
    "sg_sarsa": "SG-SARSA",
    "nstep_sarsa": "n-step SARSA",
}

TAG_LABELS = {
    "no_cvar": "No CVaR",
    "with_cvar": "With CVaR",
}

BASE_CSS = """
:root{
  --bg:#f3f6fb;
  --panel:#ffffff;
  --ink:#1f2a37;
  --ink-soft:#5b6778;
  --line:#d6deea;
  --head:#e8eef7;
  --head-deep:#dbe6f4;
  --navy:#1f3a5f;
  --green:#1b5e3a;
  --red:#8a2d2d;
  --amber:#966a12;
}
*{box-sizing:border-box;}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  margin:0;
  padding:22px;
  background:var(--bg);
  color:var(--ink);
  line-height:1.52;
}
.page{
  max-width:1120px;
  margin:0 auto;
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:14px;
  padding:20px 22px;
  box-shadow:0 4px 16px rgba(18,36,62,0.05);
}
h1,h2,h3{margin:0 0 10px 0;color:var(--navy);}
h1{font-size:26px;letter-spacing:0.2px;border-bottom:2px solid var(--head-deep);padding-bottom:8px;margin-bottom:14px;}
h2{font-size:18px;margin-top:22px;}
h3{font-size:15px;margin-top:10px;color:#25486f;}
p{margin:0 0 12px 0;}
table{
  border-collapse:separate;
  border-spacing:0;
  width:100%;
  font-size:13px;
  margin:8px 0 14px 0;
  border:1px solid var(--line);
  border-radius:10px;
  overflow:hidden;
}
th,td{
  border-right:1px solid var(--line);
  border-bottom:1px solid var(--line);
  padding:8px 10px;
  text-align:right;
  vertical-align:top;
}
th:last-child,td:last-child{border-right:none;}
tbody tr:last-child td{border-bottom:none;}
th:first-child,td:first-child{text-align:left;}
th:nth-child(2),td:nth-child(2){text-align:left;}
thead th{
  background:linear-gradient(180deg,var(--head) 0%,var(--head-deep) 100%);
  color:#23466c;
  font-weight:650;
}
tbody tr:nth-child(even){background:#f9fbff;}
tbody tr:hover{background:#f2f7ff;}
.muted{color:var(--ink-soft);font-size:12px;}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;}
.card{
  border:1px solid var(--line);
  border-left:4px solid #7d95b3;
  border-radius:10px;
  padding:12px;
  background:#fbfdff;
}
.kpi{font-size:18px;font-weight:650;}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scorecards and casebook from suite summary")
    parser.add_argument("--summary", default="runs/suite_summary.csv")
    parser.add_argument("--out-dir", default="report/insights")
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "algo",
        "tag",
        "seed",
        "train_seconds",
        "test_total_reward",
        "test_cumulative_return",
        "test_max_drawdown",
        "test_cvar",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["algo"] = pd.Categorical(df["algo"], categories=ALGO_ORDER, ordered=True)
    df["tag"] = pd.Categorical(df["tag"], categories=TAG_ORDER, ordered=True)
    return df.sort_values(["algo", "tag", "seed"]).reset_index(drop=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["algo", "tag"], observed=True)
        .agg(
            n_runs=("seed", "count"),
            return_mean=("test_cumulative_return", "mean"),
            return_std=("test_cumulative_return", "std"),
            cvar_mean=("test_cvar", "mean"),
            cvar_std=("test_cvar", "std"),
            mdd_mean=("test_max_drawdown", "mean"),
            mdd_std=("test_max_drawdown", "std"),
            reward_mean=("test_total_reward", "mean"),
            reward_std=("test_total_reward", "std"),
            train_seconds_mean=("train_seconds", "mean"),
            train_seconds_std=("train_seconds", "std"),
        )
        .reset_index()
    )
    return out


def add_composite_score(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.copy()

    # Percentile-style ranking in [0, 1], with direction-aware metrics.
    df["rank_return"] = df["return_mean"].rank(pct=True)
    df["rank_cvar"] = (-df["cvar_mean"]).rank(pct=True)
    df["rank_mdd"] = df["mdd_mean"].rank(pct=True)
    df["rank_train"] = (-df["train_seconds_mean"]).rank(pct=True)

    # Weighted composite score emphasizes risk-return, then stability/speed proxy.
    df["score"] = 100.0 * (
        0.35 * df["rank_return"]
        + 0.30 * df["rank_cvar"]
        + 0.20 * df["rank_mdd"]
        + 0.15 * df["rank_train"]
    )

    return df.sort_values("score", ascending=False).reset_index(drop=True)


def format_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["algorithm"] = out["algo"].map(ALGO_LABELS)
    out["setting"] = out["tag"].map(TAG_LABELS)

    cols = [
        "algorithm",
        "setting",
        "n_runs",
        "return_mean",
        "return_std",
        "cvar_mean",
        "cvar_std",
        "mdd_mean",
        "mdd_std",
        "reward_mean",
        "reward_std",
        "train_seconds_mean",
        "train_seconds_std",
        "score",
    ]

    out = out[cols].copy()
    return out.sort_values(["score"], ascending=False).reset_index(drop=True)


def paired_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for algo, g_algo in df.groupby("algo", observed=True):
        g_no = g_algo[g_algo["tag"] == "no_cvar"].set_index("seed")
        g_with = g_algo[g_algo["tag"] == "with_cvar"].set_index("seed")
        common = sorted(set(g_no.index).intersection(set(g_with.index)))
        if not common:
            continue

        d_ret = (g_with.loc[common, "test_cumulative_return"] - g_no.loc[common, "test_cumulative_return"]).mean()
        d_cvar = (g_with.loc[common, "test_cvar"] - g_no.loc[common, "test_cvar"]).mean()
        d_mdd = (g_with.loc[common, "test_max_drawdown"] - g_no.loc[common, "test_max_drawdown"]).mean()
        d_rew = (g_with.loc[common, "test_total_reward"] - g_no.loc[common, "test_total_reward"]).mean()
        d_time = (g_with.loc[common, "train_seconds"] - g_no.loc[common, "train_seconds"]).mean()

        rows.append(
            {
                "algo": algo,
                "delta_return_with_minus_no": float(d_ret),
                "delta_cvar_with_minus_no": float(d_cvar),
                "delta_mdd_with_minus_no": float(d_mdd),
                "delta_reward_with_minus_no": float(d_rew),
                "delta_train_seconds_with_minus_no": float(d_time),
            }
        )

    return pd.DataFrame(rows)


def classify_story(row: pd.Series) -> str:
    d_ret = row["delta_return_with_minus_no"]
    d_cvar = row["delta_cvar_with_minus_no"]
    d_mdd = row["delta_mdd_with_minus_no"]

    if d_cvar < 0 and d_ret > 0 and d_mdd > 0:
        return "Risk and return improved"
    if d_cvar < 0 and d_ret <= 0:
        return "Risk-return tradeoff"
    if d_cvar >= 0 and d_ret > 0:
        return "Return-led but risk worsened"
    return "Needs retuning"


def algo_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, tag), g in df.groupby(["algo", "tag"], observed=True):
        ret = g["test_cumulative_return"].to_numpy(dtype=float)
        cvar = g["test_cvar"].to_numpy(dtype=float)
        ret_cv = float(np.std(ret, ddof=1) / (abs(np.mean(ret)) + 1e-8)) if len(ret) > 1 else 0.0
        cvar_cv = float(np.std(cvar, ddof=1) / (abs(np.mean(cvar)) + 1e-8)) if len(cvar) > 1 else 0.0
        rows.append(
            {
                "algo": algo,
                "tag": tag,
                "return_cv": ret_cv,
                "cvar_cv": cvar_cv,
            }
        )
    return pd.DataFrame(rows)


def save_html_table(df: pd.DataFrame, path: Path, title: str) -> None:
    def _fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    html = [
        "<html><head><meta charset='utf-8'>",
        f"<style>{BASE_CSS}</style></head><body><div class='page'>",
        f"<h1>{title}</h1>",
        "<p class='muted'>Generated from multi-seed evaluation summary.</p>",
        df.to_html(index=False, formatters={c: _fmt for c in df.columns}, border=0),
        "</div></body></html>",
    ]
    path.write_text("\n".join(html), encoding="utf-8")


def save_paired_effects_html(df: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in df.iterrows():
        story = str(row["story"])
        if "tradeoff" in story.lower():
            color = "var(--amber)"
        elif "improved" in story.lower():
            color = "var(--green)"
        elif "worsened" in story.lower():
            color = "var(--red)"
        else:
            color = "var(--ink-soft)"
        rows.append(
            f"<tr>"
            f"<td>{row['algorithm']}</td>"
            f"<td style='color:{color};font-weight:600;'>{story}</td>"
            f"<td>{row['delta_return_with_minus_no']:.4f}</td>"
            f"<td>{row['delta_cvar_with_minus_no']:.4f}</td>"
            f"<td>{row['delta_mdd_with_minus_no']:.4f}</td>"
            f"<td>{row['delta_reward_with_minus_no']:.4f}</td>"
            f"<td>{row['delta_train_seconds_with_minus_no']:.4f}</td>"
            f"</tr>"
        )

    body = [
        "<h1>Paired Effects (With CVaR - No CVaR)</h1>",
        "<p class='muted'>Negative delta CVaR is preferable. Higher delta max drawdown means less negative drawdown (better).</p>",
        "<table>",
        "<thead><tr><th>Algorithm</th><th>Story</th><th>Δ Return</th><th>Δ CVaR</th><th>Δ Max DD</th><th>Δ Reward</th><th>Δ Train Sec</th></tr></thead>",
        "<tbody>",
        *rows,
        "</tbody></table>",
    ]

    html = [
        "<html><head><meta charset='utf-8'>",
        f"<style>{BASE_CSS}</style></head><body><div class='page'>",
        *body,
        "</div></body></html>",
    ]
    path.write_text("\n".join(html), encoding="utf-8")


def to_markdown_plain(df: pd.DataFrame) -> str:
    """Dependency-free markdown table renderer."""
    cols = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_casebook(
    summary: pd.DataFrame,
    agg_scored: pd.DataFrame,
    paired: pd.DataFrame,
    stability: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Risk-Sensitive Trading Casebook")
    lines.append("")
    lines.append("## What This Project Demonstrates")
    lines.append("The project studies how TD control methods change behavior when CVaR-based risk terms are added to reward, focusing on risk-aware policy learning rather than pure return maximization.")
    lines.append("")

    lines.append("## Algorithm Stories")
    for _, row in paired.iterrows():
        algo = str(row["algo"])
        story = classify_story(row)
        lines.append(f"### {ALGO_LABELS.get(algo, algo)}")
        lines.append(f"- Story: {story}")
        lines.append(f"- Delta return (with - no): {row['delta_return_with_minus_no']:.4f}")
        lines.append(f"- Delta CVaR (with - no): {row['delta_cvar_with_minus_no']:.4f}")
        lines.append(f"- Delta max drawdown (with - no): {row['delta_mdd_with_minus_no']:.4f}")
        lines.append(f"- Delta total reward (with - no): {row['delta_reward_with_minus_no']:.4f}")
        lines.append(f"- Delta training time in seconds (with - no): {row['delta_train_seconds_with_minus_no']:.4f}")
        lines.append("")

    lines.append("## Best Composite Configurations")
    top = agg_scored.head(5).copy()
    top["algo"] = top["algo"].map(ALGO_LABELS)
    top["tag"] = top["tag"].map(TAG_LABELS)
    lines.append(
        to_markdown_plain(
            top[["algo", "tag", "score", "return_mean", "cvar_mean", "mdd_mean", "train_seconds_mean"]]
        ).strip()
    )
    lines.append("")

    lines.append("## Stress Runs (Worst Drawdowns)")
    worst = summary.sort_values("test_max_drawdown", ascending=True).head(5).copy()
    worst["algo"] = worst["algo"].map(ALGO_LABELS)
    worst["tag"] = worst["tag"].map(TAG_LABELS)
    lines.append(
        to_markdown_plain(
            worst[["algo", "tag", "seed", "test_max_drawdown", "test_cumulative_return", "test_cvar"]]
        ).strip()
    )
    lines.append("")

    lines.append("## Stability Snapshot")
    st = stability.copy()
    st["algo"] = st["algo"].map(ALGO_LABELS)
    st["tag"] = st["tag"].map(TAG_LABELS)
    lines.append(to_markdown_plain(st[["algo", "tag", "return_cv", "cvar_cv"]]).strip())
    lines.append("")

    lines.append("## Practical Takeaway For Trading")
    lines.append("These results are most useful as risk-aware position-sizing evidence under fixed assumptions. They are not production-ready trading rules without stronger execution, friction, and rolling retraining validation.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_casebook_html(
    summary: pd.DataFrame,
    agg_scored: pd.DataFrame,
    paired: pd.DataFrame,
    stability: pd.DataFrame,
    out_path: Path,
) -> None:
    story_cards: List[str] = []
    for _, row in paired.iterrows():
        algo = ALGO_LABELS.get(str(row["algo"]), str(row["algo"]))
        story = classify_story(row)
        story_cards.append(
            "<div class='card'>"
            f"<h3>{algo}</h3>"
            f"<p><strong>Story:</strong> {story}</p>"
            f"<p>Δ Return: {row['delta_return_with_minus_no']:.4f}<br>"
            f"Δ CVaR: {row['delta_cvar_with_minus_no']:.4f}<br>"
            f"Δ Max DD: {row['delta_mdd_with_minus_no']:.4f}<br>"
            f"Δ Reward: {row['delta_reward_with_minus_no']:.4f}</p>"
            "</div>"
        )

    top = agg_scored.head(5).copy()
    top["algo"] = top["algo"].map(ALGO_LABELS)
    top["tag"] = top["tag"].map(TAG_LABELS)
    top_table = top[["algo", "tag", "score", "return_mean", "cvar_mean", "mdd_mean", "train_seconds_mean"]]

    worst = summary.sort_values("test_max_drawdown", ascending=True).head(5).copy()
    worst["algo"] = worst["algo"].map(ALGO_LABELS)
    worst["tag"] = worst["tag"].map(TAG_LABELS)
    worst_table = worst[["algo", "tag", "seed", "test_max_drawdown", "test_cumulative_return", "test_cvar"]]

    st = stability.copy()
    st["algo"] = st["algo"].map(ALGO_LABELS)
    st["tag"] = st["tag"].map(TAG_LABELS)
    st_table = st[["algo", "tag", "return_cv", "cvar_cv"]]

    def _table_html(df: pd.DataFrame) -> str:
        return df.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)

    html = [
        "<html><head><meta charset='utf-8'>",
        f"<style>{BASE_CSS}</style></head><body><div class='page'>",
        "<h1>Risk-Sensitive Trading Casebook</h1>",
        "<p>The project studies how TD control methods behave after adding CVaR-sensitive rewards, with emphasis on risk-aware policy learning instead of pure return maximization.</p>",
        "<h2>Algorithm Stories</h2>",
        "<div class='grid'>",
        *story_cards,
        "</div>",
        "<h2>Best Composite Configurations</h2>",
        _table_html(top_table),
        "<h2>Stress Runs (Worst Drawdowns)</h2>",
        _table_html(worst_table),
        "<h2>Stability Snapshot</h2>",
        _table_html(st_table),
        "<h2>Practical Takeaway For Trading</h2>",
        "<p>These results are most useful as risk-aware position-sizing evidence under fixed assumptions. They are not production-ready trading rules without stronger execution, friction, and rolling retraining validation.</p>",
        "</div></body></html>",
    ]
    out_path.write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    agg = aggregate(summary)
    agg_scored = add_composite_score(agg)
    scorecard = format_scorecard(agg_scored)

    paired = paired_effects(summary)
    paired["story"] = paired.apply(classify_story, axis=1)
    paired["algorithm"] = paired["algo"].map(ALGO_LABELS)

    stability = algo_stability(summary)

    scorecard.to_csv(out_dir / "scorecard.csv", index=False)
    (out_dir / "scorecard.md").write_text(to_markdown_plain(scorecard), encoding="utf-8")
    save_html_table(scorecard, out_dir / "scorecard.html", "Algorithm Scorecard")

    paired_out = paired[[
        "algorithm",
        "story",
        "delta_return_with_minus_no",
        "delta_cvar_with_minus_no",
        "delta_mdd_with_minus_no",
        "delta_reward_with_minus_no",
        "delta_train_seconds_with_minus_no",
    ]].copy()
    paired_out.to_csv(out_dir / "paired_effects.csv", index=False)
    (out_dir / "paired_effects.md").write_text(to_markdown_plain(paired_out), encoding="utf-8")
    save_paired_effects_html(paired_out, out_dir / "paired_effects.html")

    write_casebook(
        summary=summary,
        agg_scored=agg_scored,
        paired=paired,
        stability=stability,
        out_path=out_dir / "casebook.md",
    )
    write_casebook_html(
        summary=summary,
        agg_scored=agg_scored,
        paired=paired,
        stability=stability,
        out_path=out_dir / "casebook.html",
    )

    print(f"Saved insight artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
