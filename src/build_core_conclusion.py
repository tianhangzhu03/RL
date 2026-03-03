"""Build a core conclusion table for a primary asset.

Table layout (ordered for report use):
- Baseline(with_cvar) [buy_hold]
- q_learning (with_cvar, baseline cfg)
- tuned nstep_sarsa (from a tuned eval root)

Outputs CSV/Markdown plus mean/std stats CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build core conclusion table from baseline suite + tuned eval")
    p.add_argument("--asset", required=True, help="Asset ticker label, e.g. XLF or QQQ")
    p.add_argument("--baseline-suite-root", required=True, help="Root from src.run_suite, e.g. runs_xlf_baseline")
    p.add_argument("--tuned-eval-root", required=True, help="Root from src.run_best_configs for tuned eval")
    p.add_argument("--tuned-label", default="nstep_sarsa V1 (with_cvar, tuned)")
    p.add_argument("--tuned-tag", default=None, help="Optional tag filter for the tuned source (e.g., with_cvar when using a suite summary)")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def choose_latest_suite_summary(root: Path) -> Path:
    latest_ptr = root / "LATEST.txt"
    if latest_ptr.exists():
        text = latest_ptr.read_text(encoding="utf-8").strip()
        p = Path(text)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        candidate = p / "suite_summary.csv"
        if candidate.exists():
            return candidate
    # direct suite_summary at root (run_suite style)
    direct = root / "suite_summary.csv"
    if direct.exists():
        return direct
    candidates = sorted(root.glob("*/suite_summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"No suite_summary.csv found under {root}")
    return candidates[-1]


def load_buy_hold_from_baseline_suite(baseline_root: Path) -> pd.DataFrame:
    # Use q_learning/with_cvar subdirs as canonical source for buy_hold metrics under baseline config.
    files = sorted((baseline_root / "q_learning" / "with_cvar").glob("seed_*/baseline_metrics.json"))
    if not files:
        # fallback recursive search
        files = sorted(baseline_root.glob("**/q_learning/with_cvar/**/baseline_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No q_learning/with_cvar baseline_metrics.json found under {baseline_root}")
    rows: list[dict[str, Any]] = []
    for path in files:
        d = json.load(open(path, "r", encoding="utf-8"))
        b = d["buy_hold"] if "buy_hold" in d else d
        seed = next((int(p.split("_", 1)[1]) for p in path.parts if p.startswith("seed_") and p.split("_", 1)[1].isdigit()), np.nan)
        rows.append(
            {
                "seed": seed,
                "reward": float(b["total_reward"]),
                "return": float(b["cumulative_return"]),
                "CVaR": float(b["cvar"]),
                "MDD": float(b["max_drawdown"]),
                "train_time_s": 0.0,
            }
        )
    return pd.DataFrame(rows)


def extract_algo_rows(summary_csv: Path, algo: str, tag: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if "algo" in df.columns:
        df = df[df["algo"] == algo].copy()
    if tag is not None and "tag" in df.columns:
        df = df[df["tag"] == tag].copy()
    if df.empty:
        raise ValueError(f"No rows found for algo={algo!r}, tag={tag!r} in {summary_csv}")
    # Normalize naming across suite_summary variants.
    rename_map = {
        "test_total_reward": "reward",
        "test_cumulative_return": "return",
        "test_cvar": "CVaR",
        "test_max_drawdown": "MDD",
        "train_seconds": "train_time_s",
    }
    out = df.rename(columns=rename_map)
    needed = ["reward", "return", "CVaR", "MDD", "train_time_s"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {summary_csv}")
    cols = ["seed"] + needed if "seed" in out.columns else needed
    return out[cols].copy()


def summarize_mean_std(df: pd.DataFrame, model_label: str, source: str, n_runs: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = ["reward", "return", "CVaR", "MDD", "train_time_s"]
    mean_row: dict[str, Any] = {"model": model_label}
    stats_row: dict[str, Any] = {"model": model_label, "n_runs": int(n_runs if n_runs is not None else len(df)), "source": source}
    for m in metrics:
        vals = df[m].astype(float).to_numpy()
        mean_row[m] = float(np.mean(vals))
        stats_row[f"{m}_mean"] = float(np.mean(vals))
        stats_row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    mean_row["n_runs"] = int(stats_row["n_runs"])
    mean_row["source"] = source
    return mean_row, stats_row


def to_markdown_fallback(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "```"


def main() -> None:
    args = parse_args()
    asset = args.asset.upper()
    baseline_root = Path(args.baseline_suite_root)
    tuned_eval_root = Path(args.tuned_eval_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = choose_latest_suite_summary(baseline_root)
    tuned_summary = choose_latest_suite_summary(tuned_eval_root)

    buyhold_df = load_buy_hold_from_baseline_suite(baseline_root)
    ql_df = extract_algo_rows(baseline_summary, algo="q_learning", tag="with_cvar")
    nstep_df = extract_algo_rows(tuned_summary, algo="nstep_sarsa", tag=args.tuned_tag)

    rows_mean: list[dict[str, Any]] = []
    rows_stats: list[dict[str, Any]] = []

    for label, df, src in [
        ("Baseline(with_cvar)", buyhold_df, f"{baseline_root.as_posix()}/q_learning/with_cvar/*/baseline_metrics.json (buy_hold)"),
        ("q_learning (with_cvar, baseline cfg)", ql_df, f"{baseline_summary.as_posix()} [algo=q_learning,tag=with_cvar]"),
        (
            args.tuned_label,
            nstep_df,
            f"{tuned_summary.as_posix()} [algo=nstep_sarsa"
            + (f",tag={args.tuned_tag}" if args.tuned_tag is not None else "")
            + "]",
        ),
    ]:
        mean_row, stats_row = summarize_mean_std(df, label, src)
        rows_mean.append(mean_row)
        rows_stats.append(stats_row)

    core = pd.DataFrame(rows_mean)
    stats = pd.DataFrame(rows_stats)

    # Reorder columns for report use.
    core = core[["model", "reward", "return", "CVaR", "MDD", "train_time_s", "n_runs", "source"]]

    stem = f"table_{asset.lower()}_main_conclusion"
    core_csv = out_dir / f"{stem}.csv"
    core_md = out_dir / f"{stem}.md"
    stats_csv = out_dir / f"{stem}_stats.csv"
    stats_md = out_dir / f"{stem}_stats.md"

    core.to_csv(core_csv, index=False)
    core_md.write_text("\n".join([f"# {asset} Main Conclusion Table", "", to_markdown_fallback(core)]), encoding="utf-8")
    stats.to_csv(stats_csv, index=False)
    stats_md.write_text("\n".join([f"# {asset} Main Conclusion Table (Mean / Std)", "", to_markdown_fallback(stats)]), encoding="utf-8")

    print(f"Saved core conclusion tables to: {out_dir}")


if __name__ == "__main__":
    main()
