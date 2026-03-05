from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y"}


def _r4(v: Any) -> float:
    return round(float(v), 4)


def _load_gate_rows(data_dir: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(data_dir / "xlf_sync_gate.csv")
    order = ["q_learning", "sg_sarsa", "nstep_sarsa"]
    labels = {
        "q_learning": "Q-learning",
        "sg_sarsa": "SG-Sarsa",
        "nstep_sarsa": "n-step Sarsa",
    }
    decision_reasons = {
        "q_learning": "Largest utility uplift, but CVaR and MDD checks fail.",
        "sg_sarsa": "Small utility gain, but still fails CVaR and MDD checks.",
        "nstep_sarsa": "Mild utility gain with all risk checks passing under the synchronized gate.",
    }

    rows: list[dict[str, Any]] = []
    for i, algo in enumerate(order, start=1):
        row = df.loc[df["algo"] == algo].iloc[0]
        promoted = _as_bool(row["promoted"])
        rows.append(
            {
                "sequence": i,
                "algo": algo,
                "algo_label": labels[algo],
                "reward_delta": _r4(row["reward_delta"]),
                "return_delta": _r4(row["return_delta"]),
                "cvar_delta": _r4(row["CVaR_delta"]),
                "mdd_delta": _r4(row["MDD_delta"]),
                "check_primary": _as_bool(row["check_primary"]),
                "check_secondary": _as_bool(row["check_secondary"]),
                "check_cvar": _as_bool(row["check_cvar"]),
                "check_mdd": _as_bool(row["check_mdd"]),
                "check_seed_consistency": _as_bool(row["check_seed_consistency"]),
                "tuned_lambda_cvar": _r4(row["tuned_lambda_cvar"]),
                "promoted": promoted,
                "decision": "PROMOTED" if promoted else "REJECTED",
                "decision_reason": decision_reasons[algo],
            }
        )
    return rows


def _load_sg_budget(data_dir: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(data_dir / "sg_controlled_budget.csv")
    keep = [
        "experiment",
        "promoted",
        "tuning_phase3_episodes",
        "alpha",
        "epsilon_decay",
        "epsilon_min",
        "return_delta_vs_baseline",
        "reward_delta_vs_baseline",
        "cvar_delta_vs_baseline",
        "mdd_delta_vs_baseline",
        "train_seconds",
    ]
    out: list[dict[str, Any]] = []
    for _, r in df[keep].iterrows():
        out.append(
            {
                "experiment": str(r["experiment"]),
                "promoted": _as_bool(r["promoted"]),
                "episodes": int(r["tuning_phase3_episodes"]),
                "alpha": _r4(r["alpha"]),
                "epsilon_decay": _r4(r["epsilon_decay"]),
                "epsilon_min": _r4(r["epsilon_min"]),
                "return_delta": _r4(r["return_delta_vs_baseline"]),
                "reward_delta": _r4(r["reward_delta_vs_baseline"]),
                "cvar_delta": _r4(r["cvar_delta_vs_baseline"]),
                "mdd_delta": _r4(r["mdd_delta_vs_baseline"]),
                "train_seconds": _r4(r["train_seconds"]),
            }
        )
    return out


def _load_with_no(data_dir: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(data_dir / "xlf_with_vs_no.csv")
    labels = {
        "q_learning": "Q-learning",
        "sg_sarsa": "SG-Sarsa",
        "nstep_sarsa": "n-step Sarsa",
    }
    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        out.append(
            {
                "algo": str(r["algo"]),
                "algo_label": labels[str(r["algo"])],
                "reward_delta": _r4(r["delta_reward_with_minus_no"]),
                "return_delta": _r4(r["delta_return_with_minus_no"]),
                "cvar_delta": _r4(r["delta_CVaR_with_minus_no"]),
                "mdd_delta": _r4(r["delta_MDD_with_minus_no"]),
            }
        )
    return out


def _load_main_conclusion(data_dir: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(data_dir / "xlf_main_conclusion.csv")
    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        out.append(
            {
                "model": str(r["model"]),
                "reward": _r4(r["reward"]),
                "return": _r4(r["return"]),
                "cvar": _r4(r["CVaR"]),
                "mdd": _r4(r["MDD"]),
                "train_time_s": None if abs(float(r["train_time_s"])) < 1e-12 else _r4(r["train_time_s"]),
            }
        )
    return out


def _load_xlf_trend() -> list[dict[str, Any]]:
    """Load monthly normalized XLF close for visualization.

    This is used only for a qualitative market trend overlay in the demo.
    """
    try:
        df = yf.download(
            "XLF",
            start="2019-01-01",
            end="2026-01-31",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return []
        close_obj = df["Close"].dropna()
        if isinstance(close_obj, pd.DataFrame):
            if close_obj.shape[1] == 0:
                return []
            close = close_obj.iloc[:, 0]
        else:
            close = close_obj
        close = pd.to_numeric(close, errors="coerce").dropna()
        monthly = close.resample("ME").last().dropna()
        if monthly.empty:
            return []
        base = float(monthly.iloc[0])
        if abs(base) < 1e-12:
            return []
        out: list[dict[str, Any]] = []
        for ts, v in monthly.items():
            out.append(
                {
                    "date": ts.strftime("%Y-%m"),
                    "norm_close": _r4(float(v) / base),
                }
            )
        return out
    except Exception:
        # Keep demo build robust even if network is unavailable.
        return []


def _load_headline(data_dir: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(data_dir / "xlf_headline.csv")
    labels = {
        "q_learning": "Q-learning",
        "sg_sarsa": "SG-Sarsa",
        "nstep_sarsa": "n-step Sarsa",
    }
    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        out.append(
            {
                "algo": str(r["algo"]),
                "algo_label": labels.get(str(r["algo"]), str(r["algo"])),
                "promoted": _as_bool(r["promoted"]),
                "reward": _r4(r["reward"]),
                "return": _r4(r["return"]),
                "cvar": _r4(r["CVaR"]),
                "mdd": _r4(r["MDD"]),
                "train_time_s": _r4(r["train_time_s"]),
            }
        )
    return out


def main() -> None:
    demo_dir = Path(__file__).resolve().parent
    repo_root = demo_dir.parents[2]
    data_dir = repo_root / "video" / "assets" / "data"
    out_js = demo_dir / "data.js"

    payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "gate": "video/assets/data/xlf_sync_gate.csv",
            "sg_budget": "video/assets/data/sg_controlled_budget.csv",
            "with_no": "video/assets/data/xlf_with_vs_no.csv",
            "main_conclusion": "video/assets/data/xlf_main_conclusion.csv",
            "headline": "video/assets/data/xlf_headline.csv",
            "xlf_trend": "yfinance:XLF (2019-01 to 2026-01, monthly normalized close)",
        },
        "gate_rows": _load_gate_rows(data_dir),
        "sg_budget_rows": _load_sg_budget(data_dir),
        "with_no_rows": _load_with_no(data_dir),
        "main_conclusion_rows": _load_main_conclusion(data_dir),
        "headline_rows": _load_headline(data_dir),
        "xlf_trend": _load_xlf_trend(),
    }
    out_js.write_text("window.DEMO_DATA = " + json.dumps(payload, indent=2) + ";\n")
    print(f"Wrote {out_js}")


if __name__ == "__main__":
    main()
