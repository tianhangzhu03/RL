from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from IPython.display import HTML, display


def _short_model_label(label: str) -> str:
    s = str(label).strip()
    s_low = s.lower().replace("_", " ")
    if "q learning" in s_low or "q-learning" in s_low:
        return "Q-learning (CVaR cfg)"
    if "nstep" in s_low or "n-step" in s_low:
        return "n-step Sarsa (CVaR tuned)"
    if "baseline" in s_low:
        return "Baseline (CVaR)"
    return s.replace("_", "-")


def r4(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def apply_table_style() -> None:
    css = """
<style>
sup.cite {
  font-size: 0.72em;
  vertical-align: super;
  line-height: 0;
}
mjx-container[display="true"] {
  margin: 0.32em 0 !important;
}
.MathJax_Display {
  margin: 0.32em 0 !important;
}
table.dataframe {
  width: 100% !important;
  max-width: 100% !important;
  table-layout: auto !important;
  border-collapse: collapse;
  font-size: 9pt;
  margin-left: auto !important;
  margin-right: auto !important;
}
table.dataframe th, table.dataframe td {
  border: 1px solid #d9dde3;
  padding: 4px 6px;
  text-align: center;
  vertical-align: middle;
  white-space: normal !important;
  word-break: break-word;
}
table.dataframe thead th {
  background: #f3f6fa;
  font-weight: 600;
}
.table-block {
  display: block;
  width: 92%;
  max-width: 92%;
  margin: 0.06em 0 0.22em 0;
  text-align: initial;
}
.table-wrap {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 100%;
  margin: 0;
}
.jp-OutputArea,
.output_area {
  padding-left: 0 !important;
  margin-left: 0 !important;
}
.jp-OutputArea-child,
.output_wrapper,
.output_subarea {
  margin-left: 0 !important;
  width: 100% !important;
}
.table-block table.dataframe {
  margin-left: auto !important;
  margin-right: auto !important;
}
.table-title {
  text-align: center;
  font-weight: 600;
  margin: 0 0 0.08em 0;
}
.compact-table {
  width: 82%;
  max-width: 82%;
}
.compact-table table.dataframe {
  font-size: 7.8pt !important;
  table-layout: fixed !important;
}
.compact-table table.dataframe th,
.compact-table table.dataframe td {
  padding: 3px 4px;
  overflow-wrap: anywhere;
}
@media print {
  .jp-Cell, .cell, .jp-OutputArea, .jp-OutputArea-child, .output, .output_area {
    break-inside: auto !important;
    page-break-inside: auto !important;
  }
  table.dataframe, .jp-RenderedHTMLCommon table, table {
    break-inside: auto !important;
    page-break-inside: auto !important;
  }
  table.dataframe thead, .jp-RenderedHTMLCommon thead {
    display: table-header-group !important;
  }
  table.dataframe tr, .jp-RenderedHTMLCommon tr {
    break-inside: avoid !important;
    page-break-inside: avoid !important;
  }
}
</style>
"""
    display(HTML(css))


def show_table(
    title: str,
    df: pd.DataFrame,
    max_rows: int = 20,
    compact: bool = False,
    width_pct: int | None = None,
    col_widths: dict[str, str] | None = None,
) -> None:
    out = df.head(max_rows).copy() if len(df) > max_rows else df.copy()
    out = _format_for_display(out)
    table_html = out.to_html(index=False, classes="report-table", border=0, justify="center")
    table_html = table_html.replace(
        "<table ",
        '<table style="margin-left:auto;margin-right:auto;width:100%;table-layout:auto;" ',
        1,
    )
    if col_widths:
        cols = list(out.columns)
        colgroup = "<colgroup>" + "".join(
            f'<col style="width:{col_widths.get(c, "auto")};">' for c in cols
        ) + "</colgroup>"
        pos = table_html.find(">")
        if pos != -1:
            table_html = table_html[: pos + 1] + colgroup + table_html[pos + 1 :]
    block_class = "table-block compact-table" if compact else "table-block"
    default_width = 82 if compact else 92
    width = width_pct if width_pct is not None else default_width
    block_html = (
        f'<div class="table-wrap" style="width:100%;display:flex;justify-content:center;">'
        f'<div class="{block_class}" style="width:{width}%;max-width:{width}%;margin:0;">'
        f'<div class="table-title" style="text-align:center;">{title}</div>'
        f'{table_html}'
        f"</div></div>"
    )
    display(HTML(block_html))


def _format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        is_train_col = "train" in str(col).lower()

        def _trim(x: Any) -> Any:
            if isinstance(x, bool):
                return "Yes" if x else "No"
            if pd.isna(x):
                return "/" if is_train_col else ""
            if isinstance(x, int):
                if is_train_col and x == 0:
                    return "/"
                return x
            if isinstance(x, float):
                if is_train_col and abs(x) < 1e-12:
                    return "/"
                s = f"{x:.4f}".rstrip("0").rstrip(".")
                return "0" if s == "-0" else s
            return x

        out[col] = out[col].map(_trim)
    return out


def R(rel: str) -> str:
    p1 = Path(rel)
    p2 = Path("../..") / rel
    if p1.exists():
        return str(p1)
    if p2.exists():
        return str(p2)
    raise FileNotFoundError(f"Cannot find path: {rel}")


def _read_csv_with_fallback(paths: list[str]) -> pd.DataFrame:
    last_err: Exception | None = None
    for path in paths:
        try:
            return pd.read_csv(R(path))
        except Exception as err:
            last_err = err
    raise FileNotFoundError(f"None of the CSV paths exist: {paths}. Last error: {last_err}")


def load_report_tables() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    raw: Dict[str, pd.DataFrame] = {
        "t_main": _read_csv_with_fallback(
            [
                "report/final/data/xlf_main_conclusion.csv",
                "report/xlf/core_conclusion/table_xlf_main_conclusion.csv",
            ]
        ),
        "t_sync": _read_csv_with_fallback(
            [
                "report/final/data/xlf_sync_gate.csv",
                "report/xlf/core_conclusion/table_xlf_three_methods_sync_gate.csv",
            ]
        ),
        "t_head": _read_csv_with_fallback(
            [
                "report/final/data/xlf_headline.csv",
                "report/xlf/core_conclusion/table_xlf_three_methods_headline.csv",
            ]
        ),
        "t_with_no": _read_csv_with_fallback(
            [
                "report/final/data/xlf_with_vs_no.csv",
                "report/xlf/core_conclusion/table_xlf_with_vs_no_deltas.csv",
            ]
        ),
        "t_sg_budget": _read_csv_with_fallback(
            [
                "report/final/data/sg_controlled_budget.csv",
                "report/xlf/core_conclusion/table_sg_controlled_learning_budget.csv",
            ]
        ),
        "t_sg_rounds": _read_csv_with_fallback(
            [
                "report/final/data/sg_rounds_tradeoff.csv",
                "report/xlf/core_conclusion/table_sg_rounds_tradeoff.csv",
            ]
        ),
        "t_qqq": _read_csv_with_fallback(
            [
                "report/final/data/qqq_main_conclusion.csv",
                "report/qqq/core_conclusion/table_qqq_main_conclusion.csv",
            ]
        ),
    }

    tables: Dict[str, pd.DataFrame] = {
        "tbl_main": r4(
            raw["t_main"][["model", "reward", "return", "CVaR", "MDD", "train_time_s"]],
            ["reward", "return", "CVaR", "MDD", "train_time_s"],
        ),
        "tbl_head": r4(
            raw["t_head"][["algo", "promoted", "reward", "return", "CVaR", "MDD", "train_time_s"]],
            ["reward", "return", "CVaR", "MDD", "train_time_s"],
        ),
        "tbl_sync": r4(
            raw["t_sync"][
                [
                    "algo",
                    "promoted",
                    "reward_delta",
                    "return_delta",
                    "CVaR_delta",
                    "MDD_delta",
                ]
            ],
            ["reward_delta", "return_delta", "CVaR_delta", "MDD_delta"],
        ),
        "tbl_with_no": r4(
            raw["t_with_no"]
            .rename(
                columns={
                    "delta_reward_with_minus_no": "reward_delta",
                    "delta_return_with_minus_no": "return_delta",
                    "delta_CVaR_with_minus_no": "cvar_delta",
                    "delta_MDD_with_minus_no": "mdd_delta",
                }
            )[
                ["algo", "reward_delta", "return_delta", "cvar_delta", "mdd_delta"]
            ],
            ["reward_delta", "return_delta", "cvar_delta", "mdd_delta"],
        ),
        "tbl_sg": r4(
            raw["t_sg_budget"]
            .rename(
                columns={
                    "tuning_phase3_episodes": "episodes",
                    "return_delta_vs_baseline": "return_delta",
                    "reward_delta_vs_baseline": "reward_delta",
                    "cvar_delta_vs_baseline": "cvar_delta",
                    "mdd_delta_vs_baseline": "mdd_delta",
                    "train_seconds": "train_sec",
                }
            )[
                [
                    "experiment",
                    "promoted",
                    "episodes",
                    "alpha",
                    "epsilon_decay",
                    "epsilon_min",
                    "return_delta",
                    "reward_delta",
                    "cvar_delta",
                    "mdd_delta",
                    "train_sec",
                ]
            ],
            [
                "alpha",
                "epsilon_decay",
                "epsilon_min",
                "return_delta",
                "reward_delta",
                "cvar_delta",
                "mdd_delta",
                "train_sec",
            ],
        ),
        "tbl_qqq": r4(
            raw["t_qqq"][["model", "reward", "return", "CVaR", "MDD", "train_time_s"]],
            ["reward", "return", "CVaR", "MDD", "train_time_s"],
        ),
        "tbl_sg_rounds": r4(
            raw["t_sg_rounds"]
            .rename(
                columns={
                    "return_delta_vs_baseline": "return_delta",
                    "reward_delta_vs_baseline": "reward_delta",
                    "cvar_delta_vs_baseline": "cvar_delta",
                    "mdd_delta_vs_baseline": "mdd_delta",
                }
            )[
                [
                    "round",
                    "promoted",
                    "return_delta",
                    "reward_delta",
                    "cvar_delta",
                    "mdd_delta",
                    "seed_consistency_passed",
                ]
            ],
            ["return_delta", "reward_delta", "cvar_delta", "mdd_delta"],
        ),
    }
    # Shorten verbose model labels for cleaner table layout.
    for key in ["tbl_main", "tbl_qqq"]:
        if "model" in tables[key].columns:
            tables[key]["model"] = tables[key]["model"].astype(str).map(_short_model_label)
    return raw, tables
