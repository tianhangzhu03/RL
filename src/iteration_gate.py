"""Iteration gate for optimization rounds: promotion decision + proposal/instructor compliance checks.

This script codifies two workflows that should run after each optimization round:

1) Promotion gate:
   Compare a tuned candidate evaluation against the incumbent baseline and decide
   whether the version is allowed to "promote" (become the new mainline version).

2) Compliance audit:
   Verify that the experiment artifacts still align with the project proposal and
   instructor requirements (risk-sensitive reward, dataset size, algorithm comparison,
   training-time reporting, tuning evidence, etc.).

Typical usage:
  scripts/py.sh -m src.iteration_gate \
    --asset XLF \
    --config configs/main_xlf.yaml \
    --baseline-suite-root runs_xlf_baseline \
    --tuning-root runs/tuning_successive_xlf_v1 \
    --candidate-eval-root runs/tuned_best_xlf_v1_eval_ns \
    --algo nstep_sarsa \
    --tag with_cvar \
    --gate-config configs/xlf_promotion_gate.yaml \
    --out-dir report/xlf/iteration_gate_v1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from src.pipeline import prepare_dataset_split
from src.train import load_config


@dataclass
class PromotionThresholds:
    # Legacy gates (kept for backward compatibility).
    require_reward_gt_baseline: bool = True
    require_return_ge_baseline: bool = True
    max_cvar_regression: float = 0.002
    max_mdd_regression: float = 0.03
    require_positive_lambda_cvar: bool = True

    # Preferred gate (objective + tolerance + seed consistency).
    primary_metric: str | None = None  # "reward" or "return"
    primary_min_delta: float = 0.0
    secondary_metric: str | None = None  # optional, "reward" or "return"
    secondary_min_delta: float = 0.0
    seed_consistency_metric: str | None = None  # "reward" or "return"
    seed_min_nonworse_count: int = 0
    seed_delta_tolerance: float = 0.0


@dataclass
class ComplianceThresholds:
    min_total_samples: int = 3000
    min_eval_seeds: int = 5
    require_algorithms: tuple[str, ...] = ("q_learning", "nstep_sarsa")
    require_with_and_no_cvar_for_target_algo: bool = True
    require_validation_only_tuning_artifacts: bool = True
    require_tuning_artifacts: bool = True
    require_tabular_and_function_approx: bool = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promotion gate + proposal/instructor compliance audit")
    p.add_argument("--asset", required=True, help="Asset label for report output, e.g. XLF")
    p.add_argument("--config", required=True, help="Mainline config used for this optimization round")
    p.add_argument("--baseline-suite-root", required=True, help="Root produced by src.run_suite for incumbent baseline")
    p.add_argument("--candidate-eval-root", required=True, help="Root produced by src.run_best_configs for tuned candidate")
    p.add_argument("--tuning-root", required=True, help="Root produced by tuning script (e.g., src.tune_successive)")
    p.add_argument("--algo", default="nstep_sarsa", choices=["q_learning", "sg_sarsa", "nstep_sarsa"])
    p.add_argument("--tag", default="with_cvar", help="Tag used in suite/eval summaries (typically with_cvar)")
    p.add_argument("--gate-config", default=None, help="Optional YAML with promotion/compliance thresholds")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def to_markdown_fallback(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```csv\n" + df.to_csv(index=False) + "```"


def choose_latest_suite_summary(root: Path) -> Path:
    latest_ptr = root / "LATEST.txt"
    if latest_ptr.exists():
        p = Path(latest_ptr.read_text(encoding="utf-8").strip())
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        candidate = p / "suite_summary.csv"
        if candidate.exists():
            return candidate

    direct = root / "suite_summary.csv"
    if direct.exists():
        return direct

    candidates = sorted(root.glob("*/suite_summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"No suite_summary.csv found under {root}")
    return candidates[-1]


def choose_latest_tuning_run(tuning_root: Path, algo: str) -> Path | None:
    algo_root = tuning_root / algo
    latest_ptr = algo_root / "LATEST.txt"
    if latest_ptr.exists():
        p = Path(latest_ptr.read_text(encoding="utf-8").strip())
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists():
            return p
    # Prefer directories with best_config.yaml.
    candidates = sorted([d for d in algo_root.glob("20*") if (d / "best_config.yaml").exists()])
    if candidates:
        return candidates[-1]
    # Fallback to any timestamped run dir (e.g., failed attempts)
    candidates = sorted([d for d in algo_root.glob("20*") if d.is_dir()])
    return candidates[-1] if candidates else None


def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_summary_rows(summary_csv: Path, algo: str | None = None, tag: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if algo is not None and "algo" in df.columns:
        df = df[df["algo"] == algo].copy()
    if tag is not None and "tag" in df.columns:
        df = df[df["tag"] == tag].copy()
    return df.reset_index(drop=True)


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "test_total_reward": "reward",
        "test_cumulative_return": "return",
        "test_cvar": "cvar",
        "test_max_drawdown": "mdd",
        "train_seconds": "train_seconds",
    }
    out = df.rename(columns=rename_map).copy()
    required = ["reward", "return", "cvar", "mdd", "train_seconds"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")
    return out


def mean_metrics(df: pd.DataFrame) -> Dict[str, float]:
    out = normalize_metric_columns(df)
    return {
        "reward": float(out["reward"].astype(float).mean()),
        "return": float(out["return"].astype(float).mean()),
        "cvar": float(out["cvar"].astype(float).mean()),
        "mdd": float(out["mdd"].astype(float).mean()),
        "train_seconds": float(out["train_seconds"].astype(float).mean()),
        "n_runs": int(len(out)),
    }


def _metric_key_or_raise(name: str) -> str:
    allowed = {"reward", "return", "cvar", "mdd", "train_seconds"}
    if name not in allowed:
        raise ValueError(f"Unsupported metric key: {name}. Allowed: {sorted(allowed)}")
    return name


def compute_seed_deltas(
    baseline_rows: pd.DataFrame | None,
    candidate_rows: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if baseline_rows is None or candidate_rows is None:
        return None, {"available": False, "reason": "missing_rows"}

    b = normalize_metric_columns(baseline_rows)
    c = normalize_metric_columns(candidate_rows)
    if "seed" not in b.columns or "seed" not in c.columns:
        return None, {"available": False, "reason": "seed_column_missing"}

    b2 = b[["seed", "reward", "return", "cvar", "mdd", "train_seconds"]].copy()
    c2 = c[["seed", "reward", "return", "cvar", "mdd", "train_seconds"]].copy()
    merged = pd.merge(b2, c2, on="seed", how="inner", suffixes=("_baseline", "_candidate"))
    if merged.empty:
        return None, {"available": False, "reason": "no_seed_overlap"}

    for metric in ["reward", "return", "cvar", "mdd", "train_seconds"]:
        merged[f"delta_{metric}"] = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]

    info = {
        "available": True,
        "n_matched_seeds": int(len(merged)),
        "matched_seeds": [int(s) for s in merged["seed"].astype(int).tolist()],
    }
    return merged.sort_values("seed").reset_index(drop=True), info


def load_gate_thresholds(path: Path | None) -> tuple[PromotionThresholds, ComplianceThresholds]:
    promo = PromotionThresholds()
    comp = ComplianceThresholds()
    if path is None:
        return promo, comp
    cfg = read_yaml(path)
    promo_cfg = cfg.get("promotion", {})
    comp_cfg = cfg.get("compliance", {})
    for field in asdict(promo).keys():
        if field in promo_cfg:
            setattr(promo, field, promo_cfg[field])
    for field in asdict(comp).keys():
        if field in comp_cfg:
            value = comp_cfg[field]
            if field == "require_algorithms":
                value = tuple(value)
            setattr(comp, field, value)
    return promo, comp


def compute_promotion_decision(
    *,
    baseline_means: Mapping[str, float],
    candidate_means: Mapping[str, float],
    thresholds: PromotionThresholds,
    tuned_lambda_cvar: float | None,
    baseline_rows: pd.DataFrame | None = None,
    candidate_rows: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    delta_reward = float(candidate_means["reward"] - baseline_means["reward"])
    delta_return = float(candidate_means["return"] - baseline_means["return"])
    delta_cvar = float(candidate_means["cvar"] - baseline_means["cvar"])
    delta_mdd = float(candidate_means["mdd"] - baseline_means["mdd"])

    checks: dict[str, bool] = {}

    # Preferred objective-based gate, if configured.
    if thresholds.primary_metric is not None:
        primary_metric = _metric_key_or_raise(str(thresholds.primary_metric))
        primary_delta = float(candidate_means[primary_metric] - baseline_means[primary_metric])
        checks["primary_metric_within_tolerance"] = bool(primary_delta >= float(thresholds.primary_min_delta))
    else:
        checks["reward_gt_baseline"] = bool(delta_reward > 0.0) if thresholds.require_reward_gt_baseline else True
        checks["return_ge_baseline"] = bool(delta_return >= 0.0) if thresholds.require_return_ge_baseline else True

    if thresholds.secondary_metric is not None:
        secondary_metric = _metric_key_or_raise(str(thresholds.secondary_metric))
        secondary_delta = float(candidate_means[secondary_metric] - baseline_means[secondary_metric])
        checks["secondary_metric_within_tolerance"] = bool(secondary_delta >= float(thresholds.secondary_min_delta))

    checks["cvar_regression_within_tolerance"] = bool(delta_cvar <= float(thresholds.max_cvar_regression))
    # mdd is typically negative; more negative is worse, so allow only limited decrease.
    checks["mdd_regression_within_tolerance"] = bool(delta_mdd >= -float(thresholds.max_mdd_regression))
    checks["positive_lambda_cvar"] = (
        bool(tuned_lambda_cvar is not None and tuned_lambda_cvar > 0.0)
        if thresholds.require_positive_lambda_cvar
        else True
    )

    seed_deltas_df, seed_info = compute_seed_deltas(baseline_rows, candidate_rows)
    seed_consistency_detail: dict[str, Any] = dict(seed_info)
    if thresholds.seed_consistency_metric is not None and int(thresholds.seed_min_nonworse_count) > 0:
        metric = _metric_key_or_raise(str(thresholds.seed_consistency_metric))
        if seed_deltas_df is None:
            checks["seed_consistency"] = False
            seed_consistency_detail.update(
                {
                    "metric": metric,
                    "tolerance": float(thresholds.seed_delta_tolerance),
                    "min_nonworse_count": int(thresholds.seed_min_nonworse_count),
                    "passed": False,
                    "reason": seed_consistency_detail.get("reason", "seed_delta_unavailable"),
                }
            )
        else:
            col = f"delta_{metric}"
            tol = float(thresholds.seed_delta_tolerance)
            nonworse_mask = seed_deltas_df[col].astype(float) >= tol
            count = int(nonworse_mask.sum())
            passed = bool(count >= int(thresholds.seed_min_nonworse_count))
            checks["seed_consistency"] = passed
            seed_consistency_detail.update(
                {
                    "metric": metric,
                    "tolerance": tol,
                    "min_nonworse_count": int(thresholds.seed_min_nonworse_count),
                    "nonworse_count": count,
                    "passed": passed,
                }
            )

    promote = bool(all(checks.values()))
    return {
        "promote": promote,
        "checks": checks,
        "thresholds": asdict(thresholds),
        "baseline_means": dict(baseline_means),
        "candidate_means": dict(candidate_means),
        "deltas": {
            "reward": delta_reward,
            "return": delta_return,
            "cvar": delta_cvar,
            "mdd": delta_mdd,
            "train_seconds": float(candidate_means["train_seconds"] - baseline_means["train_seconds"]),
        },
        "tuned_lambda_cvar": None if tuned_lambda_cvar is None else float(tuned_lambda_cvar),
        "seed_consistency": seed_consistency_detail,
        "seed_deltas": [] if seed_deltas_df is None else seed_deltas_df.to_dict(orient="records"),
    }


def _status_row(check_id: str, passed: bool | None, detail: str, severity: str = "required") -> Dict[str, Any]:
    status = "pass" if passed is True else ("fail" if passed is False else "manual")
    return {"check_id": check_id, "status": status, "severity": severity, "detail": detail}


def _find_any_baseline_metrics_json(baseline_root: Path) -> Path | None:
    files = sorted(baseline_root.glob("**/baseline_metrics.json"))
    return files[0] if files else None


def build_compliance_report(
    *,
    config_path: Path,
    baseline_suite_summary: Path,
    candidate_eval_summary: Path,
    tuning_root: Path,
    latest_tuning_run: Path | None,
    algo: str,
    tag: str,
    thresholds: ComplianceThresholds,
) -> Dict[str, Any]:
    checks: list[dict[str, Any]] = []

    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    env_cfg = config.get("env", {})

    # Dataset checks (try exact split size; fallback to config range only)
    dataset_info: dict[str, Any] = {"symbol": dataset_cfg.get("symbol"), "size_source": "unknown"}
    try:
        split = prepare_dataset_split(config)
        total_rows = int(len(split.train) + len(split.val) + len(split.test))
        dataset_info.update(
            {
                "size_source": "prepare_dataset_split",
                "total_rows": total_rows,
                "train_rows": int(len(split.train)),
                "val_rows": int(len(split.val)),
                "test_rows": int(len(split.test)),
            }
        )
        checks.append(
            _status_row(
                "dataset_moderately_large",
                total_rows >= int(thresholds.min_total_samples),
                f"total_rows={total_rows}, threshold={int(thresholds.min_total_samples)}",
            )
        )
    except Exception as exc:  # pragma: no cover - network/data issues are environment-dependent
        dataset_info.update({"size_source": "error", "error": str(exc)})
        checks.append(
            _status_row(
                "dataset_moderately_large",
                None,
                f"Unable to auto-verify dataset size: {exc}",
            )
        )

    # Single asset / risk-sensitive config checks.
    symbol = dataset_cfg.get("symbol")
    checks.append(_status_row("single_asset_config", isinstance(symbol, str) and len(symbol) > 0, f"symbol={symbol!r}"))
    checks.append(
        _status_row(
            "risk_sensitive_reward_config_present",
            all(k in env_cfg for k in ["risk_window", "risk_alpha", "lambda_cvar", "lambda_vol"]),
            (
                "env has risk_window/risk_alpha/lambda_cvar/lambda_vol="
                f"{all(k in env_cfg for k in ['risk_window', 'risk_alpha', 'lambda_cvar', 'lambda_vol'])}"
            ),
        )
    )
    checks.append(
        _status_row(
            "cvar_penalty_enabled_in_main_config",
            float(env_cfg.get("lambda_cvar", 0.0)) > 0.0,
            f"env.lambda_cvar={float(env_cfg.get('lambda_cvar', 0.0))}",
        )
    )

    # Baseline suite algorithm comparison checks.
    baseline_df = pd.read_csv(baseline_suite_summary)
    baseline_algos = sorted(set(baseline_df["algo"].astype(str))) if "algo" in baseline_df.columns else []
    checks.append(
        _status_row(
            "at_least_two_algorithms_compared",
            len(baseline_algos) >= 2,
            f"baseline suite algos={baseline_algos}",
        )
    )
    if thresholds.require_algorithms:
        required_algos = list(thresholds.require_algorithms)
        missing_algos = [a for a in required_algos if a not in baseline_algos]
        checks.append(
            _status_row(
                "required_algorithms_present",
                len(missing_algos) == 0,
                f"required={required_algos}, missing={missing_algos}, present={baseline_algos}",
            )
        )
    if thresholds.require_tabular_and_function_approx:
        has_tabular = "q_learning" in baseline_algos
        has_fa = any(a in baseline_algos for a in ["sg_sarsa", "nstep_sarsa"])
        checks.append(
            _status_row(
                "tabular_plus_function_approx_present",
                bool(has_tabular and has_fa),
                f"has_q_learning={has_tabular}, has_function_approx={has_fa}, algos={baseline_algos}",
            )
        )

    # Training-time reporting / seed counts / risk metrics availability.
    cand_df = pd.read_csv(candidate_eval_summary)
    checks.append(
        _status_row(
            "training_time_recorded",
            ("train_seconds" in baseline_df.columns) and ("train_seconds" in cand_df.columns),
            (
                f"baseline_has_train_seconds={'train_seconds' in baseline_df.columns}, "
                f"candidate_has_train_seconds={'train_seconds' in cand_df.columns}"
            ),
        )
    )
    checks.append(
        _status_row(
            "risk_metrics_reported_in_outputs",
            ("test_cvar" in baseline_df.columns)
            and ("test_cvar" in cand_df.columns)
            and ("test_max_drawdown" in baseline_df.columns)
            and ("test_max_drawdown" in cand_df.columns),
            "Checked test_cvar/test_max_drawdown columns in baseline and candidate summaries",
        )
    )

    cand_algo_rows = extract_summary_rows(candidate_eval_summary, algo=algo, tag=tag if "tag" in cand_df.columns else None)
    checks.append(
        _status_row(
            "candidate_eval_seed_count",
            len(cand_algo_rows) >= int(thresholds.min_eval_seeds),
            f"{algo}/{tag} rows in candidate eval={len(cand_algo_rows)}, threshold={int(thresholds.min_eval_seeds)}",
        )
    )

    if thresholds.require_with_and_no_cvar_for_target_algo:
        has_with = not extract_summary_rows(baseline_suite_summary, algo=algo, tag="with_cvar").empty
        has_no = not extract_summary_rows(baseline_suite_summary, algo=algo, tag="no_cvar").empty
        checks.append(
            _status_row(
                "with_vs_no_cvar_comparison_available_for_target_algo",
                bool(has_with and has_no),
                f"{algo}: has_with_cvar={has_with}, has_no_cvar={has_no}",
            )
        )
        # Directional expectation check (professor suggestion: risk-sensitive reward should affect risk metrics)
        if has_with and has_no:
            with_mean = mean_metrics(extract_summary_rows(baseline_suite_summary, algo=algo, tag="with_cvar"))
            no_mean = mean_metrics(extract_summary_rows(baseline_suite_summary, algo=algo, tag="no_cvar"))
            direction_ok = (with_mean["cvar"] <= no_mean["cvar"]) and (with_mean["mdd"] >= no_mean["mdd"])
            checks.append(
                _status_row(
                    "risk_sensitive_direction_check_on_baseline_suite",
                    bool(direction_ok),
                    (
                        f"with_cvar(cvar={with_mean['cvar']:.6f}, mdd={with_mean['mdd']:.6f}) vs "
                        f"no_cvar(cvar={no_mean['cvar']:.6f}, mdd={no_mean['mdd']:.6f})"
                    ),
                    severity="recommended",
                )
            )

    # Baseline metrics file should include VaR/CVaR (professor comment)
    metrics_file = _find_any_baseline_metrics_json(Path(baseline_suite_summary).parent)
    if metrics_file is not None:
        try:
            data = json.load(open(metrics_file, "r", encoding="utf-8"))
            buy_hold = data["buy_hold"] if "buy_hold" in data else data
            has_var_cvar = ("var" in buy_hold) and ("cvar" in buy_hold)
            checks.append(
                _status_row(
                    "var_cvar_present_in_baseline_metrics",
                    has_var_cvar,
                    f"baseline_metrics file={metrics_file}, keys include var/cvar={has_var_cvar}",
                )
            )
        except Exception as exc:
            checks.append(_status_row("var_cvar_present_in_baseline_metrics", None, f"Unable to parse baseline_metrics: {exc}"))
    else:
        checks.append(_status_row("var_cvar_present_in_baseline_metrics", False, "No baseline_metrics.json found in baseline suite"))

    # Tuning evidence checks.
    tuning_info: dict[str, Any] = {"tuning_root": str(tuning_root), "latest_tuning_run": None}
    if latest_tuning_run is None:
        checks.append(_status_row("tuning_run_exists", False, f"No tuning run found for algo={algo} under {tuning_root}"))
    else:
        tuning_info["latest_tuning_run"] = str(latest_tuning_run)
        required_files = [
            latest_tuning_run / "selection_summary.json",
            latest_tuning_run / "best_trial.json",
            latest_tuning_run / "best_config.yaml",
            latest_tuning_run / "phase_01" / "phase_log.csv",
        ]
        has_required = all(p.exists() for p in required_files)
        checks.append(
            _status_row(
                "tuning_artifacts_present",
                has_required if thresholds.require_tuning_artifacts else True,
                "required="
                + ", ".join(f"{p.name}:{p.exists()}" for p in required_files),
            )
        )
        if thresholds.require_validation_only_tuning_artifacts:
            test_metrics_in_tuning = list(latest_tuning_run.glob("phase_*/trial_*/seed_*/test_metrics.json"))
            checks.append(
                _status_row(
                    "validation_only_tuning_evidence",
                    len(test_metrics_in_tuning) == 0,
                    f"test_metrics.json files under tuning run={len(test_metrics_in_tuning)}",
                )
            )

    # Manual checklist for instructor prompts that require written analysis.
    for check_id, detail in [
        (
            "manual_explain_parameter_adjustment_process",
            "Document how parameters were adjusted (coarse->refined or budgeted tuning) with validation-only selection rationale.",
        ),
        (
            "manual_identify_important_parameters",
            "Explain which parameters mattered most (e.g., alpha, n, epsilon_decay, lambda_cvar, lambda_vol, opportunity_cost).",
        ),
        (
            "manual_challenges_and_mitigation",
            "Describe challenges (non-stationarity, seed variance, buy&hold attractor, validation-test mismatch) and mitigation attempts.",
        ),
        (
            "manual_future_work",
            "Describe future improvements (walk-forward, regime splits, stronger features, richer action spaces, etc.).",
        ),
    ]:
        checks.append(_status_row(check_id, None, detail, severity="manual_required"))

    passed_required = all(row["status"] != "fail" for row in checks if row["severity"] == "required")
    failed_required = [row["check_id"] for row in checks if row["severity"] == "required" and row["status"] == "fail"]

    return {
        "passed_required_checks": bool(passed_required),
        "failed_required_checks": failed_required,
        "thresholds": {
            "compliance": {
                **asdict(thresholds),
                "require_algorithms": list(thresholds.require_algorithms),
            }
        },
        "dataset_info": dataset_info,
        "checks": checks,
        "artifacts": {
            "config": str(config_path),
            "baseline_suite_summary": str(baseline_suite_summary),
            "candidate_eval_summary": str(candidate_eval_summary),
            **tuning_info,
        },
    }


def write_reports(
    *,
    out_dir: Path,
    asset: str,
    promotion: Dict[str, Any],
    compliance: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON artifacts
    (out_dir / "promotion_decision.json").write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    (out_dir / "proposal_teacher_compliance.json").write_text(json.dumps(compliance, indent=2), encoding="utf-8")

    # Promotion markdown
    promo_rows = []
    for metric in ["reward", "return", "cvar", "mdd", "train_seconds"]:
        promo_rows.append(
            {
                "metric": metric,
                "baseline_mean": promotion["baseline_means"][metric],
                "candidate_mean": promotion["candidate_means"][metric],
                "delta": promotion["deltas"][metric],
            }
        )
    promo_df = pd.DataFrame(promo_rows)
    checks_df = pd.DataFrame(
        [{"check": k, "pass": bool(v)} for k, v in promotion["checks"].items()]
    )
    seed_consistency = promotion.get("seed_consistency", {})
    seed_deltas_records = promotion.get("seed_deltas", [])
    if seed_deltas_records:
        seed_df = pd.DataFrame(seed_deltas_records)
        seed_df.to_csv(out_dir / "promotion_seed_deltas.csv", index=False)
        seed_section = "\n".join(
            [
                "## Seed Consistency",
                "",
                "```json",
                json.dumps(seed_consistency, indent=2),
                "```",
                "",
                to_markdown_fallback(seed_df),
                "",
            ]
        )
    else:
        seed_section = "\n".join(
            [
                "## Seed Consistency",
                "",
                "```json",
                json.dumps(seed_consistency, indent=2),
                "```",
                "",
            ]
        )
    promo_md = "\n".join(
        [
            f"# {asset} Iteration Promotion Decision",
            "",
            f"- `promote`: `{promotion['promote']}`",
            f"- `tuned_lambda_cvar`: `{promotion.get('tuned_lambda_cvar')}`",
            "",
            "## Metrics",
            "",
            to_markdown_fallback(promo_df),
            "",
            "## Checks",
            "",
            to_markdown_fallback(checks_df),
            "",
            seed_section,
            "## Thresholds",
            "",
            "```yaml",
            yaml.safe_dump({"promotion": promotion["thresholds"]}, sort_keys=False).strip(),
            "```",
            "",
        ]
    )
    (out_dir / "promotion_decision.md").write_text(promo_md, encoding="utf-8")

    # Compliance markdown
    checks_df2 = pd.DataFrame(compliance["checks"])
    compliance_md = "\n".join(
        [
            f"# {asset} Proposal / Instructor Compliance Audit",
            "",
            f"- `passed_required_checks`: `{compliance['passed_required_checks']}`",
            f"- `failed_required_checks`: `{compliance['failed_required_checks']}`",
            "",
            "## Dataset Info",
            "",
            "```json",
            json.dumps(compliance.get("dataset_info", {}), indent=2),
            "```",
            "",
            "## Checks",
            "",
            to_markdown_fallback(checks_df2),
            "",
        ]
    )
    (out_dir / "proposal_teacher_compliance.md").write_text(compliance_md, encoding="utf-8")

    # Combined summary
    combined = "\n".join(
        [
            f"# {asset} Iteration Gate Summary",
            "",
            f"- Promotion decision: `{promotion['promote']}`",
            f"- Proposal / instructor required checks: `{compliance['passed_required_checks']}`",
            f"- Promotion details: `promotion_decision.md`",
            f"- Compliance details: `proposal_teacher_compliance.md`",
            "",
        ]
    )
    (out_dir / "README.md").write_text(combined, encoding="utf-8")


def main() -> None:
    args = parse_args()
    asset = args.asset.upper()

    config_path = Path(args.config)
    baseline_root = Path(args.baseline_suite_root)
    candidate_eval_root = Path(args.candidate_eval_root)
    tuning_root = Path(args.tuning_root)
    out_dir = Path(args.out_dir)

    promo_thresholds, compliance_thresholds = load_gate_thresholds(Path(args.gate_config) if args.gate_config else None)

    baseline_summary = choose_latest_suite_summary(baseline_root)
    candidate_summary = choose_latest_suite_summary(candidate_eval_root)
    latest_tuning_run = choose_latest_tuning_run(tuning_root, args.algo)

    baseline_rows = extract_summary_rows(baseline_summary, algo=args.algo, tag=args.tag)
    if baseline_rows.empty:
        raise ValueError(f"No baseline rows for algo={args.algo!r}, tag={args.tag!r} in {baseline_summary}")

    candidate_rows = extract_summary_rows(candidate_summary, algo=args.algo, tag=args.tag if "tag" in pd.read_csv(candidate_summary, nrows=1).columns else None)
    if candidate_rows.empty:
        raise ValueError(f"No candidate eval rows for algo={args.algo!r}, tag={args.tag!r} in {candidate_summary}")

    tuned_lambda_cvar = None
    if latest_tuning_run is not None and (latest_tuning_run / "best_config.yaml").exists():
        best_cfg = read_yaml(latest_tuning_run / "best_config.yaml")
        try:
            tuned_lambda_cvar = float(best_cfg["env"].get("lambda_cvar", 0.0))
        except Exception:
            tuned_lambda_cvar = None

    promotion = compute_promotion_decision(
        baseline_means=mean_metrics(baseline_rows),
        candidate_means=mean_metrics(candidate_rows),
        thresholds=promo_thresholds,
        tuned_lambda_cvar=tuned_lambda_cvar,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
    )
    compliance = build_compliance_report(
        config_path=config_path,
        baseline_suite_summary=baseline_summary,
        candidate_eval_summary=candidate_summary,
        tuning_root=tuning_root,
        latest_tuning_run=latest_tuning_run,
        algo=args.algo,
        tag=args.tag,
        thresholds=compliance_thresholds,
    )

    write_reports(out_dir=out_dir, asset=asset, promotion=promotion, compliance=compliance)
    print(f"Saved iteration gate reports to: {out_dir}")
    print(f"Promotion decision: {promotion['promote']}")
    print(f"Required compliance checks pass: {compliance['passed_required_checks']}")


if __name__ == "__main__":
    main()
