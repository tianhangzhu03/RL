"""Progressive budgeted tuning (successive halving) for result-oriented RL experiments.

This script samples candidates once, then evaluates them across multiple phases with
increasing episode budgets and (optionally) more tuning seeds. Ranking is lexicographic:

1) Risk feasibility under constraints / guardrails
2) Risk feasible-rate (higher better)
3) Risk violation magnitude (lower better)
4) Profit-floor feasibility (reward/return floors)
5) Validation primary objective (reward/return, configurable)
6) Validation secondary objective (the remaining one)
7) Variance proxies (lower better)

Usage:
  scripts/py.sh -m src.tune_successive --algo nstep_sarsa --space focused_v3 --trials 24
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.pipeline import prepare_dataset_split
from src.train import evaluate_baselines, load_config, run_training, to_env_config
from src.env import TradingEnv
from src.tune_budget import (
    apply_params,
    build_search_space,
    compute_constraint_status,
    compute_effective_constraint_bounds,
    parse_tuning_seeds,
    sample_trials,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Successive-halving tuning with lexicographic feasible ranking")
    parser.add_argument("--algo", required=True, choices=["q_learning", "sg_sarsa", "nstep_sarsa"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--space",
        choices=[
            "default",
            "focused",
            "focused_v2",
            "focused_v3",
            "design_ablation",
            "design_ablation_risk",
            "xlf_sync_safe",
            "xlf_sg_risk_push",
            "xlf_sg_balanced_r3",
            "xlf_sg_boundary_r4",
            "xlf_sg_learning_only",
            "xlf_sg_learning_only_guard",
        ],
        default="focused_v3",
    )
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=11, help="Search sampling seed")
    parser.add_argument("--phase-episodes", default="40,80,120")
    parser.add_argument(
        "--phase-keep",
        default="6,2,1",
        help="Number of survivors after each phase, same length as phase-episodes.",
    )
    parser.add_argument(
        "--phase-tuning-seeds",
        default="11,22;11,22;11,22,33",
        help="Semicolon-separated per-phase tuning seeds, e.g. '11,22;11,22;11,22,33'",
    )
    parser.add_argument(
        "--selection-priority",
        choices=["return_first", "reward_first"],
        default="return_first",
        help="Primary ranking objective in feasible region.",
    )
    parser.add_argument(
        "--risk-tiebreak",
        choices=["default", "strong"],
        default="default",
        help=(
            "Risk tie-break strength inside feasible region. "
            "'strong' prioritizes lower CVaR / better max drawdown earlier in ranking."
        ),
    )
    parser.add_argument("--constraint-max-cvar", type=float, default=0.065)
    parser.add_argument("--constraint-min-max-drawdown", type=float, default=-0.60)
    parser.add_argument("--constraint-weight-cvar", type=float, default=1.0)
    parser.add_argument("--constraint-weight-mdd", type=float, default=1.0)
    parser.add_argument("--constraint-min-feasible-rate", type=float, default=0.8)
    parser.add_argument("--incumbent-cvar", type=float, default=None)
    parser.add_argument("--incumbent-max-drawdown", type=float, default=None)
    parser.add_argument("--incumbent-cvar-slack", type=float, default=None)
    parser.add_argument("--incumbent-mdd-slack", type=float, default=None)
    parser.add_argument(
        "--incumbent-reward",
        type=float,
        default=None,
        help="Optional incumbent validation reward anchor used to derive a reward floor.",
    )
    parser.add_argument(
        "--incumbent-return",
        type=float,
        default=None,
        help="Optional incumbent validation cumulative-return anchor used to derive a return floor.",
    )
    parser.add_argument(
        "--reward-floor-ratio",
        type=float,
        default=None,
        help="If set with --incumbent-reward, effective reward floor = incumbent_reward * ratio.",
    )
    parser.add_argument(
        "--return-floor-ratio",
        type=float,
        default=None,
        help="If set with --incumbent-return, effective return floor = incumbent_return * ratio.",
    )
    parser.add_argument("--composite-return-weight", type=float, default=0.20)
    parser.add_argument("--composite-reward-std-weight", type=float, default=0.08)
    parser.add_argument("--composite-return-std-weight", type=float, default=0.02)
    parser.add_argument(
        "--rank-excess-vs-buyhold",
        action="store_true",
        help=(
            "Compute validation buy-hold baseline under each candidate env "
            "and rank feasible candidates by excess reward/return before absolute reward/return."
        ),
    )
    parser.add_argument(
        "--min-excess-reward-floor",
        type=float,
        default=None,
        help="Optional validation excess-reward floor vs buy_hold for feasibility.",
    )
    parser.add_argument(
        "--min-excess-return-floor",
        type=float,
        default=None,
        help="Optional validation excess-return floor vs buy_hold for feasibility.",
    )
    parser.add_argument(
        "--min-reward-floor",
        type=float,
        default=None,
        help="Optional validation reward floor for feasibility in each phase.",
    )
    parser.add_argument(
        "--min-return-floor",
        type=float,
        default=None,
        help="Optional validation cumulative return floor for feasibility in each phase.",
    )
    parser.add_argument("--output-root", default="runs/tuning_successive")
    parser.add_argument(
        "--require-final-feasible",
        action="store_true",
        help="Fail the tuning run if the final phase has no fully feasible candidate.",
    )
    parser.add_argument(
        "--include-base-candidate",
        action="store_true",
        help=(
            "Inject one baseline-projected candidate into the sampled set. "
            "Useful for boundary-focused rounds to avoid drifting away from incumbent behavior."
        ),
    )
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_phase_seed_groups(text: str) -> List[List[int]]:
    groups: List[List[int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        groups.append(parse_tuning_seeds(chunk, fallback_seed=11))
    return groups


def validate_phase_specs(args: argparse.Namespace) -> tuple[List[int], List[int], List[List[int]]]:
    phase_episodes = parse_int_list(args.phase_episodes)
    phase_keep = parse_int_list(args.phase_keep)
    phase_seed_groups = parse_phase_seed_groups(args.phase_tuning_seeds)

    if not phase_episodes or not phase_keep or not phase_seed_groups:
        raise ValueError("Phase definitions cannot be empty")
    if not (len(phase_episodes) == len(phase_keep) == len(phase_seed_groups)):
        raise ValueError("phase-episodes, phase-keep, and phase-tuning-seeds must have the same length")
    if phase_keep[-1] != 1:
        raise ValueError("Final phase keep count should be 1 to select a single best config")
    for i in range(1, len(phase_keep)):
        if phase_keep[i] > phase_keep[i - 1]:
            raise ValueError("phase-keep must be non-increasing")
    if phase_keep[0] > args.trials:
        raise ValueError("First phase keep count cannot exceed --trials")
    if (args.reward_floor_ratio is None) ^ (args.incumbent_reward is None):
        raise ValueError("Use --reward-floor-ratio together with --incumbent-reward")
    if (args.return_floor_ratio is None) ^ (args.incumbent_return is None):
        raise ValueError("Use --return-floor-ratio together with --incumbent-return")
    return phase_episodes, phase_keep, phase_seed_groups


def compute_effective_profit_floors(args: argparse.Namespace) -> tuple[float | None, float | None]:
    reward_floor = float(args.min_reward_floor) if args.min_reward_floor is not None else None
    return_floor = float(args.min_return_floor) if args.min_return_floor is not None else None

    if args.incumbent_reward is not None and args.reward_floor_ratio is not None:
        derived = float(args.incumbent_reward) * float(args.reward_floor_ratio)
        reward_floor = max(reward_floor, derived) if reward_floor is not None else derived

    if args.incumbent_return is not None and args.return_floor_ratio is not None:
        derived = float(args.incumbent_return) * float(args.return_floor_ratio)
        return_floor = max(return_floor, derived) if return_floor is not None else derived

    return reward_floor, return_floor


def _to_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _get_nested_value(cfg: Mapping[str, Any], key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


def _pick_closest_choice(base_value: Any, choices: Sequence[Any]) -> Any:
    if not choices:
        raise ValueError("choices cannot be empty")

    base_scalar = _to_scalar(base_value)
    candidate_values = [_to_scalar(c) for c in choices]

    for candidate in candidate_values:
        if _is_number(base_scalar) and _is_number(candidate):
            if np.isclose(float(candidate), float(base_scalar), rtol=0.0, atol=1e-12):
                return candidate
        elif candidate == base_scalar:
            return candidate

    if _is_number(base_scalar) and all(_is_number(c) for c in candidate_values):
        return min(candidate_values, key=lambda c: abs(float(c) - float(base_scalar)))

    return candidate_values[0]


def build_base_candidate(search_space: Mapping[str, Sequence[Any]], base_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in sorted(search_space.keys()):
        choices = search_space[key]
        try:
            base_value = _get_nested_value(base_cfg, key)
            params[key] = _pick_closest_choice(base_value, choices)
        except Exception:
            params[key] = _to_scalar(choices[0])
    return params


def _candidate_signature(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((k, _to_scalar(params[k])) for k in sorted(params.keys()))


def inject_base_candidate(
    candidates: Sequence[Dict[str, Any]],
    base_candidate: Dict[str, Any],
    trials: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()

    for params in [base_candidate, *candidates]:
        sig = _candidate_signature(params)
        if sig in seen:
            continue
        seen.add(sig)
        out.append({k: _to_scalar(v) for k, v in params.items()})
        if len(out) >= trials:
            break

    return out


def evaluate_candidate(
    *,
    algo: str,
    base_cfg: Dict[str, Any],
    params: Dict[str, Any],
    split,
    phase_idx: int,
    trial_rank: int,
    episodes: int,
    tuning_seeds: Sequence[int],
    out_dir: Path,
    args: argparse.Namespace,
    effective_constraint_max_cvar: float,
    effective_constraint_min_mdd: float,
    effective_reward_floor: float | None,
    effective_return_floor: float | None,
) -> Dict[str, Any]:
    trial_cfg = apply_params(base_cfg, params, episodes_override=episodes)
    trial_dir = out_dir / f"trial_{trial_rank:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    with open(trial_dir / "trial_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(trial_cfg, f)

    seed_rows: List[Dict[str, float]] = []
    c_rows: List[Dict[str, float]] = []

    env_cfg = to_env_config(trial_cfg["env"])

    for seed in tuning_seeds:
        seed_dir = trial_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics = run_training(
            algo=algo,
            config=trial_cfg,
            train_df=split.train,
            val_df=split.val,
            test_df=split.test,
            seed=int(seed),
            output_dir=seed_dir,
            evaluate_test=False,
        )
        val = metrics["val"]
        val_env = TradingEnv(split.val, env_cfg)
        val_baselines = evaluate_baselines(val_env, seed=int(seed))
        val_buyhold = val_baselines["buy_hold"]
        seed_rows.append(
            {
                "seed": int(seed),
                "reward": float(val.get("total_reward", np.nan)),
                "ret": float(val.get("cumulative_return", np.nan)),
                "cvar": float(val.get("cvar", np.nan)),
                "mdd": float(val.get("max_drawdown", np.nan)),
                "turnover_per_step": float(val.get("turnover_per_step", np.nan)),
                "trade_step_ratio": float(val.get("trade_step_ratio", np.nan)),
                "full_position_ratio": float(val.get("full_position_ratio", np.nan)),
                "avg_position": float(val.get("avg_position", np.nan)),
                "buyhold_reward": float(val_buyhold.get("total_reward", np.nan)),
                "buyhold_ret": float(val_buyhold.get("cumulative_return", np.nan)),
                "buyhold_cvar": float(val_buyhold.get("cvar", np.nan)),
                "buyhold_mdd": float(val_buyhold.get("max_drawdown", np.nan)),
                "buyhold_turnover_per_step": float(val_buyhold.get("turnover_per_step", np.nan)),
                "buyhold_trade_step_ratio": float(val_buyhold.get("trade_step_ratio", np.nan)),
                "buyhold_full_position_ratio": float(val_buyhold.get("full_position_ratio", np.nan)),
                "buyhold_avg_position": float(val_buyhold.get("avg_position", np.nan)),
                "train_seconds": float(metrics["training"].get("total_train_seconds", np.nan)),
            }
        )
        is_f, cv_v, md_v, tot_v = compute_constraint_status(
            cvar=float(val.get("cvar", np.nan)),
            max_drawdown=float(val.get("max_drawdown", np.nan)),
            max_cvar=effective_constraint_max_cvar,
            min_max_drawdown=effective_constraint_min_mdd,
            weight_cvar=args.constraint_weight_cvar,
            weight_mdd=args.constraint_weight_mdd,
        )
        c_rows.append(
            {
                "is_feasible": float(int(is_f)),
                "cvar_violation": float(cv_v),
                "mdd_violation": float(md_v),
                "total_violation": float(tot_v),
            }
        )

    s_df = pd.DataFrame(seed_rows)
    c_df = pd.DataFrame(c_rows)

    reward_mean = float(s_df["reward"].mean())
    ret_mean = float(s_df["ret"].mean())
    cvar_mean = float(s_df["cvar"].mean())
    mdd_mean = float(s_df["mdd"].mean())
    turnover_per_step_mean = float(s_df["turnover_per_step"].mean())
    trade_step_ratio_mean = float(s_df["trade_step_ratio"].mean())
    full_position_ratio_mean = float(s_df["full_position_ratio"].mean())
    avg_position_mean = float(s_df["avg_position"].mean())
    buyhold_reward_mean = float(s_df["buyhold_reward"].mean())
    buyhold_ret_mean = float(s_df["buyhold_ret"].mean())
    buyhold_cvar_mean = float(s_df["buyhold_cvar"].mean())
    buyhold_mdd_mean = float(s_df["buyhold_mdd"].mean())
    buyhold_turnover_per_step_mean = float(s_df["buyhold_turnover_per_step"].mean())
    buyhold_trade_step_ratio_mean = float(s_df["buyhold_trade_step_ratio"].mean())
    buyhold_full_position_ratio_mean = float(s_df["buyhold_full_position_ratio"].mean())
    buyhold_avg_position_mean = float(s_df["buyhold_avg_position"].mean())
    reward_std = float(s_df["reward"].std(ddof=1)) if len(s_df) > 1 else 0.0
    ret_std = float(s_df["ret"].std(ddof=1)) if len(s_df) > 1 else 0.0
    cvar_std = float(s_df["cvar"].std(ddof=1)) if len(s_df) > 1 else 0.0
    mdd_std = float(s_df["mdd"].std(ddof=1)) if len(s_df) > 1 else 0.0
    turnover_per_step_std = float(s_df["turnover_per_step"].std(ddof=1)) if len(s_df) > 1 else 0.0
    trade_step_ratio_std = float(s_df["trade_step_ratio"].std(ddof=1)) if len(s_df) > 1 else 0.0
    full_position_ratio_std = float(s_df["full_position_ratio"].std(ddof=1)) if len(s_df) > 1 else 0.0
    train_mean = float(s_df["train_seconds"].mean())
    train_std = float(s_df["train_seconds"].std(ddof=1)) if len(s_df) > 1 else 0.0

    excess_reward_mean = float(reward_mean - buyhold_reward_mean)
    excess_ret_mean = float(ret_mean - buyhold_ret_mean)
    excess_cvar_mean = float(cvar_mean - buyhold_cvar_mean)
    excess_mdd_mean = float(mdd_mean - buyhold_mdd_mean)
    turnover_gap_vs_buyhold = float(turnover_per_step_mean - buyhold_turnover_per_step_mean)
    trade_ratio_gap_vs_buyhold = float(trade_step_ratio_mean - buyhold_trade_step_ratio_mean)
    full_position_gap_vs_buyhold = float(full_position_ratio_mean - buyhold_full_position_ratio_mean)
    avg_position_gap_vs_buyhold = float(avg_position_mean - buyhold_avg_position_mean)

    mean_risk_feasible_rate = float(c_df["is_feasible"].mean()) if not c_df.empty else 0.0
    is_risk_feasible_mean, cvar_violation, mdd_violation, risk_violation = compute_constraint_status(
        cvar=cvar_mean,
        max_drawdown=mdd_mean,
        max_cvar=effective_constraint_max_cvar,
        min_max_drawdown=effective_constraint_min_mdd,
        weight_cvar=args.constraint_weight_cvar,
        weight_mdd=args.constraint_weight_mdd,
    )
    feasible_rate_violation = max(0.0, float(args.constraint_min_feasible_rate) - mean_risk_feasible_rate)

    reward_floor_violation = 0.0
    if effective_reward_floor is not None:
        reward_floor_violation = max(0.0, float(effective_reward_floor) - reward_mean)

    return_floor_violation = 0.0
    if effective_return_floor is not None:
        return_floor_violation = max(0.0, float(effective_return_floor) - ret_mean)

    risk_total_violation = float(risk_violation + feasible_rate_violation)
    profit_total_violation = float(reward_floor_violation + return_floor_violation)

    excess_reward_floor_violation = 0.0
    if args.min_excess_reward_floor is not None:
        excess_reward_floor_violation = max(0.0, float(args.min_excess_reward_floor) - excess_reward_mean)

    excess_return_floor_violation = 0.0
    if args.min_excess_return_floor is not None:
        excess_return_floor_violation = max(0.0, float(args.min_excess_return_floor) - excess_ret_mean)

    active_total_violation = float(excess_reward_floor_violation + excess_return_floor_violation)
    total_violation = float(risk_total_violation + profit_total_violation)
    total_violation = float(total_violation + active_total_violation)

    risk_feasible = bool(
        is_risk_feasible_mean and mean_risk_feasible_rate + 1e-12 >= float(args.constraint_min_feasible_rate)
    )
    profit_feasible = bool(reward_floor_violation <= 1e-12 and return_floor_violation <= 1e-12)
    active_feasible = bool(excess_reward_floor_violation <= 1e-12 and excess_return_floor_violation <= 1e-12)
    is_feasible = bool(risk_feasible and profit_feasible and active_feasible)

    # Composite is logged as a diagnostic, but ranking is lexicographic on feasibility/reward/return.
    composite = float(
        reward_mean
        + float(args.composite_return_weight) * ret_mean
        - float(args.composite_reward_std_weight) * reward_std
        - float(args.composite_return_std_weight) * ret_std
    )

    row: Dict[str, Any] = {
        "phase": phase_idx,
        "trial": trial_rank,
        "episodes": episodes,
        "n_tuning_seeds": int(len(tuning_seeds)),
        "tuning_seeds": ",".join(str(s) for s in tuning_seeds),
        "is_feasible": int(is_feasible),
        "risk_feasible": int(risk_feasible),
        "profit_feasible": int(profit_feasible),
        "active_feasible": int(active_feasible),
        "feasible_rate": mean_risk_feasible_rate,
        "risk_feasible_rate": mean_risk_feasible_rate,
        "feasible_rate_violation": feasible_rate_violation,
        "reward_floor_violation": reward_floor_violation,
        "return_floor_violation": return_floor_violation,
        "excess_reward_floor_violation": excess_reward_floor_violation,
        "excess_return_floor_violation": excess_return_floor_violation,
        "cvar_violation": cvar_violation,
        "mdd_violation": mdd_violation,
        "risk_total_violation": risk_total_violation,
        "profit_total_violation": profit_total_violation,
        "active_total_violation": active_total_violation,
        "total_violation": total_violation,
        "val_total_reward": reward_mean,
        "val_total_reward_std": reward_std,
        "val_cumulative_return": ret_mean,
        "val_cumulative_return_std": ret_std,
        "val_cvar": cvar_mean,
        "val_cvar_std": cvar_std,
        "val_max_drawdown": mdd_mean,
        "val_max_drawdown_std": mdd_std,
        "val_turnover_per_step": turnover_per_step_mean,
        "val_turnover_per_step_std": turnover_per_step_std,
        "val_trade_step_ratio": trade_step_ratio_mean,
        "val_trade_step_ratio_std": trade_step_ratio_std,
        "val_full_position_ratio": full_position_ratio_mean,
        "val_full_position_ratio_std": full_position_ratio_std,
        "val_avg_position": avg_position_mean,
        "val_buyhold_total_reward": buyhold_reward_mean,
        "val_buyhold_cumulative_return": buyhold_ret_mean,
        "val_buyhold_cvar": buyhold_cvar_mean,
        "val_buyhold_max_drawdown": buyhold_mdd_mean,
        "val_buyhold_turnover_per_step": buyhold_turnover_per_step_mean,
        "val_buyhold_trade_step_ratio": buyhold_trade_step_ratio_mean,
        "val_buyhold_full_position_ratio": buyhold_full_position_ratio_mean,
        "val_buyhold_avg_position": buyhold_avg_position_mean,
        "val_excess_reward_vs_buyhold": excess_reward_mean,
        "val_excess_return_vs_buyhold": excess_ret_mean,
        "val_excess_cvar_vs_buyhold": excess_cvar_mean,
        "val_excess_mdd_vs_buyhold": excess_mdd_mean,
        "val_turnover_gap_vs_buyhold": turnover_gap_vs_buyhold,
        "val_trade_step_ratio_gap_vs_buyhold": trade_ratio_gap_vs_buyhold,
        "val_full_position_ratio_gap_vs_buyhold": full_position_gap_vs_buyhold,
        "val_avg_position_gap_vs_buyhold": avg_position_gap_vs_buyhold,
        "selection_objective": composite,
        "train_seconds": train_mean,
        "train_seconds_std": train_std,
        "constraint_max_cvar": float(args.constraint_max_cvar),
        "constraint_min_max_drawdown": float(args.constraint_min_max_drawdown),
        "effective_constraint_max_cvar": effective_constraint_max_cvar,
        "effective_constraint_min_max_drawdown": effective_constraint_min_mdd,
        "constraint_min_feasible_rate": float(args.constraint_min_feasible_rate),
        "incumbent_cvar": args.incumbent_cvar,
        "incumbent_max_drawdown": args.incumbent_max_drawdown,
        "incumbent_cvar_slack": args.incumbent_cvar_slack,
        "incumbent_mdd_slack": args.incumbent_mdd_slack,
        "incumbent_reward": args.incumbent_reward,
        "incumbent_return": args.incumbent_return,
        "reward_floor_ratio": args.reward_floor_ratio,
        "return_floor_ratio": args.return_floor_ratio,
        "effective_reward_floor": effective_reward_floor,
        "effective_return_floor": effective_return_floor,
        "min_excess_reward_floor": args.min_excess_reward_floor,
        "min_excess_return_floor": args.min_excess_return_floor,
        "rank_excess_vs_buyhold": int(bool(args.rank_excess_vs_buyhold)),
        "composite_return_weight": float(args.composite_return_weight),
        "composite_reward_std_weight": float(args.composite_reward_std_weight),
        "composite_return_std_weight": float(args.composite_return_std_weight),
        **params,
    }
    return row


def rank_phase(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    sort_cols: List[str] = [
        "risk_feasible",
        "risk_feasible_rate",
        "risk_total_violation",
        "profit_feasible",
        "active_feasible",
    ]
    ascending: List[bool] = [False, False, True, False, False]

    if bool(args.rank_excess_vs_buyhold):
        if args.selection_priority == "return_first":
            sort_cols.extend(
                [
                    "val_excess_return_vs_buyhold",
                    "val_excess_reward_vs_buyhold",
                    "active_total_violation",
                ]
            )
        else:
            sort_cols.extend(
                [
                    "val_excess_reward_vs_buyhold",
                    "val_excess_return_vs_buyhold",
                    "active_total_violation",
                ]
            )
        ascending.extend([False, False, True])

    if args.selection_priority == "return_first":
        if args.risk_tiebreak == "strong":
            sort_cols.extend(
                [
                    "val_cumulative_return",
                    "val_cvar",
                    "val_max_drawdown",
                    "val_total_reward",
                    "profit_total_violation",
                    "val_cumulative_return_std",
                    "val_total_reward_std",
                ]
            )
            ascending.extend([False, True, False, False, True, True, True])
        else:
            sort_cols.extend(
                [
                    "val_cumulative_return",
                    "val_total_reward",
                    "profit_total_violation",
                    "val_cumulative_return_std",
                    "val_total_reward_std",
                    "val_cvar",
                    "val_max_drawdown",
                ]
            )
            ascending.extend([False, False, True, True, True, True, False])
    else:
        if args.risk_tiebreak == "strong":
            sort_cols.extend(
                [
                    "val_total_reward",
                    "val_cvar",
                    "val_max_drawdown",
                    "val_cumulative_return",
                    "profit_total_violation",
                    "val_total_reward_std",
                    "val_cumulative_return_std",
                ]
            )
            ascending.extend([False, True, False, False, True, True, True])
        else:
            sort_cols.extend(
                [
                    "val_total_reward",
                    "val_cumulative_return",
                    "profit_total_violation",
                    "val_total_reward_std",
                    "val_cumulative_return_std",
                    "val_cvar",
                    "val_max_drawdown",
                ]
            )
            ascending.extend([False, False, True, True, True, True, False])

    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    phase_episodes, phase_keep, phase_seed_groups = validate_phase_specs(args)
    rng = np.random.default_rng(args.seed)

    base_cfg = load_config(args.config)
    split = prepare_dataset_split(base_cfg)
    search_space = build_search_space(args.algo, args.space)
    candidates = sample_trials(search_space, args.trials, rng)
    if args.include_base_candidate:
        base_candidate = build_base_candidate(search_space, base_cfg)
        candidates = inject_base_candidate(candidates, base_candidate, int(args.trials))
    candidate_df = pd.DataFrame(
        [{"candidate_id": i + 1, **params} for i, params in enumerate(candidates)]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / args.algo / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_constraint_max_cvar, effective_constraint_min_mdd = compute_effective_constraint_bounds(args)
    effective_reward_floor, effective_return_floor = compute_effective_profit_floors(args)

    survivors = candidate_df.copy()
    phase_artifacts: List[Dict[str, Any]] = []

    for phase_idx, (episodes, keep_n, tuning_seeds) in enumerate(
        zip(phase_episodes, phase_keep, phase_seed_groups), start=1
    ):
        phase_dir = out_dir / f"phase_{phase_idx:02d}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []

        print(
            f"Phase {phase_idx}: {len(survivors)} candidates | episodes={episodes} "
            f"| keep={keep_n} | tuning_seeds={tuning_seeds}"
        )

        for j, row in survivors.reset_index(drop=True).iterrows():
            candidate_id = int(row["candidate_id"])
            params = {}
            for k in row.index:
                if k == "candidate_id":
                    continue
                v = row[k]
                params[k] = v.item() if hasattr(v, "item") else v
            trial_out_dir = phase_dir / f"candidate_{candidate_id:03d}"
            eval_row = evaluate_candidate(
                algo=args.algo,
                base_cfg=base_cfg,
                params=params,
                split=split,
                phase_idx=phase_idx,
                trial_rank=candidate_id,
                episodes=int(episodes),
                tuning_seeds=tuning_seeds,
                out_dir=phase_dir,
                args=args,
                effective_constraint_max_cvar=effective_constraint_max_cvar,
                effective_constraint_min_mdd=effective_constraint_min_mdd,
                effective_reward_floor=effective_reward_floor,
                effective_return_floor=effective_return_floor,
            )
            eval_row["candidate_id"] = candidate_id
            rows.append(eval_row)
            print(
                f"[P{phase_idx}] {j+1}/{len(survivors)} cand={candidate_id}: "
                f"risk={eval_row['risk_feasible']} profit={eval_row['profit_feasible']} "
                f"rate={eval_row['risk_feasible_rate']:.2f} "
                f"reward={eval_row['val_total_reward']:.4f} ret={eval_row['val_cumulative_return']:.4f} "
                f"cvar={eval_row['val_cvar']:.4f} mdd={eval_row['val_max_drawdown']:.4f}"
            )

        phase_log = rank_phase(pd.DataFrame(rows), args)
        phase_log.to_csv(phase_dir / "phase_log.csv", index=False)

        keep_n_eff = min(int(keep_n), len(phase_log))
        survivors = (
            phase_log.head(keep_n_eff)[["candidate_id"]]
            .merge(candidate_df, on="candidate_id", how="left")
            .reset_index(drop=True)
        )

        best_phase = phase_log.iloc[0].to_dict()
        phase_artifacts.append(
            {
                "phase": phase_idx,
                "episodes": int(episodes),
                "tuning_seeds": list(map(int, tuning_seeds)),
                "n_candidates_in": int(len(rows)),
                "n_survivors": int(keep_n_eff),
                "best_candidate_id": int(best_phase["candidate_id"]),
                "best_reward": float(best_phase["val_total_reward"]),
                "best_return": float(best_phase["val_cumulative_return"]),
                "best_cvar": float(best_phase["val_cvar"]),
                "best_max_drawdown": float(best_phase["val_max_drawdown"]),
                "best_is_feasible": int(best_phase["is_feasible"]),
                "best_risk_feasible": int(best_phase["risk_feasible"]),
                "best_profit_feasible": int(best_phase["profit_feasible"]),
                "best_active_feasible": int(best_phase.get("active_feasible", 1)),
                "best_feasible_rate": float(best_phase["feasible_rate"]),
                "best_excess_reward_vs_buyhold": float(best_phase.get("val_excess_reward_vs_buyhold", 0.0)),
                "best_excess_return_vs_buyhold": float(best_phase.get("val_excess_return_vs_buyhold", 0.0)),
            }
        )

    final_phase_log = pd.read_csv(out_dir / f"phase_{len(phase_episodes):02d}" / "phase_log.csv")
    if args.require_final_feasible and not bool((final_phase_log["is_feasible"] == 1).any()):
        with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "algo": args.algo,
                    "strategy": "successive_halving_lexicographic",
                    "status": "failed_no_final_feasible_candidate",
                    "space_mode": args.space,
                    "search_seed": int(args.seed),
                    "trials": int(args.trials),
                    "phase_episodes": phase_episodes,
                    "phase_keep": phase_keep,
                    "phase_tuning_seeds": phase_seed_groups,
                    "constraint_max_cvar": float(args.constraint_max_cvar),
                    "constraint_min_max_drawdown": float(args.constraint_min_max_drawdown),
                    "effective_constraint_max_cvar": float(effective_constraint_max_cvar),
                    "effective_constraint_min_max_drawdown": float(effective_constraint_min_mdd),
                    "effective_reward_floor": effective_reward_floor,
                    "effective_return_floor": effective_return_floor,
                    "min_excess_reward_floor": args.min_excess_reward_floor,
                    "min_excess_return_floor": args.min_excess_return_floor,
                    "rank_excess_vs_buyhold": bool(args.rank_excess_vs_buyhold),
                    "selection_priority": str(args.selection_priority),
                    "risk_tiebreak": str(args.risk_tiebreak),
                    "require_final_feasible": True,
                    "include_base_candidate": bool(args.include_base_candidate),
                    "phases": phase_artifacts,
                },
                f,
                indent=2,
            )
        raise RuntimeError(
            "Final phase contains no fully feasible candidate "
            "(risk + profit floors). Tuning run intentionally failed."
        )

    final_best = final_phase_log.iloc[0].to_dict()
    final_candidate_id = int(final_best["candidate_id"])
    final_params_row = candidate_df[candidate_df["candidate_id"] == final_candidate_id].iloc[0]
    final_params: Dict[str, Any] = {}
    for k in final_params_row.index:
        if k == "candidate_id":
            continue
        v = final_params_row[k]
        final_params[k] = v.item() if hasattr(v, "item") else v
    final_episodes = int(phase_episodes[-1])
    # Persist final best config with the final-phase episode budget so downstream
    # eval/gate reflects the same training-time setting used in phase 3.
    best_cfg = apply_params(base_cfg, final_params, episodes_override=final_episodes)

    with open(out_dir / "best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f)

    with open(out_dir / "best_trial.json", "w", encoding="utf-8") as f:
        json.dump(final_best, f, indent=2)

    with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "algo": args.algo,
                "strategy": "successive_halving_lexicographic",
                "space_mode": args.space,
                "search_seed": int(args.seed),
                "trials": int(args.trials),
                "phase_episodes": phase_episodes,
                "phase_keep": phase_keep,
                "phase_tuning_seeds": phase_seed_groups,
                "constraint_max_cvar": float(args.constraint_max_cvar),
                "constraint_min_max_drawdown": float(args.constraint_min_max_drawdown),
                "effective_constraint_max_cvar": float(effective_constraint_max_cvar),
                "effective_constraint_min_max_drawdown": float(effective_constraint_min_mdd),
                "incumbent_cvar": args.incumbent_cvar,
                "incumbent_max_drawdown": args.incumbent_max_drawdown,
                "incumbent_cvar_slack": args.incumbent_cvar_slack,
                "incumbent_mdd_slack": args.incumbent_mdd_slack,
                "constraint_min_feasible_rate": float(args.constraint_min_feasible_rate),
                "min_reward_floor": args.min_reward_floor,
                "min_return_floor": args.min_return_floor,
                "incumbent_reward": args.incumbent_reward,
                "incumbent_return": args.incumbent_return,
                "reward_floor_ratio": args.reward_floor_ratio,
                "return_floor_ratio": args.return_floor_ratio,
                "effective_reward_floor": effective_reward_floor,
                "effective_return_floor": effective_return_floor,
                "min_excess_reward_floor": args.min_excess_reward_floor,
                "min_excess_return_floor": args.min_excess_return_floor,
                "rank_excess_vs_buyhold": bool(args.rank_excess_vs_buyhold),
                "selection_priority": str(args.selection_priority),
                "risk_tiebreak": str(args.risk_tiebreak),
                "composite_return_weight": float(args.composite_return_weight),
                "composite_reward_std_weight": float(args.composite_reward_std_weight),
                "composite_return_std_weight": float(args.composite_return_std_weight),
                "include_base_candidate": bool(args.include_base_candidate),
                "final_candidate_id": final_candidate_id,
                "final_phase_episodes": final_episodes,
                "require_final_feasible": bool(args.require_final_feasible),
                "phases": phase_artifacts,
            },
            f,
            indent=2,
        )

    latest_ptr = Path(args.output_root) / args.algo / "LATEST.txt"
    latest_ptr.write_text(str(out_dir), encoding="utf-8")

    print(f"Saved successive-halving tuning artifacts to: {out_dir}")
    print(
        f"Best candidate={final_candidate_id} | reward={final_best['val_total_reward']:.6f} "
        f"| return={final_best['val_cumulative_return']:.6f} | cvar={final_best['val_cvar']:.6f} "
        f"| mdd={final_best['val_max_drawdown']:.6f}"
    )


if __name__ == "__main__":
    main()
