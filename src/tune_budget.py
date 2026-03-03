"""Budgeted random-search tuning using validation metrics only.

Usage:
  scripts/py.sh -m src.tune_budget --algo sg_sarsa --trials 24 --episodes-override 80
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.pipeline import prepare_dataset_split
from src.train import load_config, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budgeted random-search tuning (validation-only selection)")
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
        default="default",
    )
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=11, help="Random seed for search-space sampling (and fallback tuning seed).")
    parser.add_argument(
        "--tuning-seeds",
        default="",
        help="Comma-separated seeds used for validation-only model selection (e.g., '11,22'). "
        "If omitted, falls back to --seed.",
    )
    parser.add_argument("--episodes-override", type=int, default=80)
    parser.add_argument(
        "--selection-metric",
        choices=["val_total_reward", "risk_adjusted", "constrained_reward", "constrained_composite"],
        default="constrained_reward",
        help="Model-selection objective computed on validation set only.",
    )
    parser.add_argument("--risk-weight-cvar", type=float, default=10.0)
    parser.add_argument("--risk-weight-mdd", type=float, default=2.0)
    parser.add_argument(
        "--constraint-max-cvar",
        type=float,
        default=0.065,
        help="Feasibility threshold: validation CVaR must be <= this value.",
    )
    parser.add_argument(
        "--constraint-min-max-drawdown",
        type=float,
        default=-0.60,
        help="Feasibility threshold: validation max_drawdown must be >= this value (less negative).",
    )
    parser.add_argument(
        "--incumbent-cvar",
        type=float,
        default=None,
        help="Optional incumbent CVaR used to tighten constraints via a relative guardrail.",
    )
    parser.add_argument(
        "--incumbent-max-drawdown",
        type=float,
        default=None,
        help="Optional incumbent max drawdown used to tighten constraints via a relative guardrail.",
    )
    parser.add_argument(
        "--incumbent-cvar-slack",
        type=float,
        default=None,
        help="Allowed CVaR deterioration versus incumbent (effective max CVaR = incumbent_cvar + slack).",
    )
    parser.add_argument(
        "--incumbent-mdd-slack",
        type=float,
        default=None,
        help="Allowed max-drawdown deterioration (more negative) versus incumbent "
        "(effective min MDD = incumbent_max_drawdown - slack).",
    )
    parser.add_argument("--constraint-weight-cvar", type=float, default=1.0)
    parser.add_argument("--constraint-weight-mdd", type=float, default=1.0)
    parser.add_argument(
        "--constraint-min-feasible-rate",
        type=float,
        default=1.0,
        help="For multi-seed tuning, minimum fraction of tuning seeds that must satisfy constraints.",
    )
    parser.add_argument(
        "--composite-return-weight",
        type=float,
        default=0.12,
        help="Weight on validation cumulative return mean for constrained_composite selection.",
    )
    parser.add_argument(
        "--composite-reward-std-weight",
        type=float,
        default=0.20,
        help="Penalty weight on validation reward std for constrained_composite selection.",
    )
    parser.add_argument(
        "--composite-return-std-weight",
        type=float,
        default=0.05,
        help="Penalty weight on validation cumulative return std for constrained_composite selection.",
    )
    parser.add_argument("--output-root", default="runs/tuning_budget")
    return parser.parse_args()


def common_space_default() -> Dict[str, List[Any]]:
    return {
        "env.lambda_cvar": [0.01, 0.02, 0.03, 0.04, 0.06],
        "env.lambda_vol": [0.002, 0.005, 0.008, 0.012],
        "env.risk_window": [40, 60, 80],
        "env.cvar_target": [0.03, 0.04, 0.05],
        "env.vol_target": [0.015, 0.02, 0.025],
        "env.opportunity_cost_coeff": [0.1, 0.2, 0.3],
        "training.epsilon_decay": [0.95, 0.97, 0.985],
        "training.gamma": [0.98, 0.99],
    }


def algo_space_default(algo: str) -> Dict[str, List[Any]]:
    if algo == "q_learning":
        return {
            "q_learning.alpha": [0.03, 0.05, 0.08],
            "q_learning.bins": [6, 8, 10],
        }
    if algo == "sg_sarsa":
        return {
            "sg_sarsa.alpha": [0.01, 0.015, 0.02, 0.03],
        }
    if algo == "nstep_sarsa":
        return {
            "nstep_sarsa.alpha": [0.01, 0.015, 0.02, 0.03],
            "nstep_sarsa.n": [2, 3, 5],
        }
    raise ValueError(f"Unsupported algorithm: {algo}")


def focused_space(algo: str) -> Dict[str, List[Any]]:
    if algo == "sg_sarsa":
        return {
            # Requested local refinement dimensions.
            "sg_sarsa.alpha": [0.005, 0.01, 0.015, 0.02, 0.03],
            "training.epsilon_decay": [0.95, 0.97, 0.985, 0.992],
            "env.opportunity_cost_coeff": [0.1, 0.2, 0.3, 0.4],
            "env.lambda_cvar": [0.005, 0.01, 0.015, 0.02, 0.03],
        }
    if algo == "nstep_sarsa":
        return {
            # Requested local refinement dimensions with stronger risk controls.
            "nstep_sarsa.alpha": [0.01, 0.015, 0.02, 0.03],
            "nstep_sarsa.n": [3, 5, 7],
            "training.epsilon_decay": [0.95, 0.97, 0.985],
            "env.lambda_cvar": [0.005, 0.01, 0.02, 0.03, 0.04],
            "env.lambda_vol": [0.002, 0.005, 0.008],
            "env.cvar_target": [0.03, 0.04, 0.05],
            "env.vol_target": [0.015, 0.02, 0.025],
        }
    if algo == "q_learning":
        return {
            "q_learning.alpha": [0.03, 0.05, 0.08],
            "q_learning.bins": [6, 8],
            "training.epsilon_decay": [0.95, 0.97, 0.985],
            "env.lambda_cvar": [0.01, 0.02, 0.03],
        }
    raise ValueError(f"Unsupported algorithm: {algo}")


def focused_space_v2(algo: str) -> Dict[str, List[Any]]:
    """Second-round local refinement around QQQ v1 winners.

    The first QQQ tuning round showed many feasible trials and repeated validation
    metrics, so this space narrows high-impact parameters while expanding around
    boundary hits (notably small lambda_cvar and n in {3,5}).
    """
    if algo == "sg_sarsa":
        return {
            "sg_sarsa.alpha": [0.01, 0.015, 0.02, 0.025, 0.03],
            "training.epsilon_decay": [0.95, 0.965, 0.97, 0.975, 0.985],
            "env.opportunity_cost_coeff": [0.05, 0.1, 0.15, 0.2],
            "env.lambda_cvar": [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015],
            "env.lambda_vol": [0.002, 0.005, 0.008],
            "env.cvar_target": [0.03, 0.04],
        }
    if algo == "nstep_sarsa":
        return {
            "nstep_sarsa.alpha": [0.005, 0.01, 0.015, 0.02, 0.03],
            "nstep_sarsa.n": [3, 4, 5, 7],
            "training.epsilon_decay": [0.95, 0.97, 0.975, 0.985],
            "env.lambda_cvar": [0.0, 0.0025, 0.005, 0.01, 0.02],
            "env.lambda_vol": [0.002, 0.005, 0.008, 0.012],
            "env.cvar_target": [0.025, 0.03, 0.04],
            "env.vol_target": [0.02, 0.025, 0.03],
        }
    if algo == "q_learning":
        return {
            "q_learning.alpha": [0.03, 0.05, 0.08],
            "q_learning.bins": [6, 8, 10],
            "training.epsilon_decay": [0.95, 0.97, 0.985],
            "env.lambda_cvar": [0.0, 0.005, 0.01, 0.02],
        }
    raise ValueError(f"Unsupported algorithm: {algo}")


def focused_space_v3(algo: str) -> Dict[str, List[Any]]:
    """Third-round refinement for QQQ: reward-oriented within risk constraints."""
    if algo == "sg_sarsa":
        return {
            "sg_sarsa.alpha": [0.01, 0.015, 0.02, 0.025],
            "training.epsilon_decay": [0.95, 0.965, 0.97, 0.975],
            "env.opportunity_cost_coeff": [0.05, 0.1, 0.15],
            "env.lambda_cvar": [0.0, 0.0025, 0.005, 0.0075, 0.01],
            "env.lambda_vol": [0.002, 0.005],
            "env.cvar_target": [0.04, 0.05],
            "env.vol_target": [0.02, 0.025],
        }
    if algo == "nstep_sarsa":
        return {
            "nstep_sarsa.alpha": [0.01, 0.015, 0.02, 0.03],
            "nstep_sarsa.n": [3, 4, 5],
            "training.epsilon_decay": [0.97, 0.975, 0.985, 0.99],
            "env.opportunity_cost_coeff": [0.05, 0.1, 0.15],
            "env.lambda_cvar": [0.0, 0.0025, 0.005, 0.01],
            "env.lambda_vol": [0.002, 0.005, 0.008],
            "env.cvar_target": [0.03, 0.04],
            "env.vol_target": [0.02, 0.025],
        }
    if algo == "q_learning":
        return {
            "q_learning.alpha": [0.03, 0.05, 0.08],
            "q_learning.bins": [6, 8, 10],
            "training.epsilon_decay": [0.95, 0.97, 0.985],
            "env.lambda_cvar": [0.0, 0.0025, 0.005, 0.01],
        }
    raise ValueError(f"Unsupported algorithm: {algo}")


def design_ablation_space(algo: str) -> Dict[str, List[Any]]:
    """Design ablation space: tune core RL/risk parameters while fixing environment design knobs.

    Intentionally excludes `env.opportunity_cost_coeff` so D0/D1/D2 differ by config design,
    not by hyperparameter search overriding the design.
    """
    if algo == "nstep_sarsa":
        return {
            "nstep_sarsa.alpha": [0.01, 0.015, 0.02],
            "nstep_sarsa.n": [3, 4, 5],
            "training.epsilon_decay": [0.97, 0.975, 0.985, 0.99],
            "env.lambda_cvar": [0.0, 0.0025, 0.005, 0.01],
            "env.lambda_vol": [0.002, 0.005, 0.008],
            "env.cvar_target": [0.03, 0.04],
            "env.vol_target": [0.02, 0.025],
        }
    if algo == "sg_sarsa":
        return {
            "sg_sarsa.alpha": [0.01, 0.015, 0.02, 0.025],
            "training.epsilon_decay": [0.95, 0.965, 0.97, 0.975],
            "env.lambda_cvar": [0.0, 0.0025, 0.005, 0.0075, 0.01],
            "env.lambda_vol": [0.002, 0.005],
            "env.cvar_target": [0.04, 0.05],
            "env.vol_target": [0.02, 0.025],
        }
    if algo == "q_learning":
        return {
            "q_learning.alpha": [0.03, 0.05, 0.08],
            "q_learning.bins": [6, 8, 10],
            "training.epsilon_decay": [0.95, 0.97, 0.985],
            "env.lambda_cvar": [0.0, 0.0025, 0.005, 0.01],
        }
    raise ValueError(f"Unsupported algorithm: {algo}")


def design_ablation_risk_space(algo: str) -> Dict[str, List[Any]]:
    """Risk-sensitive variant of design_ablation_space.

    Same as design_ablation, but excludes zero-CVaR-penalty settings so the
    tuning result remains within the intended risk-sensitive formulation.
    """
    space = design_ablation_space(algo)
    if "env.lambda_cvar" in space:
        space["env.lambda_cvar"] = [float(v) for v in space["env.lambda_cvar"] if float(v) > 0.0]
        if not space["env.lambda_cvar"]:
            raise ValueError(f"Risk-sensitive search space for {algo} has no positive lambda_cvar choices")
    return space


def xlf_sync_safe_space(algo: str) -> Dict[str, List[Any]]:
    """Safer search space for synchronized XLF method comparison rounds.

    Goal:
    - Keep the same training budget / evaluation protocol across algorithms.
    - Expand into lower-variance regions (smaller learning rates, faster exploration decay).
    - Keep positive CVaR penalty to preserve risk-sensitive formulation.
    """
    if algo == "q_learning":
        return {
            "q_learning.alpha": [0.005, 0.01, 0.02, 0.03, 0.05],
            "q_learning.bins": [4, 6, 8],
            "training.epsilon_decay": [0.985, 0.99, 0.995],
            "env.opportunity_cost_coeff": [0.01, 0.02, 0.05],
            "env.lambda_cvar": [0.0025, 0.005, 0.01, 0.02, 0.03],
            "env.lambda_vol": [0.005, 0.008, 0.012],
            "env.cvar_target": [0.02, 0.03, 0.04],
            "env.vol_target": [0.02, 0.025],
        }
    if algo == "sg_sarsa":
        return {
            "sg_sarsa.alpha": [0.002, 0.005, 0.01, 0.015, 0.02],
            "training.epsilon_decay": [0.97, 0.985, 0.99, 0.995],
            "env.opportunity_cost_coeff": [0.01, 0.02, 0.05],
            "env.lambda_cvar": [0.0025, 0.005, 0.01, 0.02, 0.03],
            "env.lambda_vol": [0.005, 0.008, 0.012],
            "env.cvar_target": [0.02, 0.03, 0.04],
            "env.vol_target": [0.02, 0.025],
        }
    if algo == "nstep_sarsa":
        return {
            "nstep_sarsa.alpha": [0.005, 0.01, 0.015, 0.02],
            "nstep_sarsa.n": [3, 4, 5],
            "training.epsilon_decay": [0.985, 0.99, 0.995],
            "env.opportunity_cost_coeff": [0.01, 0.02, 0.05],
            "env.lambda_cvar": [0.0025, 0.005, 0.01, 0.02],
            "env.lambda_vol": [0.005, 0.008, 0.012],
            "env.cvar_target": [0.02, 0.03, 0.04],
            "env.vol_target": [0.02, 0.025],
        }
    raise ValueError(f"Unsupported algorithm: {algo}")


def xlf_sg_risk_push_space(algo: str) -> Dict[str, List[Any]]:
    """Risk-first refinement space for SG-Sarsa on XLF.

    This is used when synchronized round(s) show SG-Sarsa improves return/reward
    but fails gate risk tolerances (CVaR/MDD regression).
    """
    if algo != "sg_sarsa":
        # Keep other algos compatible with this space option.
        return xlf_sync_safe_space(algo)

    return {
        "sg_sarsa.alpha": [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008],
        "training.epsilon_decay": [0.94, 0.95, 0.96, 0.97],
        "training.epsilon_min": [0.01, 0.03, 0.05],
        "training.risk_warmup_episodes": [0, 20, 40, 60],
        "env.opportunity_cost_coeff": [0.0, 0.005, 0.01, 0.02],
        "env.lambda_cvar": [0.02, 0.03, 0.04, 0.06, 0.08],
        "env.lambda_vol": [0.008, 0.012, 0.016, 0.02],
        "env.cvar_target": [0.015, 0.02, 0.025, 0.03],
        "env.vol_target": [0.015, 0.02, 0.025],
    }


def xlf_sg_balanced_r3_space(algo: str) -> Dict[str, List[Any]]:
    """SG-Sarsa R3: balanced search band to recover return while preserving risk."""
    if algo != "sg_sarsa":
        return xlf_sync_safe_space(algo)

    return {
        "sg_sarsa.alpha": [0.001, 0.002, 0.003, 0.005, 0.008, 0.01],
        "training.epsilon_decay": [0.965, 0.975, 0.985, 0.99],
        "training.epsilon_min": [0.03, 0.05, 0.08],
        "training.risk_warmup_episodes": [20, 40, 60, 80],
        "env.opportunity_cost_coeff": [0.01, 0.02, 0.03],
        "env.lambda_cvar": [0.01, 0.015, 0.02, 0.025, 0.03, 0.04],
        "env.lambda_vol": [0.004, 0.006, 0.008, 0.01, 0.012],
        "env.cvar_target": [0.025, 0.03, 0.035],
        "env.vol_target": [0.02, 0.025, 0.03],
    }


def xlf_sg_boundary_r4_space(algo: str) -> Dict[str, List[Any]]:
    """SG-Sarsa R4: boundary-focused local sweep around baseline configuration.

    Objective:
    - keep return/reward near or above baseline,
    - reduce CVaR/MDD enough to satisfy promotion gate tolerances.
    """
    if algo != "sg_sarsa":
        return xlf_sync_safe_space(algo)

    return {
        # Keep baseline alpha=0.02 as explicit anchor; add local neighborhood.
        "sg_sarsa.alpha": [0.01, 0.015, 0.02, 0.025],
        # Exploration schedule around baseline epsilon_decay=0.97.
        "training.epsilon_decay": [0.965, 0.97, 0.975, 0.98],
        "training.epsilon_min": [0.03, 0.05],
        "training.risk_warmup_episodes": [40, 80, 120],
        # Soften opportunity-cost pressure that can force overexposure in uptrend windows.
        "env.opportunity_cost_coeff": [0.0, 0.01, 0.015, 0.02],
        # Risk penalties/targets around baseline with slightly stronger variants.
        "env.lambda_cvar": [0.01, 0.015, 0.02, 0.025, 0.03],
        "env.lambda_vol": [0.006, 0.008, 0.01, 0.012],
        "env.cvar_target": [0.02, 0.025, 0.03],
        "env.vol_target": [0.02, 0.025],
    }


def xlf_sg_learning_only_space(algo: str) -> Dict[str, List[Any]]:
    """Controlled SG study: keep env risk/reward design fixed, tune learning only."""
    if algo != "sg_sarsa":
        return xlf_sync_safe_space(algo)

    return {
        "sg_sarsa.alpha": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
        "training.epsilon_decay": [0.95, 0.96, 0.965, 0.97, 0.975, 0.985],
        "training.epsilon_min": [0.01, 0.03, 0.05, 0.08],
    }


def xlf_sg_learning_only_guard_space(algo: str) -> Dict[str, List[Any]]:
    """Final SG controlled validation space with exploration-floor guard.

    Same controlled setting as `xlf_sg_learning_only`, but excludes
    low-exploration schedules that can collapse into over-conservative behavior
    under longer (e.g., 200-episode) budgets.
    """
    if algo != "sg_sarsa":
        return xlf_sync_safe_space(algo)

    return {
        "sg_sarsa.alpha": [0.01, 0.015, 0.02, 0.025, 0.03],
        "training.epsilon_decay": [0.965, 0.97, 0.975, 0.98, 0.985, 0.99],
        "training.epsilon_min": [0.03, 0.05, 0.08, 0.1, 0.12],
    }


def build_search_space(algo: str, space_mode: str) -> Dict[str, List[Any]]:
    if space_mode == "xlf_sg_learning_only_guard":
        return xlf_sg_learning_only_guard_space(algo)
    if space_mode == "xlf_sg_learning_only":
        return xlf_sg_learning_only_space(algo)
    if space_mode == "xlf_sg_boundary_r4":
        return xlf_sg_boundary_r4_space(algo)
    if space_mode == "xlf_sg_balanced_r3":
        return xlf_sg_balanced_r3_space(algo)
    if space_mode == "xlf_sg_risk_push":
        return xlf_sg_risk_push_space(algo)
    if space_mode == "xlf_sync_safe":
        return xlf_sync_safe_space(algo)
    if space_mode == "design_ablation_risk":
        return design_ablation_risk_space(algo)
    if space_mode == "design_ablation":
        return design_ablation_space(algo)
    if space_mode == "focused_v3":
        return focused_space_v3(algo)
    if space_mode == "focused_v2":
        return focused_space_v2(algo)
    if space_mode == "focused":
        return focused_space(algo)
    space = {}
    space.update(common_space_default())
    space.update(algo_space_default(algo))
    return space


def set_nested(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def sample_trials(space: Dict[str, List[Any]], trials: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    keys = sorted(space.keys())
    seen = set()
    out: List[Dict[str, Any]] = []

    max_attempts = max(1000, trials * 30)
    attempts = 0
    while len(out) < trials and attempts < max_attempts:
        attempts += 1
        params: Dict[str, Any] = {}
        for k in keys:
            value = rng.choice(space[k])
            # Convert numpy scalar to Python scalar for stable YAML/JSON output.
            params[k] = value.item() if hasattr(value, "item") else value
        signature = tuple((k, params[k]) for k in keys)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(params)

    return out


def apply_params(base_cfg: Dict[str, Any], params: Dict[str, Any], episodes_override: int | None) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for k, v in params.items():
        set_nested(cfg, k, v)
    if episodes_override is not None:
        cfg["training"]["episodes"] = int(episodes_override)
    return cfg


def compute_objective(
    metric: str,
    val_metrics: Dict[str, Any],
    risk_weight_cvar: float,
    risk_weight_mdd: float,
) -> tuple[float, float, float]:
    reward = float(val_metrics.get("total_reward", np.nan))
    cvar = float(val_metrics.get("cvar", np.nan))
    max_drawdown = float(val_metrics.get("max_drawdown", np.nan))

    if np.isnan(reward) or np.isnan(cvar) or np.isnan(max_drawdown):
        return float("-inf"), cvar, max_drawdown

    if metric == "val_total_reward":
        return reward, cvar, max_drawdown

    if metric == "constrained_composite":
        ret = float(val_metrics.get("cumulative_return", np.nan))
        reward_std = float(val_metrics.get("total_reward_std", 0.0))
        ret_std = float(val_metrics.get("cumulative_return_std", 0.0))
        return_weight = float(val_metrics.get("_composite_return_weight", 0.0))
        reward_std_weight = float(val_metrics.get("_composite_reward_std_weight", 0.0))
        ret_std_weight = float(val_metrics.get("_composite_return_std_weight", 0.0))
        if np.isnan(ret):
            return float("-inf"), cvar, max_drawdown
        objective = reward + return_weight * ret - reward_std_weight * reward_std - ret_std_weight * ret_std
        return float(objective), cvar, max_drawdown

    # Risk-adjusted objective: maximize reward while penalizing tail risk and deep drawdown.
    cvar_penalty = max(0.0, cvar)
    drawdown_penalty = abs(min(0.0, max_drawdown))
    objective = reward - risk_weight_cvar * cvar_penalty - risk_weight_mdd * drawdown_penalty
    return float(objective), cvar, max_drawdown


def compute_constraint_status(
    cvar: float,
    max_drawdown: float,
    max_cvar: float,
    min_max_drawdown: float,
    weight_cvar: float,
    weight_mdd: float,
) -> tuple[bool, float, float, float]:
    cvar_violation = max(0.0, cvar - max_cvar)
    mdd_violation = max(0.0, min_max_drawdown - max_drawdown)
    total_violation = weight_cvar * cvar_violation + weight_mdd * mdd_violation
    is_feasible = cvar_violation <= 1e-12 and mdd_violation <= 1e-12
    return is_feasible, cvar_violation, mdd_violation, total_violation


def compute_effective_constraint_bounds(args: argparse.Namespace) -> tuple[float, float]:
    """Combine absolute thresholds with optional incumbent-relative guardrails."""
    max_cvar = float(args.constraint_max_cvar)
    min_mdd = float(args.constraint_min_max_drawdown)

    if args.incumbent_cvar is not None and args.incumbent_cvar_slack is not None:
        max_cvar = min(max_cvar, float(args.incumbent_cvar) + float(args.incumbent_cvar_slack))

    if args.incumbent_max_drawdown is not None and args.incumbent_mdd_slack is not None:
        # max_drawdown is negative; "slack" allows it to become more negative by this amount.
        min_mdd = max(min_mdd, float(args.incumbent_max_drawdown) - float(args.incumbent_mdd_slack))

    return max_cvar, min_mdd


def parse_tuning_seeds(raw: str, fallback_seed: int) -> List[int]:
    if not raw.strip():
        return [int(fallback_seed)]
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        return [int(fallback_seed)]
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    tuning_seeds = parse_tuning_seeds(args.tuning_seeds, fallback_seed=args.seed)
    effective_constraint_max_cvar, effective_constraint_min_mdd = compute_effective_constraint_bounds(args)

    base_cfg = load_config(args.config)

    split = prepare_dataset_split(base_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / args.algo / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    search_space = build_search_space(args.algo, args.space)
    trials = sample_trials(search_space, args.trials, rng)
    logs: List[Dict[str, Any]] = []

    for i, params in enumerate(trials, start=1):
        trial_cfg = apply_params(base_cfg, params, episodes_override=args.episodes_override)
        trial_dir = out_dir / f"trial_{i:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        with open(trial_dir / "trial_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(trial_cfg, f)

        seed_metrics: List[Dict[str, Any]] = []
        seed_constraint_rows: List[Dict[str, float]] = []
        for tuning_seed in tuning_seeds:
            seed_out_dir = trial_dir if len(tuning_seeds) == 1 else (trial_dir / f"seed_{tuning_seed}")
            if len(tuning_seeds) > 1:
                seed_out_dir.mkdir(parents=True, exist_ok=True)

            metrics = run_training(
                algo=args.algo,
                config=trial_cfg,
                train_df=split.train,
                val_df=split.val,
                test_df=split.test,
                seed=int(tuning_seed),
                output_dir=seed_out_dir,
                evaluate_test=False,
            )
            val_seed = metrics["val"]
            seed_metrics.append(
                {
                    "seed": int(tuning_seed),
                    "val_total_reward": float(val_seed.get("total_reward", np.nan)),
                    "val_cumulative_return": float(val_seed.get("cumulative_return", np.nan)),
                    "val_cvar": float(val_seed.get("cvar", np.nan)),
                    "val_max_drawdown": float(val_seed.get("max_drawdown", np.nan)),
                    "train_seconds": float(metrics["training"].get("total_train_seconds", np.nan)),
                }
            )
            s_feasible, s_cvar_v, s_mdd_v, s_total_v = compute_constraint_status(
                cvar=float(val_seed.get("cvar", np.nan)),
                max_drawdown=float(val_seed.get("max_drawdown", np.nan)),
                max_cvar=effective_constraint_max_cvar,
                min_max_drawdown=effective_constraint_min_mdd,
                weight_cvar=args.constraint_weight_cvar,
                weight_mdd=args.constraint_weight_mdd,
            )
            seed_constraint_rows.append(
                {
                    "is_feasible": float(int(s_feasible)),
                    "cvar_violation": float(s_cvar_v),
                    "mdd_violation": float(s_mdd_v),
                    "total_violation": float(s_total_v),
                }
            )

        seed_df = pd.DataFrame(seed_metrics)
        c_df = pd.DataFrame(seed_constraint_rows)
        val = {
            "total_reward": float(seed_df["val_total_reward"].mean()),
            "total_reward_std": float(seed_df["val_total_reward"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
            "cumulative_return": float(seed_df["val_cumulative_return"].mean()),
            "cumulative_return_std": float(seed_df["val_cumulative_return"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
            "cvar": float(seed_df["val_cvar"].mean()),
            "cvar_std": float(seed_df["val_cvar"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
            "max_drawdown": float(seed_df["val_max_drawdown"].mean()),
            "max_drawdown_std": float(seed_df["val_max_drawdown"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
            "_composite_return_weight": float(args.composite_return_weight),
            "_composite_reward_std_weight": float(args.composite_reward_std_weight),
            "_composite_return_std_weight": float(args.composite_return_std_weight),
        }

        objective, val_cvar, val_mdd = compute_objective(
            metric=args.selection_metric,
            val_metrics=val,
            risk_weight_cvar=args.risk_weight_cvar,
            risk_weight_mdd=args.risk_weight_mdd,
        )
        mean_feasible_rate = float(c_df["is_feasible"].mean()) if not c_df.empty else 0.0
        is_feasible_mean, cvar_violation, mdd_violation, total_violation = compute_constraint_status(
            cvar=val_cvar,
            max_drawdown=val_mdd,
            max_cvar=effective_constraint_max_cvar,
            min_max_drawdown=effective_constraint_min_mdd,
            weight_cvar=args.constraint_weight_cvar,
            weight_mdd=args.constraint_weight_mdd,
        )
        feasible_rate_violation = max(0.0, float(args.constraint_min_feasible_rate) - mean_feasible_rate)
        is_feasible = bool(is_feasible_mean and (mean_feasible_rate + 1e-12 >= float(args.constraint_min_feasible_rate)))
        total_violation = float(total_violation + feasible_rate_violation)
        logs.append(
            {
                "trial": i,
                "algo": args.algo,
                "space_mode": args.space,
                "search_seed": int(args.seed),
                "tuning_seeds": ",".join(str(s) for s in tuning_seeds),
                "n_tuning_seeds": int(len(tuning_seeds)),
                "selection_metric": args.selection_metric,
                "selection_objective": objective,
                "val_total_reward": val.get("total_reward", np.nan),
                "val_total_reward_std": val.get("total_reward_std", np.nan),
                "val_cumulative_return": val.get("cumulative_return", np.nan),
                "val_cumulative_return_std": val.get("cumulative_return_std", np.nan),
                "val_cvar": val_cvar,
                "val_cvar_std": val.get("cvar_std", np.nan),
                "val_max_drawdown": val_mdd,
                "val_max_drawdown_std": val.get("max_drawdown_std", np.nan),
                "constraint_max_cvar": args.constraint_max_cvar,
                "constraint_min_max_drawdown": args.constraint_min_max_drawdown,
                "effective_constraint_max_cvar": effective_constraint_max_cvar,
                "effective_constraint_min_max_drawdown": effective_constraint_min_mdd,
                "incumbent_cvar": args.incumbent_cvar,
                "incumbent_max_drawdown": args.incumbent_max_drawdown,
                "incumbent_cvar_slack": args.incumbent_cvar_slack,
                "incumbent_mdd_slack": args.incumbent_mdd_slack,
                "constraint_min_feasible_rate": args.constraint_min_feasible_rate,
                "is_feasible": int(is_feasible),
                "feasible_rate": mean_feasible_rate,
                "feasible_rate_violation": feasible_rate_violation,
                "cvar_violation": cvar_violation,
                "mdd_violation": mdd_violation,
                "total_violation": total_violation,
                "train_seconds": float(seed_df["train_seconds"].mean()),
                "train_seconds_std": float(seed_df["train_seconds"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
                "risk_weight_cvar": args.risk_weight_cvar,
                "risk_weight_mdd": args.risk_weight_mdd,
                "composite_return_weight": args.composite_return_weight,
                "composite_reward_std_weight": args.composite_reward_std_weight,
                "composite_return_std_weight": args.composite_return_std_weight,
                **params,
            }
        )
        print(
            f"[{args.algo}] trial {i}/{len(trials)}: "
            f"feasible={int(is_feasible)}, "
                f"feasible_rate={mean_feasible_rate:.2f}, "
                f"objective={objective:.4f}, "
                f"val_total_reward={val.get('total_reward', float('nan')):.4f}, "
                f"val_total_reward_std={val.get('total_reward_std', float('nan')):.4f}, "
                f"val_cvar={val.get('cvar', float('nan')):.4f}, "
                f"val_mdd={val.get('max_drawdown', float('nan')):.4f}, "
                f"val_return={val.get('cumulative_return', float('nan')):.4f}, "
                f"eff_cvar_max={effective_constraint_max_cvar:.4f}, "
                f"eff_mdd_min={effective_constraint_min_mdd:.4f}"
            )

    sort_col = "selection_objective"
    if args.selection_metric == "val_total_reward":
        sort_col = "val_total_reward"
        log_df = pd.DataFrame(logs).sort_values(sort_col, ascending=False).reset_index(drop=True)
    elif args.selection_metric == "constrained_reward":
        # Two-stage constrained selection:
        # 1) enforce risk thresholds (feasible first)
        # 2) maximize reward among feasible trials
        # fallback to minimum violation if no feasible trial exists
        log_df = (
            pd.DataFrame(logs)
            .sort_values(
                ["is_feasible", "total_violation", "val_total_reward"],
                ascending=[False, True, False],
            )
            .reset_index(drop=True)
        )
        sort_col = "val_total_reward"
    elif args.selection_metric == "constrained_composite":
        log_df = (
            pd.DataFrame(logs)
            .sort_values(
                [
                    "is_feasible",
                    "feasible_rate",
                    "total_violation",
                    "selection_objective",
                    "val_total_reward",
                    "val_cumulative_return",
                    "val_total_reward_std",
                    "val_cvar",
                    "val_max_drawdown",
                ],
                ascending=[False, False, True, False, False, False, True, True, False],
            )
            .reset_index(drop=True)
        )
        sort_col = "selection_objective"
    else:
        log_df = pd.DataFrame(logs).sort_values(sort_col, ascending=False).reset_index(drop=True)
    log_df.to_csv(out_dir / "tuning_log.csv", index=False)

    best_row = log_df.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in trials[0].keys()}
    best_cfg = apply_params(base_cfg, best_params, episodes_override=None)

    with open(out_dir / "best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f)
    with open(out_dir / "best_trial.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    # Convenience pointer to latest run.
    latest_path = Path(args.output_root) / args.algo / "LATEST.txt"
    latest_path.write_text(str(out_dir), encoding="utf-8")

    feasible_count = int(log_df["is_feasible"].sum()) if "is_feasible" in log_df.columns else 0
    with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "algo": args.algo,
                "selection_metric": args.selection_metric,
                "space_mode": args.space,
                "feasible_count": feasible_count,
                "total_trials": int(len(log_df)),
                "search_seed": int(args.seed),
                "tuning_seeds": tuning_seeds,
                "n_tuning_seeds": int(len(tuning_seeds)),
                "constraint_max_cvar": args.constraint_max_cvar,
                "constraint_min_max_drawdown": args.constraint_min_max_drawdown,
                "effective_constraint_max_cvar": effective_constraint_max_cvar,
                "effective_constraint_min_max_drawdown": effective_constraint_min_mdd,
                "incumbent_cvar": args.incumbent_cvar,
                "incumbent_max_drawdown": args.incumbent_max_drawdown,
                "incumbent_cvar_slack": args.incumbent_cvar_slack,
                "incumbent_mdd_slack": args.incumbent_mdd_slack,
                "constraint_min_feasible_rate": args.constraint_min_feasible_rate,
                "risk_weight_cvar": args.risk_weight_cvar,
                "risk_weight_mdd": args.risk_weight_mdd,
                "composite_return_weight": args.composite_return_weight,
                "composite_reward_std_weight": args.composite_reward_std_weight,
                "composite_return_std_weight": args.composite_return_std_weight,
            },
            f,
            indent=2,
        )

    print(f"Saved tuning artifacts to: {out_dir}")
    print(
        f"Best trial {sort_col}: {best_row[sort_col]:.6f} | "
        f"feasible={int(best_row.get('is_feasible', 0))}, "
        f"total_violation={float(best_row.get('total_violation', 0.0)):.6f}, "
        f"reward={best_row['val_total_reward']:.6f}, "
        f"cvar={best_row['val_cvar']:.6f}, "
        f"max_drawdown={best_row['val_max_drawdown']:.6f}"
    )


if __name__ == "__main__":
    main()
