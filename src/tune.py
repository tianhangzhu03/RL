"""Legacy two-stage hyperparameter tuning utility.

Deprecated for current experiments. Prefer `src.tune_budget` for
validation-only selection and constrained risk-sensitive tuning.

Usage:
    python -m src.tune --algo q_learning --config configs/base.yaml --seed 11
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

from src.pipeline import prepare_dataset_split
from src.train import load_config, run_training


@dataclass
class TrialResult:
    phase: str
    trial_id: int
    params: Dict[str, float]
    val_total_reward: float
    val_cvar: float
    train_seconds: float
    run_dir: str


def build_stage1_grid(algo: str) -> List[Dict[str, float]]:
    base = {
        "lambda_cvar": [0.0, 0.2, 0.4],
        "risk_window": [40, 60],
    }

    if algo == "q_learning":
        algo_grid = {
            "algo_alpha": [0.02, 0.05, 0.1],
            "bins": [6, 8],
        }
    elif algo == "sg_sarsa":
        algo_grid = {
            "algo_alpha": [0.005, 0.01, 0.02],
        }
    elif algo == "nstep_sarsa":
        algo_grid = {
            "algo_alpha": [0.005, 0.01],
            "n": [2, 3, 5],
        }
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    keys = list(base.keys()) + list(algo_grid.keys())
    values: Iterable[Iterable[float]] = list(base.values()) + list(algo_grid.values())
    grid = []
    for combo in product(*values):
        grid.append(dict(zip(keys, combo)))
    return grid


def build_stage2_grid(algo: str, best_params: Dict[str, float]) -> List[Dict[str, float]]:
    alpha = float(best_params["algo_alpha"])
    lambda_cvar = float(best_params["lambda_cvar"])
    risk_window = int(best_params["risk_window"])

    alpha_candidates = sorted(set([max(1e-4, alpha * 0.5), alpha, alpha * 1.5]))
    lambda_candidates = sorted(set([max(0.0, lambda_cvar * 0.5), lambda_cvar, lambda_cvar * 1.5]))
    window_candidates = sorted(set([max(20, risk_window - 20), risk_window, risk_window + 20]))

    grid: List[Dict[str, float]] = []

    if algo == "q_learning":
        bins = int(best_params["bins"])
        bins_candidates = sorted(set([max(4, bins - 2), bins, bins + 2]))
        for combo in product(alpha_candidates, lambda_candidates, window_candidates, bins_candidates):
            grid.append(
                {
                    "algo_alpha": combo[0],
                    "lambda_cvar": combo[1],
                    "risk_window": combo[2],
                    "bins": combo[3],
                }
            )
    elif algo == "sg_sarsa":
        for combo in product(alpha_candidates, lambda_candidates, window_candidates):
            grid.append(
                {
                    "algo_alpha": combo[0],
                    "lambda_cvar": combo[1],
                    "risk_window": combo[2],
                }
            )
    elif algo == "nstep_sarsa":
        n = int(best_params["n"])
        n_candidates = sorted(set([max(1, n - 1), n, n + 1]))
        for combo in product(alpha_candidates, lambda_candidates, window_candidates, n_candidates):
            grid.append(
                {
                    "algo_alpha": combo[0],
                    "lambda_cvar": combo[1],
                    "risk_window": combo[2],
                    "n": combo[3],
                }
            )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    return grid


def apply_trial_params(config: Dict[str, Any], algo: str, params: Dict[str, float]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["env"]["lambda_cvar"] = float(params["lambda_cvar"])
    cfg["env"]["risk_window"] = int(params["risk_window"])

    if algo == "q_learning":
        cfg["q_learning"]["alpha"] = float(params["algo_alpha"])
        cfg["q_learning"]["bins"] = int(params["bins"])
    elif algo == "sg_sarsa":
        cfg["sg_sarsa"]["alpha"] = float(params["algo_alpha"])
    elif algo == "nstep_sarsa":
        cfg["nstep_sarsa"]["alpha"] = float(params["algo_alpha"])
        cfg["nstep_sarsa"]["n"] = int(params["n"])
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    return cfg


def run_trials(
    algo: str,
    config: Dict[str, Any],
    grid: List[Dict[str, float]],
    split,
    output_root: Path,
    seed: int,
    phase: str,
) -> List[TrialResult]:
    results: List[TrialResult] = []
    for idx, params in enumerate(grid, start=1):
        trial_cfg = apply_trial_params(config, algo=algo, params=params)
        run_dir = output_root / phase / f"trial_{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "trial_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(trial_cfg, f)

        metrics = run_training(
            algo=algo,
            config=trial_cfg,
            train_df=split.train,
            val_df=split.val,
            test_df=split.test,
            seed=seed,
            output_dir=run_dir,
        )

        results.append(
            TrialResult(
                phase=phase,
                trial_id=idx,
                params=params,
                val_total_reward=float(metrics["val"]["total_reward"]),
                val_cvar=float(metrics["val"]["cvar"]),
                train_seconds=float(metrics["training"]["total_train_seconds"]),
                run_dir=str(run_dir),
            )
        )

        print(
            f"[{phase}] trial {idx}/{len(grid)}: val_total_reward={results[-1].val_total_reward:.6f}, "
            f"val_cvar={results[-1].val_cvar:.6f}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage tuning for TD trading agents")
    parser.add_argument("--algo", required=True, choices=["q_learning", "sg_sarsa", "nstep_sarsa"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-root", default="runs/tuning")
    args = parser.parse_args()

    config = load_config(args.config)

    split = prepare_dataset_split(config)

    output_root = Path(args.output_root) / args.algo
    output_root.mkdir(parents=True, exist_ok=True)

    stage1_grid = build_stage1_grid(args.algo)
    stage1_results = run_trials(
        algo=args.algo,
        config=config,
        grid=stage1_grid,
        split=split,
        output_root=output_root,
        seed=args.seed,
        phase="stage1",
    )

    best_stage1 = max(stage1_results, key=lambda x: x.val_total_reward)
    stage2_grid = build_stage2_grid(args.algo, best_stage1.params)
    stage2_results = run_trials(
        algo=args.algo,
        config=config,
        grid=stage2_grid,
        split=split,
        output_root=output_root,
        seed=args.seed,
        phase="stage2",
    )

    all_results = stage1_results + stage2_results
    best = max(all_results, key=lambda x: x.val_total_reward)

    rows = []
    for r in all_results:
        row = {
            "phase": r.phase,
            "trial_id": r.trial_id,
            "val_total_reward": r.val_total_reward,
            "val_cvar": r.val_cvar,
            "train_seconds": r.train_seconds,
            "run_dir": r.run_dir,
        }
        row.update(r.params)
        rows.append(row)

    log_df = pd.DataFrame(rows)
    log_df.to_csv(output_root / "tuning_log.csv", index=False)

    best_cfg = apply_trial_params(config, args.algo, best.params)
    with open(output_root / "best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f)

    summary = {
        "algo": args.algo,
        "seed": args.seed,
        "best_val_total_reward": best.val_total_reward,
        "best_val_cvar": best.val_cvar,
        "best_params": best.params,
        "notes": [
            "Stage-1 used coarse grid search.",
            "Stage-2 refined around best Stage-1 parameters.",
            "Model selection used validation total reward only.",
        ],
    }
    with open(output_root / "tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best params: {best.params}")
    print(f"Saved tuning artifacts to: {output_root}")


if __name__ == "__main__":
    main()
