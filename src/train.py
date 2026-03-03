"""Training entrypoint for risk-sensitive TD trading agents.

Usage:
    python -m src.train --algo q_learning --config configs/base.yaml --seed 11
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, cast

import numpy as np
import pandas as pd
import yaml

from src.agents.nstep_sarsa import NStepSarsaAgent
from src.agents.sg_sarsa import SemiGradientSarsaAgent
from src.agents.tabular_q import TabularQLearningAgent
from src.discretizer import StateDiscretizer
from src.env import EnvConfig, TradingEnv
from src.features import FEATURE_COLUMNS
from src.metrics import compute_performance_metrics
from src.pipeline import prepare_dataset_split


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_env_config(cfg: Mapping[str, Any]) -> EnvConfig:
    actions = [float(x) for x in cfg["actions"]]
    return EnvConfig(
        actions=actions,
        transaction_cost=float(cfg["transaction_cost"]),
        risk_window=int(cfg["risk_window"]),
        risk_alpha=float(cfg["risk_alpha"]),
        lambda_vol=float(cfg["lambda_vol"]),
        lambda_cvar=float(cfg["lambda_cvar"]),
        action_mode=str(cfg.get("action_mode", "target")),
        min_position=float(cfg.get("min_position", 0.0)),
        max_position=float(cfg.get("max_position", 1.0)),
        core_position=float(cfg.get("core_position", 0.0)),
        vol_target=float(cfg.get("vol_target", 0.015)),
        cvar_target=float(cfg.get("cvar_target", 0.03)),
        penalize_excess_risk_only=bool(cfg.get("penalize_excess_risk_only", True)),
        inactivity_penalty=float(cfg.get("inactivity_penalty", 0.0)),
        opportunity_cost_coeff=float(cfg.get("opportunity_cost_coeff", 0.0)),
        active_benchmark_coeff=float(cfg.get("active_benchmark_coeff", 0.0)),
    )


def make_agent(
    algo: str,
    config: Mapping[str, Any],
    state_dim: int,
    action_dim: int,
    train_states: np.ndarray,
    rng: np.random.Generator,
) -> TabularQLearningAgent | SemiGradientSarsaAgent | NStepSarsaAgent:
    gamma = config["training"]["gamma"]

    if algo == "q_learning":
        bins = int(config["q_learning"]["bins"])
        discretizer = StateDiscretizer(n_bins=bins)
        discretizer.fit(train_states)
        return TabularQLearningAgent(
            action_dim=action_dim,
            alpha=float(config["q_learning"]["alpha"]),
            gamma=float(gamma),
            discretizer=discretizer,
            rng=rng,
        )

    if algo == "sg_sarsa":
        return SemiGradientSarsaAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=float(config["sg_sarsa"]["alpha"]),
            gamma=float(gamma),
            rng=rng,
        )

    if algo == "nstep_sarsa":
        return NStepSarsaAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=float(config["nstep_sarsa"]["alpha"]),
            gamma=float(gamma),
            rng=rng,
            n=int(config["nstep_sarsa"]["n"]),
        )

    raise ValueError(f"Unsupported algorithm: {algo}")


def evaluate_policy(
    env: TradingEnv,
    policy_fn: Callable[[np.ndarray], int],
    seed: int = 0,
) -> Dict[str, float]:
    state = env.reset(seed=seed)
    done = False
    rewards: List[float] = []
    portfolio_returns: List[float] = []
    positions: List[float] = []
    turnover_total = 0.0
    trade_steps = 0.0
    prev_position = 0.0
    total_transaction_cost = 0.0
    total_opportunity_cost = 0.0
    total_inactivity_penalty = 0.0
    total_volatility_penalty_raw = 0.0
    total_cvar_penalty_raw = 0.0
    total_active_return_vs_benchmark = 0.0
    total_active_benchmark_bonus = 0.0

    while not done:
        action = policy_fn(state)
        next_state, reward, done, info = env.step(action)
        rewards.append(float(reward))
        portfolio_returns.append(float(info["portfolio_return"]))
        position = float(info.get("position", 0.0))
        positions.append(position)
        step_turnover = abs(position - prev_position)
        turnover_total += float(step_turnover)
        if step_turnover > 1e-12:
            trade_steps += 1.0
        prev_position = position
        total_transaction_cost += float(info.get("transaction_cost", 0.0))
        total_opportunity_cost += float(info.get("opportunity_cost", 0.0))
        total_inactivity_penalty += float(info.get("inactivity_penalty", 0.0))
        total_volatility_penalty_raw += float(info.get("volatility_penalty", 0.0))
        total_cvar_penalty_raw += float(info.get("cvar_penalty", 0.0))
        total_active_return_vs_benchmark += float(info.get("active_return_vs_benchmark", 0.0))
        total_active_benchmark_bonus += float(info.get("active_benchmark_bonus", 0.0))
        state = next_state

    metrics = compute_performance_metrics(portfolio_returns, alpha=env.config.risk_alpha)
    metrics["avg_reward"] = float(np.mean(rewards)) if rewards else 0.0
    metrics["total_reward"] = float(np.sum(rewards)) if rewards else 0.0
    metrics["num_steps"] = len(rewards)
    pos_arr = np.asarray(positions, dtype=float) if positions else np.asarray([], dtype=float)
    max_position = float(env.config.max_position)
    metrics["avg_position"] = float(np.mean(pos_arr)) if pos_arr.size else 0.0
    metrics["position_std"] = float(np.std(pos_arr, ddof=0)) if pos_arr.size else 0.0
    metrics["cash_ratio"] = float(np.mean(np.isclose(pos_arr, 0.0))) if pos_arr.size else 0.0
    metrics["full_position_ratio"] = float(np.mean(np.isclose(pos_arr, max_position))) if pos_arr.size else 0.0
    metrics["turnover_total"] = float(turnover_total)
    metrics["turnover_per_step"] = float(turnover_total / len(rewards)) if rewards else 0.0
    metrics["trade_steps"] = float(trade_steps)
    metrics["trade_step_ratio"] = float(trade_steps / len(rewards)) if rewards else 0.0
    metrics["total_transaction_cost"] = float(total_transaction_cost)
    metrics["total_opportunity_cost"] = float(total_opportunity_cost)
    metrics["total_inactivity_penalty"] = float(total_inactivity_penalty)
    metrics["total_volatility_penalty_raw"] = float(total_volatility_penalty_raw)
    metrics["total_cvar_penalty_raw"] = float(total_cvar_penalty_raw)
    metrics["total_volatility_penalty_weighted"] = float(env.config.lambda_vol * total_volatility_penalty_raw)
    metrics["total_cvar_penalty_weighted"] = float(env.config.lambda_cvar * total_cvar_penalty_raw)
    metrics["total_active_return_vs_benchmark"] = float(total_active_return_vs_benchmark)
    metrics["total_active_benchmark_bonus"] = float(total_active_benchmark_bonus)
    return metrics


def evaluate_baselines(env: TradingEnv, seed: int = 0) -> Dict[str, Dict[str, float]]:
    max_position_action = int(np.argmax(env.actions))
    zero_action = next((i for i, a in enumerate(env.actions) if np.isclose(float(a), 0.0)), 0)
    rng = np.random.default_rng(seed)

    buy_hold_delta_state = {"position": float(env.config.min_position)}

    def buy_hold_policy(_state: np.ndarray) -> int:
        if env.config.action_mode == "delta":
            if buy_hold_delta_state["position"] + 1e-12 < float(env.config.max_position):
                action_idx = max_position_action
                buy_hold_delta_state["position"] = float(
                    min(
                        float(env.config.max_position),
                        buy_hold_delta_state["position"] + float(env.actions[action_idx]),
                    )
                )
                return action_idx
            return zero_action
        return max_position_action

    def random_policy(_state: np.ndarray) -> int:
        return int(rng.integers(0, env.action_dim))

    return {
        "buy_hold": evaluate_policy(env, buy_hold_policy, seed=seed),
        "random": evaluate_policy(env, random_policy, seed=seed + 1),
    }


def find_convergence_episode(rewards: List[float], threshold: float, window: int) -> int | None:
    if window <= 0 or len(rewards) < window:
        return None

    arr = np.asarray(rewards, dtype=float)
    moving_avg = np.convolve(arr, np.ones(window) / window, mode="valid")
    for i, value in enumerate(moving_avg):
        if value >= threshold:
            return i + window
    return None


def run_training(
    algo: str,
    config: Mapping[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    output_dir: Path,
    evaluate_test: bool = True,
) -> Dict[str, Any]:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    env_cfg = to_env_config(config["env"])
    train_env = TradingEnv(train_df, env_cfg)
    val_env = TradingEnv(val_df, env_cfg)
    test_env = TradingEnv(test_df, env_cfg)

    train_feature_states = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    train_states = np.concatenate(
        [
            train_feature_states,
            np.zeros((len(train_feature_states), 2), dtype=float),
        ],
        axis=1,
    )

    agent = make_agent(
        algo=algo,
        config=config,
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        train_states=train_states,
        rng=rng,
    )
    q_agent = cast(TabularQLearningAgent | None, agent if algo == "q_learning" else None)
    sg_agent = cast(SemiGradientSarsaAgent | None, agent if algo == "sg_sarsa" else None)
    nstep_agent = cast(NStepSarsaAgent | None, agent if algo == "nstep_sarsa" else None)

    episodes = int(config["training"]["episodes"])
    epsilon = float(config["training"]["epsilon_start"])
    epsilon_min = float(config["training"]["epsilon_min"])
    epsilon_decay = float(config["training"]["epsilon_decay"])
    risk_warmup_episodes = int(config["training"].get("risk_warmup_episodes", 0))
    base_lambda_cvar = float(env_cfg.lambda_cvar)

    history: List[Dict[str, float]] = []
    episode_rewards: List[float] = []
    total_steps = 0

    start = time.perf_counter()

    for episode in range(1, episodes + 1):
        if risk_warmup_episodes > 0:
            warmup_scale = min(1.0, episode / risk_warmup_episodes)
            train_env.config.lambda_cvar = base_lambda_cvar * warmup_scale
        else:
            train_env.config.lambda_cvar = base_lambda_cvar

        state = train_env.reset(seed=seed + episode)
        action = agent.act(state, epsilon)
        done = False

        ep_reward = 0.0
        ep_steps = 0
        trajectory: List[Dict[str, Any]] = []

        while not done:
            next_state, reward, done, _ = train_env.step(action)
            ep_reward += reward
            ep_steps += 1
            total_steps += 1

            if algo == "q_learning":
                assert q_agent is not None
                q_agent.update(state, action, reward, next_state, done)
                state = next_state
                if not done:
                    action = q_agent.act(state, epsilon)
                continue

            assert sg_agent is not None or nstep_agent is not None
            active_agent = sg_agent if sg_agent is not None else nstep_agent
            next_action = active_agent.act(next_state, epsilon) if not done else None
            if algo == "sg_sarsa":
                assert sg_agent is not None
                sg_agent.update(state, action, reward, next_state, next_action, done)
            else:
                trajectory.append(
                    {
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "next_action": next_action,
                        "done": done,
                    }
                )

            state = next_state
            if next_action is not None:
                action = next_action

        if algo == "nstep_sarsa" and trajectory:
            assert nstep_agent is not None
            nstep_agent.update_episode(trajectory)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(ep_reward)

        history.append(
            {
                "episode": episode,
                "episode_reward": ep_reward,
                "epsilon": epsilon,
                "lambda_cvar": float(train_env.config.lambda_cvar),
                "steps": ep_steps,
            }
        )

    total_train_seconds = time.perf_counter() - start
    time_per_100k_steps = total_train_seconds / max(total_steps, 1) * 100000

    threshold = float(config["training"]["convergence_reward_threshold"])
    window = int(config["training"]["convergence_window"])
    convergence_episode = find_convergence_episode(episode_rewards, threshold, window)

    policy_agent: TabularQLearningAgent | SemiGradientSarsaAgent | NStepSarsaAgent
    if algo == "q_learning":
        assert q_agent is not None
        policy_agent = q_agent
    elif algo == "sg_sarsa":
        assert sg_agent is not None
        policy_agent = sg_agent
    else:
        assert nstep_agent is not None
        policy_agent = nstep_agent

    learned_policy: Callable[[np.ndarray], int] = (
        lambda state: policy_agent.act(state, epsilon=0.0)
    )

    val_metrics = evaluate_policy(val_env, learned_policy, seed=seed)
    if evaluate_test:
        test_metrics = evaluate_policy(test_env, learned_policy, seed=seed)
        baseline_metrics = evaluate_baselines(test_env, seed=seed)
    else:
        test_metrics = {}
        baseline_metrics = {}

    training_metrics = {
        "total_train_seconds": float(total_train_seconds),
        "total_steps": int(total_steps),
        "time_per_100k_steps": float(time_per_100k_steps),
        "convergence_episode": int(convergence_episode) if convergence_episode is not None else None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)
    with open(output_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    if evaluate_test:
        with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(baseline_metrics, f, indent=2)

    return {
        "training": training_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "baseline": baseline_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train risk-sensitive TD trading agents")
    parser.add_argument("--algo", required=True, choices=["q_learning", "sg_sarsa", "nstep_sarsa"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--disable-cvar", action="store_true", help="Set lambda_cvar to 0 for ablation")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.disable_cvar:
        config["env"]["lambda_cvar"] = 0.0

    split = prepare_dataset_split(config)

    seeds = config["training"]["seeds"] if args.all_seeds else [args.seed if args.seed is not None else config["training"]["seeds"][0]]

    output_root = Path(args.output_root)
    tag = "no_cvar" if args.disable_cvar else "with_cvar"

    for seed in seeds:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / args.algo / tag / f"seed_{seed}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

        run_training(
            algo=args.algo,
            config=config,
            train_df=split.train,
            val_df=split.val,
            test_df=split.test,
            seed=int(seed),
            output_dir=run_dir,
        )

        print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
