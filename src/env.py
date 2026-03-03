"""Single-asset trading environment with risk-sensitive reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.risk import compute_var_cvar


@dataclass
class EnvConfig:
    actions: Sequence[float]
    transaction_cost: float
    risk_window: int
    risk_alpha: float
    lambda_vol: float
    lambda_cvar: float
    action_mode: str = "target"  # "target", "delta", or "overlay_target" (core_position + action)
    min_position: float = 0.0
    max_position: float = 1.0
    core_position: float = 0.0
    vol_target: float = 0.015
    cvar_target: float = 0.03
    penalize_excess_risk_only: bool = True
    inactivity_penalty: float = 0.0
    opportunity_cost_coeff: float = 0.0
    active_benchmark_coeff: float = 0.0


class TradingEnv:
    """Episodic environment over chronological daily observations."""

    def __init__(self, data: pd.DataFrame, config: EnvConfig) -> None:
        if len(data) < 3:
            raise ValueError("Environment requires at least 3 data points")

        self.data = data.reset_index(drop=True)
        self.config = config
        self.actions = np.asarray(config.actions, dtype=float)
        if config.action_mode not in {"target", "delta", "overlay_target"}:
            raise ValueError(f"Unsupported action_mode: {config.action_mode}")
        if config.min_position > config.max_position:
            raise ValueError("min_position cannot exceed max_position")
        if not (config.min_position <= config.core_position <= config.max_position):
            raise ValueError("core_position must lie within [min_position, max_position]")

        missing_features = [c for c in FEATURE_COLUMNS if c not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if "asset_return" not in self.data.columns:
            raise ValueError("Data must include asset_return column")

        self.feature_matrix = self.data[FEATURE_COLUMNS].to_numpy(dtype=float)
        self.asset_returns = self.data["asset_return"].to_numpy(dtype=float)

        self.max_index = len(self.data) - 2
        self.current_index = 0
        self.current_position = 0.0
        self.prev_position = 0.0
        self.portfolio_returns_history: List[float] = []

    @property
    def state_dim(self) -> int:
        return len(FEATURE_COLUMNS) + 2

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        self.current_index = 0
        self.current_position = 0.0
        self.prev_position = 0.0
        self.portfolio_returns_history = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        features = self.feature_matrix[self.current_index]
        recent_trade_signal = self.current_position - self.prev_position
        return np.concatenate([features, [self.current_position, recent_trade_signal]]).astype(float)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if action < 0 or action >= self.action_dim:
            raise ValueError(f"Invalid action {action}")

        action_value = float(self.actions[action])
        if self.config.action_mode == "delta":
            target_position = float(
                np.clip(
                    self.current_position + action_value,
                    self.config.min_position,
                    self.config.max_position,
                )
            )
        elif self.config.action_mode == "overlay_target":
            target_position = float(
                np.clip(
                    self.config.core_position + action_value,
                    self.config.min_position,
                    self.config.max_position,
                )
            )
        else:
            target_position = float(np.clip(action_value, self.config.min_position, self.config.max_position))
        transaction_cost = self.config.transaction_cost * abs(target_position - self.current_position)

        # Action selected at t is applied to return at t+1.
        asset_ret = float(self.asset_returns[self.current_index + 1])
        portfolio_ret = target_position * asset_ret
        self.portfolio_returns_history.append(portfolio_ret)

        risk_slice = self.portfolio_returns_history[-self.config.risk_window :]
        raw_vol = float(np.std(risk_slice, ddof=0)) if risk_slice else 0.0
        _, raw_cvar = compute_var_cvar([-x for x in risk_slice], alpha=self.config.risk_alpha)

        if self.config.penalize_excess_risk_only:
            volatility_penalty = max(0.0, raw_vol - self.config.vol_target)
            cvar_penalty = max(0.0, raw_cvar - self.config.cvar_target)
        else:
            volatility_penalty = raw_vol
            cvar_penalty = raw_cvar

        inactivity_penalty = self.config.inactivity_penalty if np.isclose(target_position, 0.0) else 0.0
        opportunity_cost = self.config.opportunity_cost_coeff * max(asset_ret, 0.0) * (1.0 - target_position)
        benchmark_return = asset_ret  # buy&hold with full long exposure
        active_return_vs_benchmark = portfolio_ret - benchmark_return
        active_benchmark_bonus = self.config.active_benchmark_coeff * active_return_vs_benchmark

        reward = (
            portfolio_ret
            - transaction_cost
            - self.config.lambda_vol * volatility_penalty
            - self.config.lambda_cvar * cvar_penalty
            - inactivity_penalty
            - opportunity_cost
            + active_benchmark_bonus
        )

        self.prev_position = self.current_position
        self.current_position = target_position
        self.current_index += 1

        done = self.current_index >= self.max_index
        next_state = self._get_state()

        info = {
            "asset_return": asset_ret,
            "portfolio_return": portfolio_ret,
            "transaction_cost": float(transaction_cost),
            "volatility_penalty": float(volatility_penalty),
            "cvar_penalty": float(cvar_penalty),
            "raw_volatility": float(raw_vol),
            "raw_cvar": float(raw_cvar),
            "inactivity_penalty": float(inactivity_penalty),
            "opportunity_cost": float(opportunity_cost),
            "benchmark_return": float(benchmark_return),
            "active_return_vs_benchmark": float(active_return_vs_benchmark),
            "active_benchmark_bonus": float(active_benchmark_bonus),
            "action_value": float(action_value),
            "core_position": float(self.config.core_position),
            "overlay_value": float(target_position - self.config.core_position),
            "position": float(self.current_position),
        }
        return next_state, float(reward), done, info
