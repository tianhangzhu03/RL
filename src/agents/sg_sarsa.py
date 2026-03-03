"""Semi-gradient SARSA agent with linear action-value approximation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class SemiGradientSarsaAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float,
        gamma: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.rng = rng or np.random.default_rng()

        # +1 for bias feature.
        self.base_feature_dim = self.state_dim + 1
        self.weight_dim = self.base_feature_dim * self.action_dim
        # Small random init prevents deterministic collapse at early training.
        self.weights = self.rng.normal(loc=0.0, scale=1e-4, size=self.weight_dim).astype(float)

    def _state_features(self, state: Sequence[float]) -> np.ndarray:
        state_arr = np.asarray(state, dtype=float)
        return np.concatenate([state_arr, np.array([1.0], dtype=float)])

    def featurize(self, state: Sequence[float], action: int) -> np.ndarray:
        x = np.zeros(self.weight_dim, dtype=float)
        start = action * self.base_feature_dim
        end = start + self.base_feature_dim
        x[start:end] = self._state_features(state)
        return x

    def q_value(self, state: Sequence[float], action: int) -> float:
        return float(np.dot(self.weights, self.featurize(state, action)))

    def q_values(self, state: Sequence[float]) -> np.ndarray:
        return np.array([self.q_value(state, a) for a in range(self.action_dim)], dtype=float)

    def _argmax_with_random_tie(self, values: np.ndarray) -> int:
        """Return argmax index while breaking ties uniformly at random."""
        max_value = float(np.max(values))
        candidates = np.flatnonzero(np.isclose(values, max_value))
        return int(self.rng.choice(candidates))

    def act(self, state: Sequence[float], epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_dim))
        return self._argmax_with_random_tie(self.q_values(state))

    def update(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        next_action: int | None,
        done: bool,
    ) -> None:
        phi = self.featurize(state, action)
        q_sa = float(np.dot(self.weights, phi))

        target = reward
        if not done and next_action is not None:
            target += self.gamma * self.q_value(next_state, next_action)

        td_error = target - q_sa
        self.weights += self.alpha * td_error * phi
