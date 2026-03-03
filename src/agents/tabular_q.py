"""Tabular Q-learning agent."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Sequence, Tuple

import numpy as np

from src.discretizer import StateDiscretizer


class TabularQLearningAgent:
    def __init__(
        self,
        action_dim: int,
        alpha: float,
        gamma: float,
        discretizer: StateDiscretizer,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.discretizer = discretizer
        self.rng = rng or np.random.default_rng()
        self.q_table: DefaultDict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_dim, dtype=float)
        )

    def _discrete_state(self, state: Sequence[float]) -> Tuple[int, ...]:
        return self.discretizer.transform(state)

    def q_values(self, state: Sequence[float]) -> np.ndarray:
        d_state = self._discrete_state(state)
        return self.q_table[d_state]

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
        done: bool,
    ) -> None:
        ds = self._discrete_state(state)
        dns = self._discrete_state(next_state)

        target = reward
        if not done:
            target += self.gamma * float(np.max(self.q_table[dns]))

        td_error = target - self.q_table[ds][action]
        self.q_table[ds][action] += self.alpha * td_error
