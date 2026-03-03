"""n-step SARSA agent with linear function approximation."""

from __future__ import annotations

from typing import Any, Dict, List

from src.agents.sg_sarsa import SemiGradientSarsaAgent


class NStepSarsaAgent(SemiGradientSarsaAgent):
    def __init__(self, *args, n: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n

    def update_episode(self, trajectory: List[Dict[str, Any]]) -> None:
        """Apply episodic n-step SARSA updates from a full trajectory."""
        t_max = len(trajectory)
        for t in range(t_max):
            n_end = min(t + self.n, t_max)
            discounted_sum = 0.0
            gamma_k = 1.0

            for k in range(t, n_end):
                reward = float(trajectory[k]["reward"])
                discounted_sum += gamma_k * reward
                gamma_k *= self.gamma

            # Bootstrap when trajectory has future state-action beyond n steps.
            if t + self.n < t_max:
                bootstrap_step = trajectory[t + self.n - 1]
                next_state = bootstrap_step["next_state"]
                next_action = bootstrap_step["next_action"]
                done = bool(bootstrap_step["done"])
                if not done and next_action is not None:
                    discounted_sum += gamma_k * self.q_value(next_state, int(next_action))

            state = trajectory[t]["state"]
            action = int(trajectory[t]["action"])
            q_sa = self.q_value(state, action)
            td_error = discounted_sum - q_sa
            self.weights += self.alpha * td_error * self.featurize(state, action)
