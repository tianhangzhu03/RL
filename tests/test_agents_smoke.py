import numpy as np

from src.agents.nstep_sarsa import NStepSarsaAgent
from src.agents.sg_sarsa import SemiGradientSarsaAgent
from src.agents.tabular_q import TabularQLearningAgent
from src.discretizer import StateDiscretizer


def test_tabular_q_update_smoke() -> None:
    states = np.array([[0.1, 0.2, 0.3, 0.0, 0.0], [0.2, 0.1, -0.1, 0.5, 0.5]])
    disc = StateDiscretizer(n_bins=4)
    disc.fit(states)

    agent = TabularQLearningAgent(action_dim=3, alpha=0.1, gamma=0.99, discretizer=disc)
    s = states[0]
    ns = states[1]
    before = agent.q_values(s).copy()
    agent.update(s, action=1, reward=0.5, next_state=ns, done=False)
    after = agent.q_values(s)

    assert not np.allclose(before, after)


def test_sg_sarsa_update_smoke() -> None:
    agent = SemiGradientSarsaAgent(state_dim=5, action_dim=3, alpha=0.01, gamma=0.99)
    s = np.array([0.1, 0.2, 0.3, 0.0, 0.0])
    ns = np.array([0.2, 0.1, -0.1, 0.5, 0.5])

    before = agent.weights.copy()
    agent.update(s, action=0, reward=0.2, next_state=ns, next_action=1, done=False)
    after = agent.weights

    assert not np.allclose(before, after)


def test_nstep_sarsa_update_episode_smoke() -> None:
    agent = NStepSarsaAgent(state_dim=5, action_dim=3, alpha=0.01, gamma=0.99, n=3)
    s0 = np.array([0.1, 0.2, 0.1, 0.0, 0.0])
    s1 = np.array([0.2, 0.1, 0.2, 0.5, 0.5])
    s2 = np.array([0.3, 0.1, 0.1, 1.0, 0.5])

    traj = [
        {"state": s0, "action": 0, "reward": 0.1, "next_state": s1, "next_action": 1, "done": False},
        {"state": s1, "action": 1, "reward": -0.2, "next_state": s2, "next_action": 2, "done": False},
        {"state": s2, "action": 2, "reward": 0.3, "next_state": s2, "next_action": None, "done": True},
    ]

    before = agent.weights.copy()
    agent.update_episode(traj)
    after = agent.weights

    assert not np.allclose(before, after)
