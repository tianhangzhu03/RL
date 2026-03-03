import numpy as np
import pandas as pd

from src.env import EnvConfig, TradingEnv


def build_dummy_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=8, freq="D"),
            "Close": [100, 101, 102, 101, 103, 104, 105, 106],
            "asset_return": [0.0, 0.01, 0.0099, -0.0098, 0.0198, 0.0097, 0.0096, 0.0095],
            "rolling_mean_return": [0.0, 0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01],
            "rolling_vol": [0.01] * 8,
            "momentum": [0.0, 0.01, 0.02, -0.01, 0.02, 0.03, 0.03, 0.03],
        }
    )


def test_env_step_shapes_and_reward_no_penalty() -> None:
    cfg = EnvConfig(
        actions=[0.0, 0.5, 1.0],
        transaction_cost=0.0,
        risk_window=3,
        risk_alpha=0.95,
        lambda_vol=0.0,
        lambda_cvar=0.0,
    )
    env = TradingEnv(build_dummy_data(), cfg)

    state = env.reset(seed=123)
    assert state.shape[0] == env.state_dim

    next_state, reward, done, info = env.step(2)
    assert next_state.shape[0] == env.state_dim
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert np.isclose(reward, info["portfolio_return"])


def test_env_transaction_cost_is_applied() -> None:
    cfg = EnvConfig(
        actions=[0.0, 0.5, 1.0],
        transaction_cost=0.01,
        risk_window=3,
        risk_alpha=0.95,
        lambda_vol=0.0,
        lambda_cvar=0.0,
    )
    env = TradingEnv(build_dummy_data(), cfg)
    env.reset(seed=123)

    _, reward, _, info = env.step(2)
    expected = info["portfolio_return"] - info["transaction_cost"]
    assert np.isclose(reward, expected)


def test_env_excess_risk_penalty_behavior() -> None:
    cfg = EnvConfig(
        actions=[0.0, 0.5, 1.0],
        transaction_cost=0.0,
        risk_window=3,
        risk_alpha=0.95,
        lambda_vol=1.0,
        lambda_cvar=1.0,
        vol_target=10.0,
        cvar_target=10.0,
        penalize_excess_risk_only=True,
        inactivity_penalty=0.0,
    )
    env = TradingEnv(build_dummy_data(), cfg)
    env.reset(seed=1)
    _, reward, _, info = env.step(2)

    # Targets are intentionally high, so excess-risk penalties should be zero.
    assert np.isclose(info["volatility_penalty"], 0.0)
    assert np.isclose(info["cvar_penalty"], 0.0)
    assert np.isclose(reward, info["portfolio_return"])


def test_env_opportunity_cost_penalizes_underexposure() -> None:
    cfg = EnvConfig(
        actions=[0.0, 0.5, 1.0],
        transaction_cost=0.0,
        risk_window=3,
        risk_alpha=0.95,
        lambda_vol=0.0,
        lambda_cvar=0.0,
        opportunity_cost_coeff=1.0,
    )
    env = TradingEnv(build_dummy_data(), cfg)
    env.reset(seed=2)

    # action 0 -> position 0.0, so full opportunity-cost on positive asset return.
    _, reward, _, info = env.step(0)
    assert info["asset_return"] > 0.0
    expected = info["portfolio_return"] - info["opportunity_cost"]
    assert np.isclose(reward, expected)
    assert info["opportunity_cost"] > 0.0


def test_env_delta_action_mode_updates_position_incrementally_and_clips() -> None:
    cfg = EnvConfig(
        actions=[-0.25, 0.0, 0.25],
        transaction_cost=0.0,
        risk_window=3,
        risk_alpha=0.95,
        lambda_vol=0.0,
        lambda_cvar=0.0,
        action_mode="delta",
        min_position=0.0,
        max_position=1.0,
    )
    env = TradingEnv(build_dummy_data(), cfg)
    env.reset(seed=3)

    # Increment from 0.0 -> 0.25
    _, reward1, _, info1 = env.step(2)
    assert np.isclose(info1["position"], 0.25)
    assert np.isclose(reward1, info1["portfolio_return"])

    # Decrement attempts below 0.0 should clip at 0.0
    env.current_position = 0.1
    _, _, _, info2 = env.step(0)
    assert np.isclose(info2["position"], 0.0)

    # Increment from near max should clip at 1.0
    env.current_position = 0.9
    _, _, _, info3 = env.step(2)
    assert np.isclose(info3["position"], 1.0)


def test_env_overlay_target_action_mode_uses_core_position_and_clips() -> None:
    cfg = EnvConfig(
        actions=[-0.25, 0.0, 0.25],
        transaction_cost=0.0,
        risk_window=3,
        risk_alpha=0.95,
        lambda_vol=0.0,
        lambda_cvar=0.0,
        action_mode="overlay_target",
        min_position=0.0,
        max_position=1.0,
        core_position=0.75,
    )
    env = TradingEnv(build_dummy_data(), cfg)
    env.reset(seed=4)

    # Overlay 0.0 -> core position
    _, reward0, _, info0 = env.step(1)
    assert np.isclose(info0["position"], 0.75)
    assert np.isclose(info0["overlay_value"], 0.0)
    assert np.isclose(reward0, info0["portfolio_return"])

    env.current_position = 0.9
    # Overlay +0.25 should target 1.0 after clipping (0.75 + 0.25 = 1.0)
    _, _, _, info1 = env.step(2)
    assert np.isclose(info1["position"], 1.0)
    assert np.isclose(info1["overlay_value"], 0.25)

    # Change core and verify lower clip case through direct config replacement
    env.config.core_position = 0.1
    env.current_position = 0.1
    _, _, _, info2 = env.step(0)
    assert np.isclose(info2["position"], 0.0)
