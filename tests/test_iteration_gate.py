import pandas as pd

from src.iteration_gate import PromotionThresholds, compute_promotion_decision


def test_promotion_decision_requires_reward_return_and_risk_bounds() -> None:
    thresholds = PromotionThresholds(
        require_reward_gt_baseline=True,
        require_return_ge_baseline=True,
        max_cvar_regression=0.002,
        max_mdd_regression=0.03,
        require_positive_lambda_cvar=True,
    )

    baseline = {"reward": 1.0, "return": 1.0, "cvar": 0.02, "mdd": -0.20, "train_seconds": 10.0}
    candidate = {"reward": 1.05, "return": 1.00, "cvar": 0.021, "mdd": -0.22, "train_seconds": 11.0}

    decision = compute_promotion_decision(
        baseline_means=baseline,
        candidate_means=candidate,
        thresholds=thresholds,
        tuned_lambda_cvar=0.005,
    )
    assert decision["promote"] is True

    # Exceeding CVaR tolerance should fail promotion.
    candidate_bad_cvar = dict(candidate)
    candidate_bad_cvar["cvar"] = 0.0235
    decision_bad = compute_promotion_decision(
        baseline_means=baseline,
        candidate_means=candidate_bad_cvar,
        thresholds=thresholds,
        tuned_lambda_cvar=0.005,
    )
    assert decision_bad["promote"] is False
    assert decision_bad["checks"]["cvar_regression_within_tolerance"] is False


def test_promotion_decision_fails_if_lambda_cvar_not_positive_when_required() -> None:
    thresholds = PromotionThresholds(require_positive_lambda_cvar=True)
    baseline = {"reward": 0.5, "return": 0.5, "cvar": 0.02, "mdd": -0.2, "train_seconds": 10.0}
    candidate = {"reward": 0.6, "return": 0.6, "cvar": 0.021, "mdd": -0.205, "train_seconds": 11.0}
    decision = compute_promotion_decision(
        baseline_means=baseline,
        candidate_means=candidate,
        thresholds=thresholds,
        tuned_lambda_cvar=0.0,
    )
    assert decision["promote"] is False
    assert decision["checks"]["positive_lambda_cvar"] is False


def test_promotion_decision_supports_primary_objective_tolerance_and_seed_consistency() -> None:
    thresholds = PromotionThresholds(
        # disable legacy dual-hard checks in favor of primary/secondary logic
        require_reward_gt_baseline=False,
        require_return_ge_baseline=False,
        primary_metric="return",
        primary_min_delta=-0.002,
        secondary_metric="reward",
        secondary_min_delta=-0.01,
        seed_consistency_metric="return",
        seed_min_nonworse_count=2,
        seed_delta_tolerance=-0.03,
        max_cvar_regression=0.004,
        max_mdd_regression=0.05,
        require_positive_lambda_cvar=True,
    )
    baseline = {"reward": 0.5, "return": 0.7, "cvar": 0.02, "mdd": -0.16, "train_seconds": 10.0}
    candidate = {"reward": 0.495, "return": 0.699, "cvar": 0.022, "mdd": -0.19, "train_seconds": 11.0}

    baseline_rows = pd.DataFrame(
        {
            "seed": [11, 22, 33],
            "test_total_reward": [0.40, 0.50, 0.60],
            "test_cumulative_return": [0.60, 0.70, 0.80],
            "test_cvar": [0.02, 0.02, 0.02],
            "test_max_drawdown": [-0.16, -0.16, -0.16],
            "train_seconds": [10.0, 10.0, 10.0],
        }
    )
    candidate_rows = pd.DataFrame(
        {
            "seed": [11, 22, 33],
            "test_total_reward": [0.39, 0.49, 0.61],  # mean delta ~0.0 (within reward tolerance)
            "test_cumulative_return": [0.58, 0.71, 0.79],  # deltas [-0.02, +0.01, -0.01], 3 seeds >= -0.03
            "test_cvar": [0.022, 0.022, 0.022],
            "test_max_drawdown": [-0.18, -0.19, -0.20],
            "train_seconds": [11.0, 11.0, 11.0],
        }
    )

    decision = compute_promotion_decision(
        baseline_means=baseline,
        candidate_means=candidate,
        thresholds=thresholds,
        tuned_lambda_cvar=0.005,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
    )
    assert decision["checks"]["primary_metric_within_tolerance"] is True
    assert decision["checks"]["secondary_metric_within_tolerance"] is True
    assert decision["checks"]["seed_consistency"] is True
    assert decision["promote"] is True
