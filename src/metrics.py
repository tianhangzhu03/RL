"""Evaluation and diagnostics metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from src.risk import compute_var_cvar


def compute_max_drawdown(returns: Sequence[float]) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0

    curve = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(curve)
    drawdowns = curve / running_max - 1.0
    return float(np.min(drawdowns))


def compute_performance_metrics(returns: Iterable[float], alpha: float = 0.95) -> Dict[str, float]:
    ret = np.asarray(list(returns), dtype=float)
    if ret.size == 0:
        return {
            "cumulative_return": 0.0,
            "mean_return": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "var": 0.0,
            "cvar": 0.0,
        }

    cumulative_return = float(np.prod(1.0 + ret) - 1.0)
    mean_return = float(np.mean(ret))
    volatility = float(np.std(ret, ddof=0))
    sharpe = float(np.sqrt(252.0) * mean_return / volatility) if volatility > 1e-12 else 0.0
    max_dd = compute_max_drawdown(ret)
    var, cvar = compute_var_cvar(-ret, alpha=alpha)

    return {
        "cumulative_return": cumulative_return,
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "var": float(var),
        "cvar": float(cvar),
    }
