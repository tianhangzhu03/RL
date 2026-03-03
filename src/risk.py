"""Risk utility functions for VaR and CVaR computation."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def compute_var_cvar(losses: Iterable[float], alpha: float = 0.95) -> Tuple[float, float]:
    """Compute historical Value-at-Risk and Conditional Value-at-Risk.

    Args:
        losses: Sequence of loss values. Higher values imply worse outcomes.
        alpha: Tail probability level in (0, 1).

    Returns:
        (var, cvar) pair. Returns (0.0, 0.0) for empty input.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    losses_array = np.asarray(list(losses), dtype=float)
    if losses_array.size == 0:
        return 0.0, 0.0

    var_value = float(np.quantile(losses_array, alpha))
    tail_losses = losses_array[losses_array >= var_value]

    if tail_losses.size == 0:
        return var_value, var_value

    cvar_value = float(np.mean(tail_losses))
    return var_value, cvar_value
