import numpy as np

from src.risk import compute_var_cvar


def test_compute_var_cvar_known_values() -> None:
    losses = np.array([0.01, 0.02, 0.03, 0.04, 0.10])
    var, cvar = compute_var_cvar(losses, alpha=0.8)

    assert np.isclose(var, np.quantile(losses, 0.8))
    tail = losses[losses >= var]
    assert np.isclose(cvar, np.mean(tail))


def test_compute_var_cvar_empty_input() -> None:
    var, cvar = compute_var_cvar([], alpha=0.95)
    assert var == 0.0
    assert cvar == 0.0
