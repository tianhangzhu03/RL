"""State discretization utilities for tabular methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class StateDiscretizer:
    n_bins: int

    def __post_init__(self) -> None:
        self.bin_edges: List[np.ndarray] = []

    def fit(self, states: np.ndarray) -> None:
        """Fit per-dimension bin edges using min/max values in states."""
        if states.ndim != 2:
            raise ValueError("states must be a 2D array")

        self.bin_edges = []
        for col in range(states.shape[1]):
            column = states[:, col]
            min_v = float(np.min(column))
            max_v = float(np.max(column))
            if np.isclose(min_v, max_v):
                edges = np.array([min_v], dtype=float)
            else:
                edges = np.linspace(min_v, max_v, self.n_bins + 1)[1:-1]
            self.bin_edges.append(edges)

    def transform(self, state: Sequence[float]) -> Tuple[int, ...]:
        """Convert a continuous state vector into discrete bin indices."""
        if not self.bin_edges:
            raise RuntimeError("Discretizer is not fitted")

        state_arr = np.asarray(state, dtype=float)
        if state_arr.shape[0] != len(self.bin_edges):
            raise ValueError("State dimensionality mismatch")

        indices = [int(np.digitize(state_arr[i], self.bin_edges[i])) for i in range(len(self.bin_edges))]
        return tuple(indices)
