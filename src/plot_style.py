"""Shared plotting style helpers for report-quality figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns


REPORT_COLORS = {
    "blue": "#1F4E79",
    "blue_light": "#5B84B1",
    "orange": "#D77A2B",
    "green": "#2E7D5A",
    "red": "#B24A3A",
    "ink": "#243447",
    "grid": "#D7DEE8",
}


def set_report_theme() -> None:
    """Apply a clean, paper-friendly plotting theme."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C9D2DE",
            "axes.labelcolor": REPORT_COLORS["ink"],
            "axes.titlecolor": REPORT_COLORS["ink"],
            "xtick.color": REPORT_COLORS["ink"],
            "ytick.color": REPORT_COLORS["ink"],
            "grid.color": REPORT_COLORS["grid"],
            "grid.alpha": 0.30,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#CDD6E3",
            "savefig.facecolor": "white",
        }
    )


def finish_figure(
    fig: plt.Figure,
    path: Path,
    *,
    tight_rect: tuple[float, float, float, float] | None = None,
    dpi: int = 220,
) -> None:
    """Finalize layout and save without clipping titles/legends."""
    if tight_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_rect)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def unique_legend(handles: Iterable[object], labels: Iterable[str]) -> tuple[list[object], list[str]]:
    """Deduplicate legend entries while preserving order."""
    seen: set[str] = set()
    out_h: list[object] = []
    out_l: list[str] = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        out_h.append(h)
        out_l.append(l)
    return out_h, out_l
