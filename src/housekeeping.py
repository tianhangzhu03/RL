"""Archive stale summary artifacts without touching training run directories.

Usage:
  scripts/py.sh -m src.housekeeping --runs-root runs --archive-root archive_runs
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive stale non-run artifacts")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--archive-root", default="archive_runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_legacy_summary_files(runs_root: Path) -> List[Path]:
    files = sorted(runs_root.glob("suite_summary_*.csv"))
    return [p for p in files if p.name != "suite_summary.csv"]


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    archive_root = Path(args.archive_root)

    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    stale = find_legacy_summary_files(runs_root)
    if not stale:
        print("No stale summary files found.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = archive_root / "summaries" / stamp

    print(f"Found {len(stale)} stale summary files.")
    for src in stale:
        dst = out_dir / src.name
        print(f"- {src} -> {dst}")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            src.rename(dst)

    if not args.dry_run:
        print(f"Archived files to: {out_dir}")


if __name__ == "__main__":
    main()
