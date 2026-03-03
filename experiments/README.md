# Experiments Package

This folder is the compact, post-cleanup experiment artifact set.

## Structure

- `core/summary/xlf_suite_summaries.csv`
  - merged suite-level outputs for mainline XLF rounds
- `core/summary/xlf_eval_agg.csv`
  - aggregated evaluation snapshots (XLF)
- `core/summary/xlf_tuning_best_trials.csv`
  - flattened best-trial records from tuning runs
- `core/summary/report_core_tables_long.csv`
  - long-format merge of core report tables
- `core/best_configs/*.yaml`
  - selected best configs retained for reproducibility

## Notes

- Raw historical `runs*` folders were pruned after extracting core summaries.
- This package is intended for report/preparation and quick reproducibility checks.
