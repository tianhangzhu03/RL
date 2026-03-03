# Risk-Sensitive Single-Asset Trading with TD Learning

Course project repository (EE473) for risk-sensitive reinforcement learning in single-asset trading.

## Canonical Deliverables

- Final report notebook: `report/final/report.ipynb`
- Final report PDF: `report/final/report.pdf`
- Final report figures: `report/final/figures/`
- Final report data tables: `report/final/data/`

## Project Structure

- `src/` : environment, agents, training/eval/tuning utilities
- `configs/` : active configs (`base.yaml`, `main_xlf.yaml`, `xlf_promotion_gate.yaml`)
- `experiments/` : compact experiment package (aggregated summaries + selected best configs)
- `video/` : reusable figures/data for presentation
- `tests/` : unit and smoke tests

## Environment

```bash
bash scripts/setup_venv.sh
scripts/py.sh -m pytest -q
```

## Export Final PDF

```bash
bash scripts/export_report_pdf.sh report/final/report.ipynb webpdf
```
