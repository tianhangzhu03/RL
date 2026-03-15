# Risk-Sensitive Single-Asset Trading with TD Learning

EE473 course project repository for risk-sensitive reinforcement learning in single-asset trading.

## Project Goal

The project compares three temporal-difference (TD) methods under the same environment and promotion gate:

1. Tabular Q-learning
2. Semi-gradient Sarsa (linear function approximation)
3. n-step Sarsa (linear function approximation)

The objective is not unconstrained return maximization. The objective is policy selection under explicit downside-risk constraints.

## Main Setup

- Primary asset: `XLF`
- Action design: long-only position sizing with delta actions (configured in `configs/main_xlf.yaml`)
- Reward components: portfolio return, transaction cost, volatility/CVaR penalties, inactivity/opportunity cost terms
- Evaluation focus: reward/return plus risk metrics (`CVaR`, `MDD`) and training-time behavior

## Canonical Deliverables

- Final report notebook: `report/final/report.ipynb`
- Final report PDF: `report/final/report.pdf`
- Final report figures: `report/final/figures/`
- Final report tables/data: `report/final/data/`
- Video/demo package:
  - `video/final/demo/index.html`
  - `video/final/video_script_en.md`
  - `video/final/slide_outline.md`

## Repository Layout

- `src/`: environment, agents, training, evaluation, tuning, and visualization utilities
- `configs/`: active experiment configs (`base.yaml`, `main_xlf.yaml`, `xlf_promotion_gate.yaml`)
- `experiments/`: compact experiment summaries and selected best configs
- `report/`: final report assets (notebook, PDF, figures, tables)
- `video/`: presentation assets and interactive demo
- `tests/`: unit tests and smoke tests
- `scripts/`: setup and export helpers

## Environment Setup

```bash
bash scripts/setup_venv.sh
scripts/py.sh -m pip install -r requirements.txt
scripts/py.sh -m pytest -q
```

## Reproduce Core Workflow

### 1) Build data and run experiment suite

```bash
scripts/py.sh -m src.run_suite --config configs/base.yaml --output-root runs --include-no-cvar
```

### 2) Train/evaluate a specific algorithm

```bash
scripts/py.sh -m src.train --algo nstep_sarsa --config configs/main_xlf.yaml --seed 11
scripts/py.sh -m src.eval --run-dir <RUN_DIR>
```

### 3) Build final report visuals (if needed)

```bash
scripts/py.sh -m src.final_viz
```

## Run Interactive Demo

```bash
scripts/py.sh video/final/demo/build_demo_data.py
scripts/py.sh -m http.server 8080 --directory video/final/demo
```

Then open `http://localhost:8080`.

## Export Report PDF

```bash
bash scripts/export_report_pdf.sh report/final/report.ipynb webpdf
```

## Notes on Upload Scope

This repository tracks the English project package for report and presentation. Local-only drafts and temporary outputs are ignored via `.gitignore` (for example `tmp/`, `output/`, and bilingual private script drafts).
