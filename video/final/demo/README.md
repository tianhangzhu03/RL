# Gate Demo (Video Pre)

This folder contains an interactive visualization demo for the 3-algorithm comparison.

## Files
- `index.html`: interactive frontend demo with five panels:
  - Why XLF + project purpose
  - TD mechanics lab (parameter sliders + live update equations)
  - Gate decision replay (animated algorithm comparison)
  - ETF application case (risk-profile-based policy choice + expected capital)
  - Parameter/setback explorer (SG tuning trajectory)
- `build_demo_data.py`: builds `data.js` from existing CSV artifacts
- `data.js`: generated runtime data for the demo

## Data provenance
All displayed numbers come from existing repository CSV files:
- `video/assets/data/xlf_sync_gate.csv`
- `video/assets/data/sg_controlled_budget.csv`
- `video/assets/data/xlf_with_vs_no.csv`
- `video/assets/data/xlf_main_conclusion.csv`

No new training run is required.

Application panel chart overlay:
- `xlf_trend` is fetched from `yfinance` (`XLF`, monthly normalized close, 2019-01 to 2026-01) when building `data.js`.
- If network/data fetch fails, the demo still runs and only hides the historical trend line.

## Run
1. Build data bundle:
```bash
scripts/py.sh video/final/demo/build_demo_data.py
```

2. Launch local static server:
```bash
cd video/final/demo
python -m http.server 8080
```

3. Open:
- `http://localhost:8080`

## Suggested recording usage
- Start from panel 1 to explain *why XLF* and practical deployment purpose.
- Use panel 2 as the principle demo (live slider interaction for TD update rules).
- Use panel 3 as the core evidence demo (gate replay).
- Use panel 4 as the practical ETF application segment.
- Use panel 5 for trials/setbacks and parameter sensitivity summary.
