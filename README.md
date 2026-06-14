# ADF Test with Macro‑Determined Breaks

Performs Augmented Dickey‑Fuller (ADF) unit root tests on ETF returns with structural breakpoints determined by macro variables (e.g., VIX > 70th percentile). The per‑ETF score is the negative ADF statistic – higher = stronger mean reversion, lower = trending.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Primary macro (e.g., VIX) determines break locations
- ADF regression with break dummies
- Score = -ADF t‑statistic (higher = mean‑reverting)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-adf-macro-breaks-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF is mean‑reverting (predictable, range‑bound).
- Low score → ETF is trending (random walk, momentum‑like).

## Requirements

See `requirements.txt`.
