# ML-VaR: Forecasting Portfolio Tail Risk with Machine Learning

Predicting a portfolio's one-day 95% Value-at-Risk (VaR) by directly estimating
the 5th percentile of tomorrow's return with quantile regression — instead of
assuming returns are normally distributed.

> On a bad day, how much could a $1,000,000 stock portfolio realistically lose?
> Classic VaR answers this by assuming a bell-curve shape for returns. Real
> markets have fatter tails than that (this portfolio's daily returns show
> **9.7x excess kurtosis** vs. 0 for a normal distribution), so this project
> trains ML models to estimate the tail directly from market conditions
> instead of assuming a distribution shape.

## Table of contents

- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Setup](#setup)
- [Running the notebook](#running-the-notebook)
- [Running the API](#running-the-api)
- [API reference](#api-reference)
- [Docker](#docker)
- [Limitations & next steps](#limitations--next-steps)


## Data

- **Portfolio:** AAPL (25%), MSFT (25%), NVDA (20%), JPM (15%), XOM (15%)
- **Systematic factors:** S&P 500 (^GSPC), VIX (^VIX), 10-Year Treasury yield
  (^TNX), US Dollar Index (DX-Y.NYB), WTI crude oil (CL=F)
- **Source:** Yahoo Finance via `yfinance`, daily data from 2015-01-01 to
  present
- **After feature engineering:** 2,885 daily observations x 13 features
  (lagged returns, rolling volatility at 5/20/60 days, 60-day drawdown,
  factor moves, portfolio dollar-volume change)
- **Split:** chronological, no shuffling — 70% train / 15% validation / 15%
  test, so no future information leaks into training

## Methodology

Three models predict the conditional 5th percentile of next-day portfolio
return, `q_0.05(R_{t+1} | X_t)`, which converts to a 95% VaR as
`VaR = -q_0.05`:

1. **Historical Quantile (baseline)** — fixed 5% quantile of training-period
   returns, same number every day
2. **Linear Quantile Regression** — interpretable, minimizes pinball loss
   directly
3. **XGBoost Quantile Regression** — gradient-boosted trees tuned over 36
   hyperparameter combinations (grid search on the validation set)

Evaluation uses pinball loss (the actual training objective), MAE, and VaR
breach rate (how often the actual return fell below the predicted quantile —
the practical calibration check, since it should land near 5% for a
well-calibrated 95% VaR).

## Results

Test set: 433 trading days, Dec 2024 – Sep 2026.

| Model                        | Pinball Loss | MAE    | VaR Breach Rate |
|-------------------------------|:------------:|:------:|:---------------:|
| Historical Quantile (baseline)| 0.001581     | 0.0261 | 2.31%            |
| Linear Quantile Regression    | 0.001420     | 0.0221 | 3.70%            |
| **XGBoost Quantile Regression**| **0.001401**| **0.0207**| **4.85%**    |

XGBoost has the lowest error on both metrics, and its 4.85% breach rate is
closest to the 5% target — meaning it isn't just more accurate on average, its
risk estimate is better calibrated. The most important feature is the
60-day market drawdown: how far the market currently sits below its recent
peak predicts tomorrow's tail risk better than any single day's move.


## Running the notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Run all cells top to bottom. This downloads market data, engineers features,
trains and compares all three models, and saves the trained XGBoost model to
`models/ml_var_xgboost.pkl` — required before the API will start.

## Running the API

```bash
uvicorn app:app --reload
```

- Web UI: http://13.53.212.35:8000/app
- Interactive API docs: http://13.53.212.35:8000/docs
- Health check: http://13.53.212.35:8000/health

## API reference

| Endpoint   | Method | Description                                  |
|------------|--------|-----------------------------------------------|
| `/`        | GET    | API metadata                                  |
| `/health`  | GET    | Health check                                  |
| `/info`    | GET    | Model metadata (features, quantile, version)  |
| `/predict` | POST   | Returns predicted VaR for a market snapshot   |
| `/app`     | GET    | Serves the web UI                             |

Example request to `/predict`:

```json
{
  "market_return": 0.001,
  "vix_change": 0.02,
  "rate_change": 0.01,
  "usd_return": -0.003,
  "oil_return": 0.005,
  "ret_lag1": 0.002,
  "ret_lag2": -0.001,
  "ret_lag5": 0.004,
  "vol_5d": 0.01,
  "vol_20d": 0.012,
  "vol_60d": 0.015,
  "drawdown_60d": -0.03,
  "portfolio_volume_change": 0.01,
  "portfolio_value": 1000000
}
```

## Docker

```bash
docker build -t var-ml-api .
docker run -p 8000:8000 var-ml-api
```

(Build the model first — the image copies `models/ml_var_xgboost.pkl` in at
build time, so it must exist before running `docker build`.)

## Limitations & next steps

- **No formal statistical backtest yet.** The breach rate is compared to the
  5% target by eye; the next step is a proper Kupiec proportion-of-failures
  test to check that statistically.
- **Single chronological split, not walk-forward.** A rolling/expanding
  retraining scheme would better reflect how the model would perform across
  changing market regimes (e.g. 2020, 2022).
- **No classical parametric baseline** (e.g. GARCH volatility model) to
  benchmark against, only the naive historical quantile.


