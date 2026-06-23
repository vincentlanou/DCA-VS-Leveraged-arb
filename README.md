# Leveraged Lump Sum vs. Dollar-Cost Averaging: A Quantitative Analysis

## Overview
This repository contains a Python-based financial model evaluating the risk-adjusted performance of a leveraged lump-sum investment strategy against a Dollar-Cost Averaging (DCA) baseline. 

The model simulates the cash flow mechanics of a professional student line of credit (36-month interest-only grace period followed by a 48-month capital amortization) and evaluates the probability of the leveraged strategy generating a positive real yield (Alpha) over a 7-year horizon.

## Methodology

The pipeline is divided into three components:

1. **Data Acquisition & Portfolio Optimization**
   * Monthly historical pricing for the S&P 500, TSX Composite, and MSCI EAFE extracted via the `yfinance` API.
   * Historical Bank of Canada (BoC) policy rates fetched via the BoC Valet API to establish the true cost of debt.
   * Asset weighting optimized by maximizing a modified Sharpe Ratio, using the borrowing cost as the hurdle rate.

2. **Deterministic Backtesting**
   * Replays historical sequences, since 2009, to calculate the exact cash flow requirements and nominal portfolio values of both strategies.

3. **Stochastic Modeling (Monte Carlo)**
   * **Bootstrap Resampling:** Reconstructs 7-year trajectories using random historical monthly blocks to preserve dynamic cross-asset and interest rate correlations.
   * **Parametric Simulations:** Uses Multivariate Normal and Student's t-distributions (via Cholesky decomposition) to stress-test the portfolio against fat-tail risks and extreme market conditions.

## Key Findings
https://vincentlanou.github.io/DCA-VS-Leveraged-arb/Notebook.html
*Note: All final outputs are discounted by a 2% target inflation rate to reflect real purchasing power rather than nominal returns.*

Based on a 7-year horizon with a 5% average borrowing cost and a loan of $1000 and data starting in 2003:
* **Probability of Outperformance (Leverage > DCA):** ~90%
* **Mean Real Spread:** ~$430
* **Tail Risk (5th Percentile):** ~$100

## Tech Stack
* `Python 3.x`
* `NumPy`, `Pandas` (Data manipulation)
* `SciPy` (Covariance matrices, optimization)
* `Matplotlib` (Dispersion and frequency histograms)
* `yfinance`, `requests` (API data extraction)

## Usage
Run `Notebook.ipynb` to execute the data pipeline, perform the Sharpe optimization, and generate the Monte Carlo dispersion plots.

## Disclaimer
This model is for academic research and portfolio demonstration. It does not constitute financial advice.
