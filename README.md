# Portfolio Analysis and VaR Backtesting

This project performs **portfolio construction, rolling window backtesting, and VaR (Value-at-Risk) analysis** using historical stock and market factor data.

---

## Features

1. **Data preprocessing**: Cleans and filters stock and factor datasets.
2. **Portfolio construction**: Builds multiple portfolios:
   - GMV (Global Minimum Variance)
   - MV (Mean-Variance / Tangency)
   - EW (Equally Weighted)
   - Active (based on factor models)
   - NIFTY50 (market benchmark)
3. **Rolling window backtest**:
   - Formation period: 6 months
   - Holding period: 3 months
   - Computes cumulative returns for each windowg
4. **VaR backtesting**:
   - 99% confidence level
   - Historical simulation method
   - Records violations per portfolio
5. **Performance metrics**: Calculates standard metrics (annualized return, volatility, Sharpe ratio, etc.)
6. **Plots**: Generates cumulative returns and VaR backtest charts.

---

## Getting Started

### Requirements

- Python 3.x
- Required libraries:
  ```bash
    pip install pandas numpy matplotlib scipy
  ```

### Running the Code

- Run the main script:
  ```bash
  python3 main.py
  ```
