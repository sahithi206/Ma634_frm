import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules
import data_preprocessing as dp
import portfolio_weights_calculations as pc
import rolling_window_backtesting as bt
import performance_analysis as pm

def main():
    print("=" * 80)
    print("PORTFOLIO ANALYSIS AND VAR BACKTESTING")
    print("=" * 80)
    
    rf_rate_annual = 0.05
    # -----------------------------
    # 1. Load and preprocess data
    # -----------------------------
    print("\n1. Loading and preprocessing data...")
    stocks_df, factors_df = dp.load_and_preprocess_dates('Stocks_data.csv', 
                                                    'market_Factor_risk_Free.csv')
    stocks_df = dp.remove_nan_cols(stocks_df)
    print(f"Number of stocks after filtering: {len(stocks_df.columns) - 2}")  # Exclude Date and NIFTY 50
    
    # Save cleaned data
    cleaned_data_path = 'results/cleaned_stocks_data.csv'
    stocks_df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned stock data saved to '{cleaned_data_path}'")

    # -----------------------------
    # 2. Construct portfolios & save weights
    # -----------------------------
    # ============================================================================
# STEP: CALCULATE PORTFOLIO WEIGHTS AND SAVE COMBINED CSV
# ============================================================================

    print("\n2. Calculating portfolio weights...")

    # Extract returns for weight calculation
    stock_names = [col for col in stocks_df.columns if col != 'Date']

    # Compute covariance and mean returns
    returns_matrix = stocks_df[stock_names].pct_change().dropna()
    mu = returns_matrix.mean().values
    Sigma = returns_matrix.cov().values
    N = len(stock_names)

    # Compute weights
    weights = pd.DataFrame({'Stock': stock_names})
    weights['GMV'] = pc.construct_gmv_portfolio(mu, Sigma)
    weights['MV'] = pc.construct_tangency_portfolio(mu, Sigma, rf=rf_rate_annual)
    weights['EW'] = pc.construct_ew_portfolio(N)

    # Active portfolio
    w_active, active_stocks = pc.construct_active_portfolio(stocks_df, factors_df, stock_names)
    weights['Active'] = 0  # Initialize all as 0
    if w_active is not None:
        for stock, w in zip(active_stocks, w_active):
            weights.loc[weights['Stock'] == stock, 'Active'] = w

    # Save combined weights CSV
    combined_weights_path = 'results/portfolio_weights.csv'
    weights.to_csv(combined_weights_path, index=False)
    print(f"Portfolio weights saved to '{combined_weights_path}'")


    # -----------------------------
    # 3. Rolling window backtest
    # -----------------------------
    print("\n3. Running rolling window backtest...")
    results_df, var_results, windows = bt.rolling_window_backtest(stocks_df, factors_df)
    
    # Save rolling returns
    rolling_returns_path = 'results/rolling_returns.csv'
    results_df.to_csv(rolling_returns_path, index=False)
    print(f"Rolling returns saved to '{rolling_returns_path}'")

    # -----------------------------
    # 4. Performance metrics
    # -----------------------------
    print("\n4. Calculating performance metrics...")
    rf_rate_annual = 0.05  # 5% annual risk-free rate
    performance = pm.compute_performance_stats(results_df, rf_rate=rf_rate_annual)
    performance_path = 'results/performance_metrics.csv'
    performance.to_csv(performance_path, index=False)
    print("Performance metrics saved to '{performance_path}'")
    print(performance.to_string())

    # -----------------------------
    # 5. Plots
    # -----------------------------
    print("\n5. Generating cumulative returns plot...")
    pm.plot_cum_returns(results_df, save_path='results/cumulative_returns.png')
    
    print("\n6. Generating VaR backtest plot...")
    pm.plot_portfolio_var_backtest(var_results, save_path='results/var_backtest.png')
    
    # -----------------------------
    # 6. VaR violation summary
    # -----------------------------
    print("\n7. VaR Violation Summary (99% Confidence):")
    print("-" * 50)

    # Prepare data for CSV
    var_summary = []

    for portfolio in ['GMV', 'MV', 'EW', 'Active']:
        if portfolio in var_results:
            total = len(var_results[portfolio]['var'])
            violations = var_results[portfolio]['violations']
            rate = violations / total * 100 if total > 0 else 0
            print(f"{portfolio:10s}: {violations:2d}/{total:2d} violations ({rate:.1f}%)")
            var_summary.append({
                'Portfolio': portfolio,
                'Violations': violations,
                'Total Windows': total,
                'Violation Rate (%)': rate
            })

    # Save to CSV
    var_summary_df = pd.DataFrame(var_summary)
    var_summary_csv_path = 'results/var_violation_summary.csv'
    var_summary_df.to_csv(var_summary_csv_path, index=False)
    print(f"\nVaR violation summary saved to '{var_summary_csv_path}'")


    print("\nAnalysis complete. All files saved in 'results/' folder.")

if __name__ == "__main__":
    main()
