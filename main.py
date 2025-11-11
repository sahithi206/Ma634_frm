import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import data_preprocessing as dp
import rolling_window_backtesting as bt
import performance_analysis as pm

STOCKS_DIR='Dataset/Stocks_data.csv'
FACTORS_DIR='Dataset/market_Factor_risk_Free.csv'
RESULTS_DIR='results/'
RF_RATE = 0.05 

def main():
    # Load and preprocess data
    if(os.path.exists(RESULTS_DIR)==False):
        os.makedirs(RESULTS_DIR)
        
    print("______Loading and preprocessing data______")
    stocks_df, factors_df = dp.load_and_preprocess_dates(STOCKS_DIR,FACTORS_DIR)
    
    # Use the target index name
    market_index_name = 'NIFTY Index'
    stocks_df = dp.remove_nan_cols(stocks_df, keep_cols=[market_index_name])
    stock_names = [col for col in stocks_df.columns if col not in ['Date', market_index_name]]
    
    print(f"Number of stocks after filtering: {len(stock_names)}")
    
    # Save cleaned data
    cleaned_data_path = f'{RESULTS_DIR}/cleaned_stocks_data.csv'
    stocks_df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned stock data saved to '{cleaned_data_path}'")

    # Rolling window backtest 
    print("_____Rolling window backtest_____")
    results, var, windows = bt.perform_rolling_backtest(stocks_df, factors_df)
    
    # Save rolling returns
    rr_path = f'{RESULTS_DIR}/rolling_returns.csv'
    results.to_csv(rr_path, index=False)
    print(f"Rolling returns saved to '{rr_path}'")

    # Performance metrics
    print("_____Performance metrics_____")
    metrics = pm.compute_performance_stats(results, rf_rate=RF_RATE)
    metrics_path = f'{RESULTS_DIR}/performance_metrics.csv'
    metrics.to_csv(metrics_path, index=True) 
    print(f"Performance metrics saved to file: '{metrics}'")
    print(metrics.to_string(float_format='%.4f'))

    # Plots
    print("_____Cumulative returns plot_____")
    pm.plot_cum_returns(results, save_path=f'{RESULTS_DIR}/cumulative_returns.png')
    
    print("_____VaR backtest plot_____")
    pm.plot_portfolio_var_backtest(var, save_path=f'{RESULTS_DIR}/var_backtest.png')

     # VaR violation summary
    print("VaR Violation Summary:")

    # Prepare data for CSV
    var_summary = []
    portfolios = ['GMV', 'MV', 'EW', 'Active', market_index_name]

    for p in portfolios:
        if p in var:
            size = len(var[p]['var'])
            violations = var[p]['violations']
            rate = violations/size*100 if size > 0 else 0
            print(f"{p:12s}: {violations:2d}/{size:2d} violations ({rate:.1f}%)")
            var_summary.append({
                'Portfolio': p,
                'Violations': violations,
                'Total Windows': size,
                'Violation Rate (%)': rate
            })

    # Save to CSV
    var_summary_df = pd.DataFrame(var_summary)
    var_summary_csv_path = f'{RESULTS_DIR}/var_violation_summary.csv'
    var_summary_df.to_csv(var_summary_csv_path, index=False)
    print(f"VaR violation summary saved to '{var_summary_csv_path}'")


    print(f"\nAnalysis complete. All files saved in {RESULTS_DIR} folder.")

if __name__ == "__main__":
    main()