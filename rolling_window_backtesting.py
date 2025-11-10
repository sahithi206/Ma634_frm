#rolling window backtesting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import data_preprocessing as dp
import portfolio_weights_calculations as pc

def extract_window_data(returns_df, factors_df, start_date, end_date):
    """Select returns and factor data within a date range"""
    mask = (returns_df['Date'] >= start_date) & (returns_df['Date'] <= end_date)
    window_returns = returns_df[mask].reset_index(drop=True)
    
    if factors_df is not None:
        mask_factors = (factors_df['Date'] >= start_date) & (factors_df['Date'] <= end_date)
        window_factors = factors_df[mask_factors].reset_index(drop=True)
    else:
        window_factors = None
    
    return window_returns, window_factors

def compute_holding_returns(returns_df, weights, stock_names, start_date, end_date):
    """
    Compute portfolio return over a holding period.
    - Uses weights from formation period.
    - Daily portfolio return: sum(weight * stock return)
    - Compounded over the period: cumulative = prod(1+daily returns) - 1
    """
    window_returns, _ = extract_window_data(returns_df, None, start_date, end_date)
    
    if weights is None: 
        portfolio_returns = window_returns['NIFTY Index'].values
    else:
        stock_returns = window_returns[stock_names].values      # Portfolio returns each day: R_p,t = sum(weight * Return)
        portfolio_returns = stock_returns @ weights             # Computing simple returns over holding period: (1+R_1)*(1+R_2)*...*(1+R_T) - 1
    
    cumulative_return = np.prod(1 + portfolio_returns) - 1
    return cumulative_return

def perform_rolling_backtest(stocks_df, factors_df):
    """Run rolling window backtest over multiple periods"""
    returns_df = dp.compute_returns(stocks_df)
    
    # Define formation and holding windows
    windows = []
    start_date = pd.Timestamp('2009-01-01')
    end_date = pd.Timestamp('2022-12-31')
    current_formation_start = start_date
    
    while True:
        # formation period = 6 months
        formation_end = current_formation_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        # holding period = 3 months after formation
        holding_start = formation_end + pd.Timedelta(days=1)
        holding_end = holding_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        if holding_end > end_date:
            break
        windows.append({
            'formation_start': current_formation_start,
            'formation_end': formation_end,
            'holding_start': holding_start,
            'holding_end': holding_end
        })
        # move by 3 months
        current_formation_start += pd.DateOffset(months=3)
    
    print(f"\nTotal rolling windows: {len(windows)}")
    
    # Store returns and VaR results
    results = {p: [] for p in ['GMV', 'MV', 'EW', 'Active', 'NIFTY50']}
    var_results = {p: {'var': [], 'realized': [], 'violations': 0} for p in results.keys()}
    
    # Loop through each window
    for i, window in enumerate(windows):
        print(f"\nWindow {i+1}/{len(windows)}: Formation {window['formation_start'].date()} to {window['formation_end'].date()}")
        
        formation_returns, formation_factors = extract_window_data(
            returns_df, factors_df,
            window['formation_start'], window['formation_end']
        )
        
        stock_names = [col for col in formation_returns.columns if col not in ['Date', 'NIFTY Index']]
        stock_names = [s for s in stock_names if formation_returns[s].notna().all()]
        N = len(stock_names)
        
        if N < 2:
            print(f"  Skipping window: insufficient stocks ({N})")
            continue
        
        # Compute mean and covariance
        stock_returns = formation_returns[stock_names].values
        mu = np.mean(stock_returns, axis=0)
        Sigma = np.cov(stock_returns, rowvar=False) + np.eye(N) * 1e-8  # regularization
        
        rf_avg = formation_factors['RF'].mean()
        
        # Construct portfolios
        w_gmv = pc.gmv_weights(Sigma)
        w_mv = pc.tangency_weights(mu, Sigma, rf_avg)
        w_ew = pc.equal_weights(N)
        w_active, active_stocks = pc.construct_active_portfolio(
            formation_returns, formation_factors, stock_names
        )
        
        # Holding period returns
        holding_returns = {
            'GMV': compute_holding_returns(returns_df, w_gmv, stock_names, window['holding_start'], window['holding_end']),
            'MV': compute_holding_returns(returns_df, w_mv, stock_names, window['holding_start'], window['holding_end']),
            'EW': compute_holding_returns(returns_df, w_ew, stock_names, window['holding_start'], window['holding_end']),
            'Active': compute_holding_returns(returns_df, w_active, active_stocks, window['holding_start'], window['holding_end']) 
                     if w_active is not None else compute_holding_returns(returns_df, None, None, window['holding_start'], window['holding_end']),
            'NIFTY50': compute_holding_returns(returns_df, None, None, window['holding_start'], window['holding_end'])
        }
        
        for p in holding_returns.keys():
            results[p].append(holding_returns[p])
        
        # 99% VaR via historical simulation
        holding_data, _ = extract_window_data(returns_df, factors_df, window['holding_start'], window['holding_end'])
        L = len(holding_data)
        
        for portfolio in ['GMV', 'MV', 'EW', 'Active','NIFTY50']:
            if portfolio == 'Active' and w_active is None:
                weights = stocks = None
            elif portfolio == 'NIFTY50':
                weights = stocks = None
            else:
                weights = {'GMV': w_gmv, 'MV': w_mv, 'EW': w_ew, 'Active': w_active}[portfolio]
                stocks = {'GMV': stock_names, 'MV': stock_names, 'EW': stock_names, 'Active': active_stocks}[portfolio]
            
            daily_returns = formation_returns['NIFTY Index'].values if weights is None else formation_returns[stocks].values @ weights
            
            # Simulate L-day returns
            simulated_returns = [np.prod(1 + daily_returns[j:j+L]) - 1 for j in range(len(daily_returns) - L + 1)]
            if not simulated_returns:
                continue
            simulated_returns = np.array(simulated_returns)
            
            var_99 = -np.percentile(simulated_returns, 1)
            realized_return = holding_returns[portfolio]
            violation = realized_return < -var_99
            
            var_results[portfolio]['var'].append(var_99)
            var_results[portfolio]['realized'].append(realized_return)
            if violation:
                var_results[portfolio]['violations'] += 1
    
    results_df = pd.DataFrame(results)
    return results_df, var_results, windows