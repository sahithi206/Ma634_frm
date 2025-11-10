import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

import data_preprocessing as dp
import portfolio_weights_calculations as pc
# ============================================================================
# STEP 3: ROLLING WINDOW BACKTEST
# ============================================================================

def get_window_data(returns_df, factors_df, start_date, end_date):
    """Extract data for a specific window"""
    mask = (returns_df['Date'] >= start_date) & (returns_df['Date'] <= end_date)
    window_returns = returns_df[mask].reset_index(drop=True)
    
    if factors_df is not None:
        mask_factors = (factors_df['Date'] >= start_date) & (factors_df['Date'] <= end_date)
        window_factors = factors_df[mask_factors].reset_index(drop=True)
    else:
        window_factors = None
    
    return window_returns, window_factors

def calculate_holding_return(returns_df, weights, stock_names, start_date, end_date):
    """
    Calculate realized portfolio return over holding period
    
    Methodology:
    - Uses portfolio weights from formation period (no rebalancing during holding)
    - Computes daily portfolio returns: R_p,t = sum(w_i * R_i,t)
    - Compounds simple returns over holding period: (1+R_1)*(1+R_2)*...*(1+R_T) - 1
    
    Return Definition:
    - All returns are simple returns (not log returns)
    - Compounding follows: R_{0,T} = prod(1 + R_t) - 1
    """
    window_returns, _ = get_window_data(returns_df, None, start_date, end_date)
    
    if weights is None:  # Market portfolio (NIFTY Index)
        portfolio_returns = window_returns['NIFTY Index'].values
    else:
        # Portfolio returns each day: R_p,t = sum(w_i * R_i,t)
        stock_returns = window_returns[stock_names].values
        portfolio_returns = stock_returns @ weights
    
    # Compound simple returns over holding period: (1+R_1)*(1+R_2)*...*(1+R_T) - 1
    cumulative_return = np.prod(1 + portfolio_returns) - 1
    
    return cumulative_return

def rolling_window_backtest(stocks_df, factors_df):
    """Perform rolling window backtest"""
    returns_df = dp.compute_returns(stocks_df)
    
    # Define windows
    windows = []
    start_date = pd.Timestamp('2009-01-01')
    end_date = pd.Timestamp('2022-12-31')
    
    current_formation_start = start_date
    
    while True:
        # Formation period: 6 months
        formation_end = current_formation_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        
        # Holding period: 3 months after formation
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
        
        # Shift by 3 months
        current_formation_start = current_formation_start + pd.DateOffset(months=3)
    
    print(f"\nTotal rolling windows: {len(windows)}")
    
    # Results storage
    results = {
        'GMV': [],
        'MV': [],
        'EW': [],
        'Active': [],
        'NIFTY50': []
    }
    
    var_results = {
        'GMV': {'var': [], 'realized': [], 'violations': 0},
        'MV': {'var': [], 'realized': [], 'violations': 0},
        'EW': {'var': [], 'realized': [], 'violations': 0},
        'Active': {'var': [], 'realized': [], 'violations': 0}
    }
    
    # Process each window
    for i, window in enumerate(windows):
        print(f"\nWindow {i+1}/{len(windows)}: Formation {window['formation_start'].date()} to {window['formation_end'].date()}")
        
        # Get formation period data
        formation_returns, formation_factors = get_window_data(
            returns_df, factors_df,
            window['formation_start'], window['formation_end']
        )
        
        # Get stock names (exclude Date and NIFTY Index)
        stock_names = [col for col in formation_returns.columns if col not in ['Date', 'NIFTY Index']]
        
        # Remove stocks with missing data in formation period
        valid_stocks = []
        for stock in stock_names:
            if formation_returns[stock].isna().sum() == 0:
                valid_stocks.append(stock)
        
        stock_names = valid_stocks
        N = len(stock_names)
        
        if N < 2:
            print(f"  Skipping window: insufficient stocks ({N})")
            continue
        
        # Calculate mean and covariance from formation period
        stock_returns = formation_returns[stock_names].values
        mu = np.mean(stock_returns, axis=0)
        Sigma = np.cov(stock_returns, rowvar=False)
        
        # Add small regularization to ensure positive definite
        Sigma = Sigma + np.eye(N) * 1e-8
        
        # Risk-free rate (average over formation period)
        rf_avg = formation_factors['RF'].mean()
        
        # Construct portfolios
        w_gmv = pc.construct_gmv_portfolio(mu, Sigma)
        w_mv = pc.construct_tangency_portfolio(mu, Sigma, rf_avg)
        w_ew = pc.construct_ew_portfolio(N)
        w_active, active_stocks = pc.construct_active_portfolio(
            formation_returns, formation_factors, stock_names
        )
        
        # Calculate holding period returns
        holding_returns = {
            'GMV': calculate_holding_return(returns_df, w_gmv, stock_names,
                                           window['holding_start'], window['holding_end']),
            'MV': calculate_holding_return(returns_df, w_mv, stock_names,
                                          window['holding_start'], window['holding_end']),
            'EW': calculate_holding_return(returns_df, w_ew, stock_names,
                                          window['holding_start'], window['holding_end']),
            'Active': calculate_holding_return(returns_df, w_active, active_stocks,
                                               window['holding_start'], window['holding_end']) if w_active is not None else
                     calculate_holding_return(returns_df, None, None,
                                            window['holding_start'], window['holding_end']),
            'NIFTY50': calculate_holding_return(returns_df, None, None,
                                               window['holding_start'], window['holding_end'])
        }
        
        for portfolio in ['GMV', 'MV', 'EW', 'Active', 'NIFTY50']:
            results[portfolio].append(holding_returns[portfolio])
        
        # VaR calculation (99%, historical simulation)
        # Methodology: Historical Simulation Method
        # - Use formation period data to estimate L-period VaR
        # - L = number of trading days in holding period (typically ~63 days for 3 months)
        # - Simulate L-period returns by rolling window over formation period
        # - Compound daily returns: (1+r1)*(1+r2)*...*(1+rL) - 1
        # - 99% VaR = negative of 1st percentile of simulated L-period returns
        # - Violation occurs if: realized_return < -VaR (i.e., loss exceeds VaR)
        
        # Get holding period data for VaR
        holding_data, _ = get_window_data(returns_df, factors_df,
                                         window['holding_start'], window['holding_end'])
        L = len(holding_data)  # Number of trading days in holding period
        # Note: L ≈ 63 trading days for 3-month holding period (assuming ~21 trading days/month)
        
        # For each portfolio, calculate VaR
        for portfolio in ['GMV', 'MV', 'EW', 'Active']:
            if portfolio == 'Active' and w_active is None:
                weights = None
                stocks = None
            else:
                weights = {'GMV': w_gmv, 'MV': w_mv, 'EW': w_ew, 'Active': w_active}[portfolio]
                stocks = {'GMV': stock_names, 'MV': stock_names, 'EW': stock_names, 'Active': active_stocks}[portfolio]
            
            # Historical simulation: use formation period to estimate VaR
            # Portfolio returns are computed using formation period weights
            if weights is None:  # Market portfolio (NIFTY Index)
                daily_returns = formation_returns['NIFTY Index'].values
            else:
                stock_rets = formation_returns[stocks].values
                daily_returns = stock_rets @ weights  # Portfolio daily returns
            
            # Simulate L-period returns by rolling window over formation period
            # Each simulation takes L consecutive daily returns and compounds them
            n_simulations = len(daily_returns) - L + 1
            simulated_returns = []
            
            for j in range(n_simulations):
                period_rets = daily_returns[j:j+L]
                # Compound simple returns: (1+r1)*(1+r2)*...*(1+rL) - 1
                cumulative = np.prod(1 + period_rets) - 1
                simulated_returns.append(cumulative)
            
            if len(simulated_returns) == 0:
                continue
            
            simulated_returns = np.array(simulated_returns)
            
            # 99% VaR: 1st percentile of simulated L-period returns
            # VaR is the loss level, so we take negative of 1st percentile
            var_99 = -np.percentile(simulated_returns, 1)
            
            # Check violation: VaR violation occurs if realized loss exceeds VaR
            # Violation: R_p,real < -VaR (i.e., realized_return < -var_99)
            realized_return = holding_returns[portfolio]
            violation = realized_return < -var_99
            
            var_results[portfolio]['var'].append(var_99)
            var_results[portfolio]['realized'].append(realized_return)
            if violation:
                var_results[portfolio]['violations'] += 1
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, var_results, windows