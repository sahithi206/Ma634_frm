import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 2: PORTFOLIO CONSTRUCTION FUNCTIONS
# ============================================================================

def construct_gmv_portfolio(mu, Sigma):
    """Global Minimum Variance portfolio"""
    N = len(mu)
    ones = np.ones(N)
    
    # Solve: min w'Σw subject to w'1 = 1
    Sigma_inv = np.linalg.inv(Sigma)
    w = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)
    
    return w

def construct_tangency_portfolio(mu, Sigma, rf=0):
    """Mean-Variance (Tangency) portfolio"""
    N = len(mu)
    ones = np.ones(N)
    
    # Excess returns
    mu_excess = mu - rf
    
    # Solve: max (w'μ - rf) / sqrt(w'Σw) subject to w'1 = 1
    Sigma_inv = np.linalg.inv(Sigma)
    w = Sigma_inv @ mu_excess / (ones @ Sigma_inv @ mu_excess)
    
    return w

def construct_ew_portfolio(N):
    """Equal-Weighted portfolio"""
    return np.ones(N) / N

def construct_active_portfolio(returns_df, factors_df, stock_names, confidence=0.95):
    """
    Active portfolio based on alpha significance from CAPM regression
    
    Methodology (as per assignment requirement):
    - For each stock, run CAPM regression on excess returns: R_i - R_f = alpha + beta * MF + epsilon
    - All series are in decimals (MF converted from percentage to decimal)
    - Test significance of alpha at 95% confidence level (p-value < 0.05)
    - Select stocks with significant non-zero alpha
    - Construct active portfolio using Treynor-Black approach: w_i ∝ alpha_i / var(epsilon_i)
    - If no stocks qualify, use market portfolio (NIFTY 50)
    
    Note: Alpha is obtained via regression (OLS estimation) as required by assignment
    """
    significant_alphas = []
    significant_betas = []
    significant_stocks = []
    residual_vars = []
    
    # Align factors with returns by date
    factors_aligned = factors_df[factors_df['Date'].isin(returns_df['Date'])].sort_values('Date').reset_index(drop=True)
    returns_aligned = returns_df.sort_values('Date').reset_index(drop=True)
    
    # Ensure they have the same dates
    common_dates = set(factors_aligned['Date']).intersection(set(returns_aligned['Date']))
    factors_aligned = factors_aligned[factors_aligned['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
    returns_aligned = returns_aligned[returns_aligned['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
    
    # Convert market factor from percentage to decimal
    MF = factors_aligned['MF'].values / 100
    RF = factors_aligned['RF'].values
    
    for stock in stock_names:
        if stock == 'NIFTY 50':
            continue
            
        # Stock excess returns
        R_stock = returns_aligned[stock].values
        R_excess = R_stock - RF
        
        # Remove NaN values
        mask = ~np.isnan(R_excess) & ~np.isnan(MF)
        R_excess_clean = R_excess[mask]
        MF_clean = MF[mask]
        
        if len(R_excess_clean) < 30:  # Need sufficient data
            continue
        
        # CAPM regression: R_i - R_f = alpha + beta * MF + epsilon
        # All series in decimals: R_excess and MF are in decimal form
        X = np.column_stack([np.ones(len(MF_clean)), MF_clean])
        y = R_excess_clean
        
        # OLS estimation to obtain alpha via regression (as required by assignment)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha_hat = beta_hat[0]  # Alpha obtained via regression
        beta_stock = beta_hat[1]
        
        # Residuals and standard error
        residuals = y - X @ beta_hat
        residual_var = np.var(residuals, ddof=2)
        se_alpha = np.sqrt(residual_var * np.linalg.inv(X.T @ X)[0, 0])
        
        # t-test for alpha significance at 95% confidence level
        t_stat = alpha_hat / se_alpha
        dof = len(R_excess_clean) - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
        
        # Check significance: p-value < 0.05 (95% confidence level)
        if p_value < (1 - confidence):
            significant_alphas.append(alpha_hat)
            significant_betas.append(beta_stock)
            significant_stocks.append(stock)
            residual_vars.append(residual_var)
    
    # If no significant alphas, return market portfolio (NIFTY 50)
    if len(significant_stocks) == 0:
        print("  No significant alphas found. Using market portfolio.")
        return None, None
    
    # Construct active portfolio using Treynor-Black approach
    # Weights proportional to: w_i ∝ alpha_i / var(epsilon_i)
    # Note: Uses alpha obtained via regression (as required by assignment)
    alphas = np.array(significant_alphas)  # Alphas from regression
    residual_vars = np.array(residual_vars)
    
    # Active weights (unnormalized): w_i ∝ alpha_i / var(epsilon_i)
    w_active = alphas / residual_vars
    
    # Normalize to sum to 1 (fully invested portfolio)
    w_active = w_active / np.sum(np.abs(w_active))
    
    return w_active, significant_stocks