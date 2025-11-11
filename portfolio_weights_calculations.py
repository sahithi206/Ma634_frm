import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# PORTFOLIO CONSTRUCTION FUNCTIONS

def gmv_weights(cov):
    
    "Global Minimum Variance (Risk is Minimized and risk here refers to variance)"

    inv = np.linalg.inv(cov) #inv=cov^(-1)
    ones = np.ones((inv.shape[0], 1)) #e=ones
    w = inv @ ones
    denom = float(ones.T @ inv @ ones)
    """Formula w = inv*e / e'*inv*e """
    w = (w / denom).flatten()
    return w

def tangency_weights(mu, cov, rf):
    """Tangency Portfolio: Maximizes Sharpe Ratio."""

    mu = mu.reshape(-1, 1)
    
    N = len(mu)
    ones = np.ones((N, 1))
    
    # Excess returns: (mu - rf*e)
    excess = mu - (rf * ones) 
    inv = np.linalg.inv(cov) # inv=cov^(-1)   

    # w = inv * (mu - rf*e) / (ones' * inv * (mu - rf*e))
    w = inv @ excess / (ones.T @ inv @ excess)
    
    return w.flatten()

def equal_weights(N):
    """Equal-Weighted portfolio
       Formula w = 1/N for all assets"""
       
    return np.ones(N) / N

def construct_active_portfolio(returns_df, factors_df, stock_names, confidence=0.95):
    """
    Build an active portfolio based on CAPM alpha significance (Treynor-Black).
    """

    # Align dates between returns and factors
    merged_dates = set(returns_df['Date']).intersection(set(factors_df['Date']))
    returns_sorted = returns_df[returns_df['Date'].isin(merged_dates)].sort_values("Date").reset_index(drop=True)
    factors_sorted = factors_df[factors_df['Date'].isin(merged_dates)].sort_values("Date").reset_index(drop=True)

    # Market factor and risk-free rate (convert market % to decimal)
    market_factor = factors_sorted['MF'].values / 100
    risk_free = factors_sorted['RF'].values

    chosen_alphas = []
    chosen_stocks = []
    chosen_resvars = []
    
    # Use the unified index name
    market_index_name = 'NIFTY Index'

    # Run CAPM for each stock
    for asset in stock_names:
        if asset == market_index_name:
            continue

        asset_ret = returns_sorted[asset].values
        excess_ret = asset_ret - risk_free

        # Handle missing values
        valid = ~np.isnan(excess_ret) & ~np.isnan(market_factor)
        y = excess_ret[valid]
        x = market_factor[valid]

        if len(y) < 30:
            continue

        # CAPM regression: y = alpha + beta*x + err
        X = np.column_stack([np.ones(len(x)), x])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha_est = coef[0]

        # Residuals and variance
        residual = y - X @ coef
        res_var = np.var(residual, ddof=2)

        # Standard error of alpha
        inv_xtx = np.linalg.inv(X.T @ X)
        se_alpha = np.sqrt(res_var * inv_xtx[0, 0])

        # t-statistic and p-value
        t_alpha = alpha_est / se_alpha
        dof = len(y) - 2
        p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), dof))

        # Significance test
        if p_alpha < (1 - confidence):
            chosen_stocks.append(asset)
            chosen_alphas.append(alpha_est)
            chosen_resvars.append(res_var)

    if len(chosen_stocks) == 0:
        return None, None 

    # Treynor–Black active portfolio : w_i ∝ alpha_i / residual_var_i
    alphas = np.array(chosen_alphas)
    resvars = np.array(chosen_resvars)

    raw_w = alphas / resvars

    # Normalize using absolute sum
    weights = raw_w / np.sum(np.abs(raw_w))

    return weights, chosen_stocks