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

def gmv_weights(cov):
    
    "Global Minimum Variance(Risk is Minimized and risk here refers to variance)"

    inv = np.linalg.inv(cov) #inv=cov^(-1)
    ones = np.ones((inv.shape[0], 1)) #e=ones
    w = inv @ ones
    denom = float(ones.T @ inv @ ones)
    """Formula w = inv*e / e'*inv*e """
    w = (w / denom).flatten()
    return w

def tangency_weights(mu, cov, rf):

    mu = mu.reshape(-1, 1)
    inv = np.linalg.inv(cov)

    # Excess Return Vector: (mu - rf*e)
    ones = np.ones((inv.shape[0], 1))
    
    """Tangency Portfolio: Maximizes Sharpe Ratio.
       Formula w = inv*(m-rf) / ones'cov^(-1)(m-rf) 
    """
    N = len(mu)
    ones = np.ones(N)
    
    # Excess returns
    excess = mu - rf
    inv = np.linalg.inv(cov) # inv=cov^(-1)   

    # w = inv * (mu - rf*e) / (ones' * inv * (mu - rf*e))

    w = inv @ excess / (ones @ inv @ excess)
    
    return w

def equal_weights(N):
    """Equal-Weighted portfolio
       Formula w = 1/N for all assets"""
       
    return np.ones(N) / N

def construct_active_portfolio(returns_df, factors_df, stock_names, confidence=0.95):
    """
    Build an active portfolio based on CAPM alpha significance.

    Steps (as required):
    1. Estimate CAPM for each stock: (Ri - Rf) = alpha + beta * MF + error
    2. Test alpha significance using 95% confidence (p-value < 0.05)
    3. Keep only stocks with significant alphas
    4. Assign Treynor-Black weights: wi ∝ alpha_i / residual_variance_i
    5. If none pass the test → return market-only portfolio
    """

    import numpy as np
    from scipy import stats

    # ----------------------------
    # Align dates between returns and factors
    # ----------------------------
    merged_dates = set(returns_df['Date']).intersection(set(factors_df['Date']))
    returns_sorted = returns_df[returns_df['Date'].isin(merged_dates)].sort_values("Date").reset_index(drop=True)
    factors_sorted = factors_df[factors_df['Date'].isin(merged_dates)].sort_values("Date").reset_index(drop=True)

    # Market factor and risk-free rate (convert market % to decimal)
    market_factor = factors_sorted['MF'].values / 100
    risk_free = factors_sorted['RF'].values

    chosen_alphas = []
    chosen_betas = []
    chosen_stocks = []
    chosen_resvars = []

    # ----------------------------
    # Run CAPM for each stock
    # ----------------------------
    for asset in stock_names:
        if asset == "NIFTY Index":
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
        alpha_est, beta_est = coef

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
            chosen_betas.append(beta_est)
            chosen_resvars.append(res_var)

    # No significant alphas then market-only
    if len(chosen_stocks) == 0:
        print("  No significant alphas found. Using market portfolio.")
        return None, None

    # Treynor–Black active portfolio : w_i ∝ alpha_i / residual_var_i
    alphas = np.array(chosen_alphas)
    resvars = np.array(chosen_resvars)

    raw_w = alphas / resvars

    # Normalize using absolute sum for stability
    weights = raw_w / np.sum(np.abs(raw_w))

    return weights, chosen_stocks
