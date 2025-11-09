import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 4: PERFORMANCE ANALYSIS
# ============================================================================

def calculate_performance_metrics(returns_df, rf_rate=0.0):
    """
    Calculate performance metrics for each portfolio
    
    Annualization Convention:
    - 3-month holding periods are treated as quarterly returns
    - 4 periods per year (quarterly compounding)
    - Alternative: 252 trading days/year, 3 months ≈ 63 trading days
    - We use quarterly periods (4 per year) for simplicity
    
    Return Definitions:
    - Input returns_df contains 3-month holding period returns (simple returns)
    - We convert to log returns for annualization, then back to simple returns
    - Compounding: (1+r1)*(1+r2)*...*(1+rn) - 1 for multi-period returns
    
    Risk-free Rate:
    - rf_rate should be annualized (e.g., 0.05 for 5% annual)
    """
    metrics = {}
    
    # Annualization: 4 periods per year (quarterly returns)
    # Note: 3-month holding periods = 1 quarter
    # Alternative convention: 252 trading days/year, 3 months ≈ 63 trading days
    periods_per_year = 4
    
    for portfolio in returns_df.columns:
        rets = returns_df[portfolio].values
        
        # Convert to log returns for annualization
        log_rets = np.log(1 + rets)
        
        # Annualized return
        mean_log_return = np.mean(log_rets)
        ann_return = np.exp(mean_log_return * periods_per_year) - 1
        
        # Annualized volatility
        std_log_return = np.std(log_rets, ddof=1)
        ann_volatility = std_log_return * np.sqrt(periods_per_year)
        
        # Sharpe ratio (assuming rf_rate is annual)
        sharpe = (ann_return - rf_rate) / ann_volatility if ann_volatility > 0 else 0
        
        metrics[portfolio] = {
            'Mean Return (Ann.)': ann_return,
            'Std Dev (Ann.)': ann_volatility,
            'Sharpe Ratio': sharpe
        }
    
    # Information ratio relative to NIFTY50
    nifty_rets = returns_df['NIFTY50'].values
    for portfolio in returns_df.columns:
        if portfolio == 'NIFTY50':
            metrics[portfolio]['Information Ratio'] = np.nan
        else:
            excess_rets = returns_df[portfolio].values - nifty_rets
            tracking_error = np.std(excess_rets, ddof=1) * np.sqrt(periods_per_year)
            mean_excess = np.mean(excess_rets) * periods_per_year
            ir = mean_excess / tracking_error if tracking_error > 0 else 0
            metrics[portfolio]['Information Ratio'] = ir
    
    return pd.DataFrame(metrics).T

def plot_cumulative_returns(results_df, save_path=None):
    """Plot cumulative returns of portfolios"""
    cum_returns = (1 + results_df).cumprod()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cum_returns.plot(ax=ax)
    
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)

def plot_var_backtest(var_results, save_path=None):
    """Plot VaR vs realized returns"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    portfolios = ['GMV', 'MV', 'EW', 'Active']
    
    for idx, portfolio in enumerate(portfolios):
        ax = axes[idx]
        
        var_values = -np.array(var_results[portfolio]['var'])  # Plot as negative
        realized = np.array(var_results[portfolio]['realized'])
        violations = var_results[portfolio]['violations']
        
        x = range(len(var_values))
        
        ax.plot(x, var_values, 'r-', label='99% VaR (Negative)', linewidth=2)
        ax.plot(x, realized, 'b-', label='Realized Return', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Highlight violations
        violation_mask = realized < var_values
        if np.any(violation_mask):
            ax.scatter(np.array(x)[violation_mask], realized[violation_mask],
                       color='red', s=50, zorder=5, label='VaR Violation')
        
        ax.set_xlabel('Window Number', fontsize=10)
        ax.set_ylabel('Return', fontsize=10)
        ax.set_title(f'{portfolio} Portfolio (Violations: {violations}/{len(var_values)})',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
