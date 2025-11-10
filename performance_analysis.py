import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

"""
Compute annualized performance metrics for each portfolio.

- Treat each 3-month holding period as one quarter (4 periods per year).
- For simplicity, use quarterly compounding instead of daily (252-day) scaling.
- Convert 3-month simple returns to log form for annualization, then back to simple returns.
- The risk-free rate (rf_rate) is provided on an annual basis.
"""
def compute_performance_stats(returns_df, rf_rate=0.0):
    metrics = {}
    
    # Annualization convention: 4 periods per year (quarterly compounding)
    # Each 3-month holding period represents one quarter.
    # Alternatively, a trading-day basis assumes 252 days/year ≈ 63 days/quarter.

    periods_per_year = 4
    
    for portfolio in returns_df.columns:
        rets = returns_df[portfolio].values
        
        # Convert to log returns for annualization
        log_returns = np.log(1 + rets)
        
        # Annualized return
        mean_log_return = np.mean(log_returns)
        ann_return = np.exp(mean_log_return * periods_per_year) - 1
        
        # Annualized volatility
        std_log_return = np.std(log_returns, ddof=1)
        ann_volatility = std_log_return * np.sqrt(periods_per_year)
        
        # Sharpe ratio (assuming rf_rate is annual)
        sharpe_ratio = (ann_return - rf_rate) / ann_volatility if ann_volatility > 0 else 0
        
        metrics[portfolio] = {
            'Mean Return (Ann.)': ann_return,
            'Std Dev (Ann.)': ann_volatility,
            'Sharpe Ratio': sharpe_ratio
        }
    
    # Information ratio relative to NIFTY50
    nifty_returns = returns_df['NIFTY50'].values
    for portfolio in returns_df.columns:
        if portfolio == 'NIFTY50':
            metrics[portfolio]['Information Ratio'] = np.nan
        else:
            excess_rets = returns_df[portfolio].values - nifty_returns
            tracking_error = np.std(excess_rets, ddof=1) * np.sqrt(periods_per_year)
            avg_excess = np.mean(excess_rets) * periods_per_year
            ir = avg_excess / tracking_error if tracking_error > 0 else 0
            metrics[portfolio]['Information Ratio'] = ir
    
    return pd.DataFrame(metrics).T


"""Plot cumulative returns of portfolios"""
def plot_cum_returns(results_df, save_path=None):
    cumulative_returns = (1 + results_df).cumprod()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative_returns.plot(ax=ax)
    
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)


"""Plot VaR vs Realized returns"""
def plot_portfolio_var_backtest(var_results, save_path=None):
    portfolios = ['GMV', 'MV', 'EW', 'Active', 'NIFTY50']
    
    # 3x2 grid for 5 portfolios
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, portfolio in enumerate(portfolios):
        ax = axes[idx]

        var_values = -np.array(var_results[portfolio]['var'])
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

    for j in range(len(portfolios), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
