"""
Script to create VaR backtest plot from var_backtest.csv
Similar to the plot_var_backtest function in main_code.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read var_backtest.csv
var_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'var_backtest.csv'))

# Read var_violations.txt to get violation counts
violations = {}
if os.path.exists(os.path.join(OUTPUT_DIR, 'var_violations.txt')):
    with open(os.path.join(OUTPUT_DIR, 'var_violations.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                portfolio = parts[0].strip()
                count = int(parts[1].strip().split()[0])
                violations[portfolio] = count

# Create plot similar to main_code.py
def plot_var_backtest(var_df, violations):
    """Plot VaR vs realized returns"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    portfolios = ['GMV', 'MV', 'EW', 'Active']
    portfolio_mapping = {
        'GMV': ('VaR_GMV', 'Real_GMV'),
        'MV': ('VaR_MV', 'Real_MV'),
        'EW': ('VaR_EW', 'Real_EW'),
        'Active': ('VaR_Active', 'Real_Active')
    }
    
    for idx, portfolio in enumerate(portfolios):
        ax = axes[idx]
        
        var_col, real_col = portfolio_mapping[portfolio]
        var_values = -np.array(var_df[var_col].values)  # Plot as negative
        realized = np.array(var_df[real_col].values)
        
        # Get violation count
        violation_count = violations.get(portfolio, 0)
        total_windows = len(var_values)
        
        x = range(len(var_values))
        
        ax.plot(x, var_values, 'r-', label='99% VaR (Negative)', linewidth=2)
        ax.plot(x, realized, 'b-', label='Realized Return', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Highlight violations: violation occurs if realized < -VaR (i.e., realized < var_values)
        violation_mask = realized < var_values
        if np.any(violation_mask):
            ax.scatter(np.array(x)[violation_mask], realized[violation_mask],
                      color='red', s=50, zorder=5, label='VaR Violation')
        
        ax.set_xlabel('Window Number', fontsize=10)
        ax.set_ylabel('Return', fontsize=10)
        ax.set_title(f'{portfolio} Portfolio (Violations: {violation_count}/{total_windows})', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'var_backtest.png'), dpi=300, bbox_inches='tight')
    print(f"VaR backtest plot saved to {os.path.join(OUTPUT_DIR, 'var_backtest.png')}")
    plt.close()

if __name__ == "__main__":
    plot_var_backtest(var_df, violations)