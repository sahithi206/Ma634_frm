import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def load_and_match_dates(stocks_file, factors_file):
    """Load data and match trading dates across all files"""
    # Load stock data
    stocks_df = pd.read_csv(stocks_file)
    # Handle both "Date" and "Dates" column names
    if 'Dates' in stocks_df.columns:
        stocks_df = stocks_df.rename(columns={'Dates': 'Date'})
    # Handle NIFTY column name variations
    if 'NIFTY Index' in stocks_df.columns:
        stocks_df = stocks_df.rename(columns={'NIFTY Index': 'NIFTY 50'})
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'], dayfirst=True)
    
    # Load market factor and risk-free data
    factors_df = pd.read_csv(factors_file)
    factors_df['Date'] = pd.to_datetime(factors_df['Date'], dayfirst=True)
    
    # Find common dates
    common_dates = set(stocks_df['Date']).intersection(set(factors_df['Date']))
    common_dates = sorted(list(common_dates))
    
    # Filter to common dates
    stocks_df = stocks_df[stocks_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
    factors_df = factors_df[factors_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
    
    print(f"Common trading dates: {len(common_dates)}")
    print(f"Date range: {common_dates[0]} to {common_dates[-1]}")
    
    return stocks_df, factors_df

def calculate_returns(prices_df):
    """
    Calculate simple returns from prices
    
    Return Definition:
    - Uses simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}
    - NOT log returns: r_t = log(P_t / P_{t-1})
    - This matches assignment requirement: "use daily simple returns"
    
    Compounding:
    - For multi-period returns, we compound simple returns:
      R_{0,T} = (1+R_1)*(1+R_2)*...*(1+R_T) - 1
    """
    returns_df = prices_df.copy()
    returns_df.iloc[:, 1:] = prices_df.iloc[:, 1:].pct_change()  # Simple returns
    return returns_df.iloc[1:]  # Drop first row with NaN

def remove_incomplete_stocks(stocks_df, threshold=0.95):
    """Remove stocks with too many missing values"""
    stock_columns = [col for col in stocks_df.columns if col != 'Date']
    
    valid_stocks = []
    for col in stock_columns:
        if col == 'NIFTY 50':
            valid_stocks.append(col)
            continue
        non_missing_ratio = stocks_df[col].notna().sum() / len(stocks_df)
        if non_missing_ratio >= threshold:
            valid_stocks.append(col)
        else:
            print(f"Removing {col}: {non_missing_ratio:.2%} data available")
    
    return stocks_df[['Date'] + valid_stocks]
