import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_dates(stocks_file, factors_file):
    """Load data and Preprocess trading dates across all files"""
    
    prices_df = pd.read_csv(stocks_file) # Load stock data

    if 'Dates' in prices_df.columns:
        prices_df = prices_df.rename(columns={'Dates': 'Date'})
    prices_df['Date'] = pd.to_datetime(prices_df['Date'], dayfirst=True)

    factors_df = pd.read_csv(factors_file) # Load market factor and risk-free data
    factors_df['Date'] = pd.to_datetime(factors_df['Date'], dayfirst=True)
    
    # Find common dates
    common_dates = sorted(set(factors_df['Date']).intersection(set(prices_df['Date'])))
    print(f"Total common trading dates found: {len(common_dates)}")
    
    # Filter to common dates
    prices_df = prices_df[prices_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
    factors_df = factors_df[factors_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

    print(f"Common trading dates: {len(common_dates)}")
    return prices_df, factors_df

def compute_returns(prices_df):
    """compute simple returns from prices : R_t=(P_t - P_{t-1})/P_{t-1}"""
    returns =prices_df.copy()
    price_cols=returns.columns[1:]
    returns[price_cols] = returns[price_cols].pct_change()
    returns = returns.iloc[1:]
    return returns

def remove_nan_cols(prices_df, threshold=0.95, keep_cols=None):
    """
    Remove stock columns with NaN percentage > (1 - threshold).
    keep_cols = columns to always keep (e.g., ['NIFTY Index'])
    """
    if keep_cols is None:
        keep_cols = []

    price_cols = prices_df.columns[1:]
    not_nan_ratio = prices_df[price_cols].notna().mean()
    valid_cols = [
        col for col in price_cols
        if col in keep_cols or not_nan_ratio[col] >= threshold
    ]

    return prices_df[['Date'] + valid_cols]


