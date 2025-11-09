import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.linalg import pinv
from numpy.linalg import LinAlgError
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Output directory: store all results in gpt_test/output
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
TRADING_DAYS_PER_YEAR = 252
FORMATION_DAYS_APPROX = int(252/2)  # ~126 trading days ~ 6 months
HOLDING_DAYS_APPROX = int(252/4)    # ~63 trading days ~ 3 months
ALPHA = 0.05
SHRINKAGE_EPS = 1e-6

# File names: read from parent directory
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # Go up one level to FRM_Project

STOCKS_FILE = os.path.join(PARENT_DIR, "Stocks_data.csv")
FACTORS_FILE = os.path.join(PARENT_DIR, "market_Factor_risk_Free.csv")

# Helper functions

def read_and_align(stocks_file, factors_file):
    # Read stocks (daily close prices) and factors
    stocks = pd.read_csv(stocks_file)
    factors = pd.read_csv(factors_file)

    # Handle date column names: could be "Date" or "Dates"
    if stocks.columns[0].lower() in ['dates', 'date', 'dt']:
        stocks.rename(columns={stocks.columns[0]: 'Date'}, inplace=True)
    else:
        # If first column doesn't look like date, try to find it
        for col in stocks.columns:
            if col.lower() in ['dates', 'date', 'dt']:
                stocks.rename(columns={col: 'Date'}, inplace=True)
                break
    
    # Handle NIFTY column name variations
    for col in stocks.columns:
        if 'nifty' in col.lower() and 'index' in col.lower():
            stocks.rename(columns={col: 'NIFTY 50'}, inplace=True)
            break

    # Parse dates with dayfirst=True for DD-MM-YYYY format
    stocks['Date'] = pd.to_datetime(stocks['Date'], dayfirst=True)
    factors['Date'] = pd.to_datetime(factors['Date'], dayfirst=True)

    # Merge on intersection of dates
    # Keep only trading dates present in all files
    common_dates = set(stocks['Date']).intersection(set(factors['Date']))
    # Some stocks file contains index in last column; we'll keep it as NIFTY50 if present
    stocks = stocks[stocks['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
    factors = factors[factors['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

    # Reindex both by date
    stocks.set_index('Date', inplace=True)
    factors.set_index('Date', inplace=True)

    # Ensure same index order
    stocks = stocks.loc[factors.index]

    return stocks, factors


def compute_daily_returns(prices_df):
    # Simple returns (not log)
    returns = prices_df.pct_change().dropna(how='all')
    return returns


def safe_invert_cov(cov, eps=SHRINKAGE_EPS):
    # Add tiny ridge if ill-conditioned
    try:
        inv = np.linalg.inv(cov)
    except LinAlgError:
        cov2 = cov + eps * np.eye(cov.shape[0])
        inv = np.linalg.inv(cov2)
    return inv


def gmvp_weights(cov):
    # unconstrained GMV: w = inv(Sigma) 1 / (1' inv(Sigma) 1)
    inv = safe_invert_cov(cov)
    ones = np.ones(inv.shape[0])
    w = inv.dot(ones)
    w = w / (ones.dot(w))
    return w


def tangency_weights(mu_excess, cov):
    # unconstrained tangency: w ~ inv(Sigma) * mu_excess; normalized to sum to 1
    inv = safe_invert_cov(cov)
    raw = inv.dot(mu_excess)
    if raw.sum() == 0:
        # fallback to EW
        w = np.ones_like(raw) / len(raw)
    else:
        w = raw / raw.sum()
    return w


def equal_weighted(n):
    return np.ones(n) / n


def active_portfolio_weights(returns_window, mf_series, rf_series, alpha=ALPHA):
    # returns_window: DataFrame with columns = stock tickers, index aligned with factors
    selected = []
    for col in returns_window.columns:
        y = returns_window[col].values - rf_series.values
        X = sm.add_constant(mf_series.values)
        # If y is NaN or all zeros skip
        if np.isnan(y).all() or np.isfinite(y).sum() < 5:
            continue
        try:
            res = sm.OLS(y, X, missing='drop').fit()
        except Exception:
            continue
        # Check two-sided significance
        if 'const' in res.params.index if isinstance(res.params, pd.Series) else True:
            t_alpha = res.tvalues[0]
            p_alpha = res.pvalues[0]
            if p_alpha < alpha:
                # For active, include if alpha significantly > 0 (outperformance)
                if res.params[0] > 0:
                    selected.append(col)
    if len(selected) == 0:
        return None, []
    w = np.ones(len(selected)) / len(selected)
    weights = pd.Series(0.0, index=returns_window.columns)
    for i, s in enumerate(selected):
        weights[s] = w[i]
    return weights.values, selected


def portfolio_return_from_weights(weights, returns_df):
    # returns_df rows are daily returns; compute compounded return over the period
    # weights aligned to columns order
    # daily portfolio returns
    port_daily = returns_df.values.dot(weights)
    total = np.prod(1 + port_daily) - 1
    return total, port_daily


def compute_L_day_historical_var(port_daily_returns, L, alpha=0.01):
    # port_daily_returns: 1D array covering formation window days
    # compute rolling L-day compounded returns within formation window
    if len(port_daily_returns) < L:
        return None
    rolling = []
    for start in range(0, len(port_daily_returns) - L + 1):
        seq = port_daily_returns[start:start+L]
        r = np.prod(1 + seq) - 1
        rolling.append(r)
    rolling = np.array(rolling)
    # VaR at 99% (alpha=0.01): the loss that is exceeded with prob 1% -> take 1st percentile of returns
    q = np.percentile(rolling, 1)
    # VaR expressed as positive number (loss); we will return -q if q<0 else small
    VaR = -q if q < 0 else 0.0
    return VaR, rolling


# Main pipeline
if __name__ == '__main__':
    stocks_raw, factors = read_and_align(STOCKS_FILE, FACTORS_FILE)

    # Identify NIFTY50 column if present (case-insensitive 'nifty')
    nifty_col = None
    for c in stocks_raw.columns[::-1]:
        if 'nifty' in c.lower() or 'index' in c.lower():
            nifty_col = c
            break

    if nifty_col is None:
        raise RuntimeError('Could not locate NIFTY column in Stocks_data.csv. Please ensure the index column is present.')

    # Separate stock prices and index
    stock_prices = stocks_raw.drop(columns=[nifty_col])
    nifty_prices = stocks_raw[[nifty_col]]

    # Drop stocks with many missing values (strict rule: require full coverage across period)
    # Alternatively you may drop those with > 1% missing
    allowed_missing_fraction = 0.01
    good_cols = [c for c in stock_prices.columns if stock_prices[c].isna().mean() <= allowed_missing_fraction]
    stock_prices = stock_prices[good_cols]

    # Forward/backfill small gaps, then drop remaining NAs
    stock_prices = stock_prices.ffill().bfill()

    # Compute daily simple returns
    stock_returns = compute_daily_returns(stock_prices)
    nifty_returns = compute_daily_returns(nifty_prices)
    # Align factors index with returns (drop first day mismatch)
    factors = factors.loc[stock_returns.index]

    mf = factors['MF'] / 100.0 if factors['MF'].abs().mean() > 1 else factors['MF']  # detect if MF in percent
    rf = factors['RF']  # assume RF in decimal

    # Rolling windows: formation length = approx 6 months (we will pick exact based on trading days)
    # We'll generate formation windows by calendar: start at Jan 2009: formation Jan-Jun 2009 -> holding Jul-Sep 2009
    # Simpler implementation: generate list of formation start dates by stepping every 3 months

    # Build a list of (formation_start, formation_end, holding_start, holding_end) using calendar months
    def month_add(dt, months):
        y = dt.year + (dt.month - 1 + months) // 12
        m = (dt.month - 1 + months) % 12 + 1
        return datetime(y, m, 1)

    # Determine earliest and latest dates available
    first_date = stock_returns.index[0]
    last_date = stock_returns.index[-1]

    # We'll construct windows by months: formation = 6 months inclusive, holding next 3 months
    windows = []
    # Start formation at Jan 2009 first business day
    start = datetime(2009, 1, 1)
    # Move start to first available trading date on/after start
    if start not in stock_returns.index:
        start = stock_returns.index[stock_returns.index.searchsorted(start)]

    while True:
        formation_start = start
        formation_end_month = month_add(formation_start, 6)
        formation_end = formation_end_month - pd.Timedelta(days=1)
        # snap to nearest trading days inside our data
        # find last trading day <= formation_end
        formation_end = stock_returns.index[stock_returns.index.searchsorted(formation_end, side='right') - 1]

        holding_start = formation_end + pd.Timedelta(days=1)
        holding_end_month = month_add(formation_end + pd.Timedelta(days=1), 3)
        holding_end = holding_end_month - pd.Timedelta(days=1)
        holding_end = stock_returns.index[stock_returns.index.searchsorted(holding_end, side='right') - 1]

        if holding_end > last_date:
            break

        # Ensure we have at least 1 day
        if formation_start >= formation_end or holding_start >= holding_end:
            start = month_add(start, 3)
            if start not in stock_returns.index:
                if stock_returns.index.searchsorted(start) >= len(stock_returns.index):
                    break
                start = stock_returns.index[stock_returns.index.searchsorted(start)]
            continue

        windows.append((formation_start, formation_end, holding_start, holding_end))

        # shift start by 3 months
        start = month_add(start, 3)
        if start not in stock_returns.index:
            pos = stock_returns.index.searchsorted(start)
            if pos >= len(stock_returns.index):
                break
            start = stock_returns.index[pos]

    print(f"Total windows: {len(windows)}")

    results = []
    var_records = []

    for (fstart, fend, hstart, hend) in windows:
        # Select formation and holding slices
        formation_prices = stock_prices.loc[fstart:fend]
        formation_returns = stock_returns.loc[fstart:fend]
        holding_returns = stock_returns.loc[hstart:hend]
        holding_nifty_ret = nifty_returns.loc[hstart:hend]

        # Align factors
        mf_f = mf.loc[fstart:fend]
        rf_f = rf.loc[fstart:fend]
        rf_h = rf.loc[hstart:hend]

        # Compute sample mean and covariance (on formation-window simple returns)
        mu = formation_returns.mean().values  # daily mean
        cov = formation_returns.cov().values

        # If covariance singular, add small diag
        cov = cov + SHRINKAGE_EPS * np.eye(cov.shape[0])

        # GMV
        try:
            w_gmv = gmvp_weights(cov)
        except Exception:
            w_gmv = equal_weighted(len(mu))

        # Tangency: use excess mean = mu - avg_rf_formation
        avg_rf_f = rf_f.mean()
        mu_excess = mu - avg_rf_f
        w_tan = tangency_weights(mu_excess, cov)

        # EW
        w_ew = equal_weighted(len(mu))

        # Active
        w_active, selected = active_portfolio_weights(formation_returns, mf_f, rf_f)
        if w_active is None:
            # use market portfolio: we use the NIFTY index daily returns for holding return; for weights, we cannot reproduce index weights
            # We'll indicate by None and later copy NIFTY realized return
            w_active = None

        # Compute realized holding returns for portfolios
        cols = formation_returns.columns
        # GMV
        rg_gmv, daily_gmv = portfolio_return_from_weights(w_gmv, holding_returns[cols])
        rg_tan, daily_tan = portfolio_return_from_weights(w_tan, holding_returns[cols])
        rg_ew, daily_ew = portfolio_return_from_weights(w_ew, holding_returns[cols])

        if w_active is None:
            # Use NIFTY realized return as active
            rg_act = np.prod(1 + holding_nifty_ret.values.flatten()) - 1
            daily_act = holding_nifty_ret.values.flatten()
        else:
            rg_act, daily_act = portfolio_return_from_weights(w_active, holding_returns[cols])

        rg_nifty = np.prod(1 + holding_nifty_ret.values.flatten()) - 1

        results.append({
            'formation_start': fstart, 'formation_end': fend,
            'holding_start': hstart, 'holding_end': hend,
            'GMV': rg_gmv, 'MV_Tangency': rg_tan, 'EW': rg_ew, 'Active': rg_act, 'NIFTY50': rg_nifty,
            'active_selected': ";".join(selected) if selected else ''
        })

        # VaR estimation using historical simulation of L-day returns within formation window
        L = len(holding_returns)
        # For each portfolio, compute formation-period daily portfolio returns
        # GMV formation daily
        _, port_daily_gmv_f = portfolio_return_from_weights(w_gmv, formation_returns[cols])
        var_gmv = compute_L_day_historical_var(port_daily_gmv_f, L)

        _, port_daily_tan_f = portfolio_return_from_weights(w_tan, formation_returns[cols])
        var_tan = compute_L_day_historical_var(port_daily_tan_f, L)

        _, port_daily_ew_f = portfolio_return_from_weights(w_ew, formation_returns[cols])
        var_ew = compute_L_day_historical_var(port_daily_ew_f, L)

        if w_active is None:
            # For active==market, compute historical L-day returns of NIFTY in formation window
            nifty_f = nifty_returns.loc[fstart:fend].values.flatten()
            var_act = compute_L_day_historical_var(nifty_f, L)
        else:
            _, port_daily_act_f = portfolio_return_from_weights(w_active, formation_returns[cols])
            var_act = compute_L_day_historical_var(port_daily_act_f, L)

        # Record
        var_records.append({
            'formation_start': fstart, 'formation_end': fend,
            'holding_start': hstart, 'holding_end': hend,
            'L_days': L,
            'VaR_GMV': var_gmv[0] if var_gmv is not None else np.nan,
            'VaR_MV': var_tan[0] if var_tan is not None else np.nan,
            'VaR_EW': var_ew[0] if var_ew is not None else np.nan,
            'VaR_Active': var_act[0] if var_act is not None else np.nan,
            'Real_GMV': rg_gmv, 'Real_MV': rg_tan, 'Real_EW': rg_ew, 'Real_Active': rg_act
        })

    df_results = pd.DataFrame(results)
    df_var = pd.DataFrame(var_records)

    # Save rolling returns matrix (n x 5)
    rolling_returns = df_results[['GMV', 'MV_Tangency', 'EW', 'Active', 'NIFTY50']].copy()
    rolling_returns.to_csv(os.path.join(OUTPUT_DIR, 'rolling_returns.csv'), index=False)

    # Performance reporting
    R = rolling_returns.values  # rows: windows, cols: portfolios
    port_names = ['GMV', 'MV_Tangency', 'EW', 'Active', 'NIFTY50']

    # Cumulative returns plot: compound sequentially the 3-month holding returns
    cum = []
    for j in range(R.shape[1]):
        seq = R[:, j]
        cumrets = np.cumprod(1 + seq) - 1
        cum.append(cumrets)
    plt.figure(figsize=(10,6))
    for j,name in enumerate(port_names):
        plt.plot(df_results['holding_end'], cum[j], label=name)
    plt.legend()
    plt.xlabel('Holding end')
    plt.ylabel('Cumulative return')
    plt.title('Cumulative returns (sequential compounding of 3-month holding returns)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cumulative_returns.png'))
    plt.close()

    # Performance table: mean, std, Sharpe, Info ratio (relative to NIFTY50)
    # Annualization choices: each window is ~3 months; map to annual factors using 252 trading days and average holding length
    avg_L = df_var['L_days'].mean()
    months_per_year = 12
    windows_per_year = int(252 / avg_L * months_per_year / 12) if avg_L>0 else 4
    # We'll annualize using log-returns method for mean and sqrt scaling for std

    perf_rows = []
    # Annual risk-free: use mean daily RF across all data * 252
    RF_annual = rf.mean() * TRADING_DAYS_PER_YEAR

    # For info ratio, compute active mean difference and std of differences
    nifty_col_idx = port_names.index('NIFTY50')

    for j,name in enumerate(port_names):
        series = R[:, j]
        # convert to log returns per window
        logr = np.log1p(series)
        mean_log = np.nanmean(logr)
        mean_annual = np.expm1(mean_log * (TRADING_DAYS_PER_YEAR / avg_L))
        std_window = np.nanstd(series, ddof=1)
        std_annual = std_window * np.sqrt(TRADING_DAYS_PER_YEAR / avg_L)
        sharpe = (mean_annual - RF_annual) / std_annual if std_annual>0 else np.nan

        # information ratio relative to NIFTY
        if name != 'NIFTY50':
            diff = series - R[:, nifty_col_idx]
            mean_diff = np.nanmean(diff)
            # annualize mean diff via log approx
            mean_diff_log = np.nanmean(np.log1p(diff)) if np.all(diff>-1) else mean_diff * (TRADING_DAYS_PER_YEAR / avg_L)
            mean_diff_annual = mean_diff * (TRADING_DAYS_PER_YEAR / avg_L)
            std_diff = np.nanstd(diff, ddof=1)
            std_diff_annual = std_diff * np.sqrt(TRADING_DAYS_PER_YEAR / avg_L)
            info = mean_diff_annual / std_diff_annual if std_diff_annual>0 else np.nan
        else:
            info = np.nan

        perf_rows.append({'Portfolio': name, 'Mean_Annual': mean_annual, 'Std_Annual': std_annual, 'Sharpe': sharpe, 'Info_vs_NIFTY': info})

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_table.csv'), index=False)

    # VaR backtest: count violations
    violations = {}
    for idx, row in df_var.iterrows():
        # For each window
        for p in ['GMV','MV','EW','Active']:
            var_col = f'VaR_{p}' if p!='MV' else 'VaR_MV'
            real_col = f'Real_{"GMV" if p=="GMV" else ("MV" if p=="MV" else ("EW" if p=="EW" else "Active"))}'
            VaR = row[var_col]
            Real = row[real_col]
            key = p
            if key not in violations:
                violations[key] = 0
            if np.isnan(VaR):
                continue
            # VaR expressed as positive loss; violation if realized return < -VaR
            if Real < -VaR:
                violations[key] += 1

    # Save var results and violation counts
    df_var.to_csv(os.path.join(OUTPUT_DIR, 'var_backtest.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'var_violations.txt'), 'w') as f:
        for k,v in violations.items():
            f.write(f"{k}: {v} violations\n")
    print('Done. Outputs written to', OUTPUT_DIR)