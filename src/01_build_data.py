# 01_build_data.py  (Member 1 - Data lead)
#
# Builds the two files Sophia's scripts (03 and 04) read. Columns follow DATA_CONTRACT.md:
#   data/clean/fund_month.parquet  one row per fund per month
#   data/clean/fund_day.parquet    one row per fund per trading day
#
# Run from the repo's top folder:   python src/01_build_data.py
#
# Inputs (committed to git, because data/ is git-ignored), made by src/00_build_ticker_list.py:
#   inputs/tickers.csv          column: ticker   (active US equity mutual funds, one share class each)
#   inputs/expense_ratios.csv   columns: ticker, exp_ratio, named_bench, manager, source_url, date_checked
#                               exp_ratio is a decimal (0.0075 = 0.75%). manager looks like "Jane Smith (since 1998)"
#
# Cleaning rules (each one prints how many rows it removes - put the numbers in the report):
#   C1  Days with no price are dropped, never filled.
#   C2  Bad Yahoo prices: if a fund's DAILY return is beyond +/-20% (the biggest real market day in our
#       sample is under 10%), that day and that fund's whole month are removed from both files.
#   C3  Only dates that have French factors are kept.
import os
import time
from datetime import date

import pandas as pd
import yfinance as yf

os.makedirs('data/raw', exist_ok=True)      # git does not keep empty folders
os.makedirs('data/clean', exist_ok=True)

FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'


# ======================= Step 1: ticker list =======================
tickers_df = pd.read_csv('inputs/tickers.csv')
tickers = list(tickers_df['ticker'].str.strip().str.upper())
print('Funds on our list:', len(tickers))


# ======================= Step 2: download prices ONCE =======================
# If the file is already there we never call Yahoo again (Yahoo rate-limits).
price_file = 'data/raw/prices.parquet'

if os.path.exists(price_file):
    print('prices.parquet already exists - skipping the download')
else:
    all_tickers = tickers + ['SPY']
    batches = []
    for start in range(0, len(all_tickers), 50):          # 50 tickers at a time
        batch = all_tickers[start:start + 50]
        print('Downloading', start, 'to', start + len(batch))
        data = yf.download(batch, start='1994-12-01', auto_adjust=False, progress=False)
        batches.append(data['Adj Close'])                  # adjusted close includes distributions
        time.sleep(2)                                      # be polite to Yahoo
    prices = pd.concat(batches, axis=1)
    prices.to_parquet(price_file)

    # source log: the plan says to write down when we downloaded
    with open('data/raw/source_log.txt', 'a') as log:
        log.write('Yahoo prices downloaded on ' + str(date.today()) + '\n')
    print('Saved', price_file)

prices = pd.read_parquet(price_file)
print('Price table:', prices.shape[0], 'days x', prices.shape[1], 'tickers')

# keep only the funds on today's list (+ SPY): a fund dropped from tickers.csv after the download
# (e.g. by exclude_manual.csv) is still in the saved price file and must not reach the clean files
keep_cols = [c for c in prices.columns if c in tickers or c == 'SPY']
print('Price columns not on the ticker list (left out):', prices.shape[1] - len(keep_cols))
prices = prices[keep_cols]


# ======================= Step 3: which tickers have no data? =======================
# These are download failures. They are NOT a survivorship count: the screener only lists funds
# that still exist today, so dead funds never reach this list (the report must say so).
no_data = []
for t in tickers:
    if t not in prices.columns or prices[t].notna().sum() == 0:
        no_data.append(t)
print('Tickers with NO price data:', len(no_data), no_data)

# first month with a price, for every fund (before we cut to the two eras)
first_rows = []
for t in tickers:
    if t not in no_data:
        first_day = prices[t].first_valid_index()
        first_rows.append({'ticker': t, 'first_month': first_day.strftime('%Y-%m')})
first_month = pd.DataFrame(first_rows)


# ======================= Step 4: daily and monthly returns =======================
# fill_method=None: a missing price stays missing (same answer in pandas 2 and 3).
daily_ret = prices.pct_change(fill_method=None)

month_prices = prices.resample('ME').last()       # last price of each month ('M' on old pandas)
monthly_ret = month_prices.pct_change(fill_method=None)

# keep only the two eras now (dot-com 1995-2002, AI 2022 on), so the long tables stay small.
# Sophia's quant_common.py cuts the partial year 2026 itself.
def in_eras(dates):
    return ((dates.year >= 1995) & (dates.year <= 2002)) | (dates.year >= 2022)

daily_ret = daily_ret[in_eras(daily_ret.index)]
monthly_ret = monthly_ret[in_eras(monthly_ret.index)]


# ======================= Step 5: wide table -> long table =======================
daily_ret = daily_ret.reset_index()
day = daily_ret.melt(id_vars='Date', var_name='ticker', value_name='ret')
day = day.rename(columns={'Date': 'date'})
rows_before = len(day)
day = day.dropna(subset=['ret'])                   # C1: drop days with no price, never fill
print('C1 removed', rows_before - len(day), 'empty fund-days (days before a fund existed, gaps)')

monthly_ret = monthly_ret.reset_index()
month = monthly_ret.melt(id_vars='Date', var_name='ticker', value_name='ret_net')
month['month'] = month['Date'].dt.strftime('%Y-%m')
month = month.drop(columns='Date')
month = month.dropna(subset=['ret_net'])

# SPY becomes its own column, bench_ret, next to every fund-month
spy = month[month['ticker'] == 'SPY'][['month', 'ret_net']]
spy = spy.rename(columns={'ret_net': 'bench_ret'})
month = month[month['ticker'] != 'SPY']
month = month.merge(spy, on='month', how='left')

day = day[day['ticker'] != 'SPY']                  # the daily file has no benchmark column

# C2: bad Yahoo prices. Find fund-days beyond +/-20%, then remove the whole fund-month.
day['month'] = day['date'].dt.strftime('%Y-%m')
bad_days = day[day['ret'].abs() > 0.20]
bad_months = bad_days[['ticker', 'month']].drop_duplicates()
bad_months['bad'] = True
bad_days[['ticker', 'date', 'ret']].to_csv('data/raw/bad_price_days.csv', index=False)

month = month.merge(bad_months, on=['ticker', 'month'], how='left')
day = day.merge(bad_months, on=['ticker', 'month'], how='left')
print('C2 found', len(bad_days), 'fund-days beyond +/-20% in', bad_days['ticker'].nunique(), 'funds')
print('C2 removed', month['bad'].notna().sum(), 'fund-months and', day['bad'].notna().sum(), 'fund-days')
month = month[month['bad'].isna()].drop(columns='bad')
day = day[day['bad'].isna()].drop(columns=['bad', 'month'])


# ======================= Step 6: Ken French files =======================
# French files start with a few lines of text, then the data, then extra sections
# (annual numbers, or a second block of equal-weighted returns), then a copyright line.
# This function keeps only the first block of real data rows.
def read_french(file_name, date_length):
    # names=range(20): read every line into 20 columns, so short text lines don't cause errors
    raw = pd.read_csv(FRENCH + file_name, header=None, names=list(range(20)), dtype=str)

    # the header row is the first row whose first cell is empty: ",Mkt-RF,SMB,HML,RF"
    header = raw[raw[0].isna() & raw[1].notna()].iloc[0]
    header = header.dropna()                       # keep only the cells that have a name
    names = list(header.str.strip())

    # data rows start with a date: 8 digits (daily, 20260930) or 6 digits (monthly, 202609)
    raw[0] = raw[0].fillna('').str.strip()
    is_date = raw[0].str.isdigit() & (raw[0].str.len() == date_length)
    rows = raw[is_date].copy()
    rows = rows[[0] + list(header.index)]          # first column + the named columns
    rows.columns = ['date'] + names
    rows = rows.drop_duplicates(subset='date', keep='first')   # keep the first block only

    for col in names:
        rows[col] = pd.to_numeric(rows[col].str.strip())
    return rows


ff_day = read_french('F-F_Research_Data_Factors_daily_CSV.zip', 8)
ind_day = read_french('12_Industry_Portfolios_daily_CSV.zip', 8)
ff_month = read_french('F-F_Research_Data_Factors_CSV.zip', 6)
ind_month = read_french('12_Industry_Portfolios_CSV.zip', 6)

# keep and rename the columns we need
ff_day = ff_day[['date', 'Mkt-RF', 'SMB', 'HML', 'RF']]
ff_month = ff_month[['date', 'Mkt-RF', 'SMB', 'HML', 'RF']]
ind_day = ind_day[['date', 'BusEq']]
ind_month = ind_month[['date', 'BusEq']]

ff_day = ff_day.merge(ind_day, on='date', how='left')
ff_month = ff_month.merge(ind_month, on='date', how='left')

new_names = {'Mkt-RF': 'mkt_rf', 'SMB': 'smb', 'HML': 'hml', 'RF': 'rf', 'BusEq': 'tech_ret'}
ff_day = ff_day.rename(columns=new_names)
ff_month = ff_month.rename(columns=new_names)

# French publishes PERCENTS. The contract wants decimals.
for col in ['mkt_rf', 'smb', 'hml', 'rf', 'tech_ret']:
    ff_day[col] = ff_day[col] / 100
    ff_month[col] = ff_month[col] / 100

# dates in the same format as our fund data
ff_day['date'] = pd.to_datetime(ff_day['date'], format='%Y%m%d')
ff_month['month'] = ff_month['date'].str[:4] + '-' + ff_month['date'].str[4:]
ff_month = ff_month.drop(columns='date')

with open('data/raw/source_log.txt', 'a') as log:
    log.write('French factors read on ' + str(date.today()) + ', last month ' + ff_month['month'].max() + '\n')
print('French monthly factors end in', ff_month['month'].max())


# ======================= Step 7: merge everything =======================
# C3, inner merge: keep only dates that have BOTH a fund return and French factors
rows_before = len(month)
day = day.merge(ff_day, on='date', how='inner')
month = month.merge(ff_month, on='month', how='inner')
print('C3 removed', rows_before - len(month), 'fund-months with no French factors (after their last month)')

# expense ratio, named benchmark and manager (filled in by hand from prospectuses)
if os.path.exists('inputs/expense_ratios.csv'):
    info = pd.read_csv('inputs/expense_ratios.csv')
    info = info[['ticker', 'exp_ratio', 'named_bench', 'manager']]
else:
    print('WARNING: inputs/expense_ratios.csv not found - exp_ratio, named_bench, manager left empty')
    info = pd.DataFrame({'ticker': tickers, 'exp_ratio': None, 'named_bench': None, 'manager': None})
month = month.merge(info, on='ticker', how='left')
month = month.merge(first_month, on='ticker', how='left')

missing_er = month[month['exp_ratio'].isna()]['ticker'].unique()
print('Funds still missing an expense ratio:', len(missing_er))


# ======================= Step 8: year column for the checks =======================
month['year'] = month['month'].str[:4].astype(int)


# ======================= Step 9: checks =======================
# Sophia's loader stops if any return is above 1.0 (100%). After C2 there should be none.
too_big = month[month['ret_net'].abs() > 1.0]
if len(too_big) > 0:
    print('WARNING: returns above 100% are left - Sophia\'s loader will stop:')
    print(too_big[['ticker', 'month', 'ret_net']])

big_months = month[month['ret_net'].abs() > 0.30]
print('Monthly returns bigger than 30% (look at them; Feb 2000 tech moves are real):', len(big_months))
print(big_months[['ticker', 'month', 'ret_net']].head(20))

# months where a fund is more than 10 points away from SPY: often a Yahoo distribution error
far = month[(month['ret_net'] - month['bench_ret']).abs() > 0.10]
print('Fund-months more than 10 points away from SPY (check by hand):', len(far))
far[['ticker', 'month', 'ret_net', 'bench_ret']].to_csv('data/raw/months_to_check.csv', index=False)

print('Funds per year (monthly file) - Member 3 needs these counts:')
print(month.groupby('year')['ticker'].nunique())


# ======================= Step 10: save in the contract's exact columns =======================
month_columns = ['ticker', 'month', 'ret_net', 'bench_ret', 'rf', 'mkt_rf', 'smb', 'hml',
                 'tech_ret', 'exp_ratio', 'named_bench', 'first_month', 'manager']
day_columns = ['ticker', 'date', 'ret', 'rf', 'mkt_rf', 'smb', 'hml', 'tech_ret']

month = month[month_columns].sort_values(['ticker', 'month'])
day = day[day_columns].sort_values(['ticker', 'date'])

month.to_parquet('data/clean/fund_month.parquet', index=False)
day.to_parquet('data/clean/fund_day.parquet', index=False)
print('Saved data/clean/fund_month.parquet:', len(month), 'rows')
print('Saved data/clean/fund_day.parquet:', len(day), 'rows')
