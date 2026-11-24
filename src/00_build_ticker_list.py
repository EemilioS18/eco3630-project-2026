# 00_build_ticker_list.py  (Member 1 - Data lead)
#
# Builds the fund list that 01_build_data.py downloads. Professor's guidance: scrape the mutual fund
# tickers from Yahoo (at least 1000), then separate active from passive (index) funds.
#
# Run from the repo's top folder:   python src/00_build_ticker_list.py
#
# Writes (committed to git, they are small):
#   inputs/tickers.csv                 one row per ACTIVE fund (one share class per fund)
#   inputs/ticker_selection_log.csv    every share class Yahoo listed, and why it was kept or dropped
#   inputs/expense_ratios.csv          Yahoo's expense ratio for each fund (only created if missing,
#                                      so hand corrections are never overwritten)
#   inputs/expense_ratio_spotcheck.csv 30 random funds to check by hand against the prospectus
#
# Rules (write these in the report):
#   R1  Universe: Yahoo's fund screener, the 9 Morningstar US equity style categories, Nasdaq-listed.
#   R2  One share class per fund: the oldest class (longest price history) is kept.
#       Limitation: this works on fund NAMES. Where Yahoo shortens a name differently for two classes
#       of the same fund ("Gro" vs "Growth"), that fund can still be counted twice.
#   R3  Passive funds and ETFs dropped: "Index", "Indx", "Idx" or "ETF" in the fund name.
#   R4  Funds with an expense ratio under 0.15% are printed for a manual look (possible index funds).
#   R5  Funds the group decided to drop after that look: inputs/exclude_manual.csv (ticker, reason).
import os
import random
import time
from datetime import date

import pandas as pd
import yfinance as yf
from yfinance import FundQuery

os.makedirs('data/raw', exist_ok=True)
os.makedirs('inputs', exist_ok=True)

CATEGORIES = ['Large Growth', 'Large Blend', 'Large Value',
              'Mid-Cap Growth', 'Mid-Cap Blend', 'Mid-Cap Value',
              'Small Growth', 'Small Blend', 'Small Value']
MIN_FUNDS = 1000


# ======================= Step 1: screen Yahoo ONCE =======================
# Same idea as the price download: save the raw result, never ask Yahoo twice.
screen_file = 'data/raw/screen_raw.csv'

if os.path.exists(screen_file):
    print('screen_raw.csv already exists - skipping the Yahoo screener')
else:
    rows = []
    for category in CATEGORIES:
        # "category is X AND listed on Nasdaq" (US mutual funds trade under Nasdaq tickers)
        query = FundQuery('and', [FundQuery('eq', ['categoryname', category]),
                                  FundQuery('eq', ['exchange', 'NAS'])])
        offset = 0
        while True:                                    # Yahoo returns at most 250 funds per call
            result = yf.screen(query, size=250, offset=offset)
            for quote in result['quotes']:
                rows.append({'ticker': quote.get('symbol'),
                             'long_name': quote.get('longName'),
                             'short_name': quote.get('shortName'),
                             'category': category,
                             'first_trade_ms': quote.get('firstTradeDateMilliseconds')})
            offset = offset + 250
            if offset >= result['total']:
                break
            time.sleep(1)
        print(category, ':', result['total'], 'share classes')
    screen = pd.DataFrame(rows)
    screen.to_csv(screen_file, index=False)
    with open('data/raw/source_log.txt', 'a') as log:
        log.write('Yahoo fund screener run on ' + str(date.today()) + '\n')

screen = pd.read_csv(screen_file)
screen = screen.drop_duplicates(subset='ticker')
screen['fund_name'] = screen['long_name'].fillna(screen['short_name']).str.strip()
screen['first_trade'] = pd.to_datetime(screen['first_trade_ms'], unit='ms')   # a date we can sort
print('Share classes on Yahoo:', len(screen))


# ======================= Step 2: one share class per fund (R2) =======================
# "AB Growth A" and "AB Growth C" are the same fund. Remove the class words at the end of the
# name to get the fund's name, then keep the oldest share class of each fund.
CLASS_WORDS = ['A', 'B', 'C', 'D', 'E', 'F', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S',
               'T', 'Y', 'Z', 'INV', 'INVESTOR', 'INSTL', 'INST', 'INSTITUTIONAL', 'ADM', 'ADMIN',
               'ADMIRAL', 'ADV', 'ADVISOR', 'RETAIL', 'SVC', 'SERVICE', 'SELECT', 'PREMIER',
               'CLASS', 'CL', 'SHS', 'SHARES', 'FUND', 'TR', 'ORD', 'ORDINARY', 'IS', 'IA', 'INVMT',
               'AAA', 'RETIRE', 'RETIREMENT', 'INS', 'FD', 'FI', 'PLS', 'PLUS']


def is_class_word(word):
    w = word.upper().strip('()-')
    if w == '':                                        # a lone "-" or "()" at the end of the name
        return True
    if w in CLASS_WORDS or w.isdigit():
        return True
    if len(w) == 1 and w.isalpha():                    # any single-letter class: A, C, G, H ...
        return True
    # codes with a digit in them, like R6, I-2, R-5E, 529-F-2
    if w.replace('-', '').isalnum() and any(ch.isdigit() for ch in w):
        return True
    return False


def fund_stem(name):
    words = str(name).split()
    while len(words) > 1 and is_class_word(words[-1]):
        words = words[:-1]                             # drop the last word
    return ' '.join(words)


screen['stem'] = screen['fund_name'].apply(fund_stem)
screen = screen.sort_values(['stem', 'first_trade'])
screen['dropped_because'] = ''

duplicate_class = screen.duplicated(subset='stem', keep='first')    # every class after the oldest
screen.loc[duplicate_class, 'dropped_because'] = 'other share class of same fund (kept oldest)'


# ======================= Step 3: drop index funds and ETFs (R3) =======================
# Yahoo also shortens Index to "Indx", or glues "Idx" to the next word ("IdxInstl")
is_index = screen['fund_name'].str.contains(r'\bindex\b|\bindx\b|idx', case=False, regex=True)
screen.loc[is_index & (screen['dropped_because'] == ''), 'dropped_because'] = 'index fund (name)'

is_etf = screen['fund_name'].str.contains(r'\bETF\b', regex=True)    # the plan says no ETFs
screen.loc[is_etf & (screen['dropped_because'] == ''), 'dropped_because'] = 'ETF (name)'

# R5: funds dropped by hand after looking at the R4 list (passive, or no real fee at fund level)
if os.path.exists('inputs/exclude_manual.csv'):
    manual = pd.read_csv('inputs/exclude_manual.csv')
    is_manual = screen['ticker'].isin(manual['ticker'])
    screen.loc[is_manual & (screen['dropped_because'] == ''), 'dropped_because'] = 'manual exclude (exclude_manual.csv)'
    print('R5 manual exclude list:', is_manual.sum(), 'funds')

kept = screen[screen['dropped_because'] == '']
print('Distinct funds after R2:', (~duplicate_class).sum())
print('Active funds kept after R3 and R5:', len(kept))
print(kept['category'].value_counts())

if len(kept) < MIN_FUNDS:
    print('WARNING: fewer than', MIN_FUNDS, 'funds. Add more US equity categories to CATEGORIES,')
    print('         delete data/raw/screen_raw.csv and run again.')

tickers = kept[['ticker', 'fund_name', 'category', 'first_trade', 'stem']].sort_values('ticker')
tickers['first_trade'] = tickers['first_trade'].dt.strftime('%Y-%m-%d')
screen['first_trade'] = screen['first_trade'].dt.strftime('%Y-%m-%d')
tickers.to_csv('inputs/tickers.csv', index=False)
log_cols = ['ticker', 'fund_name', 'category', 'first_trade', 'stem', 'dropped_because']
screen[log_cols].to_csv('inputs/ticker_selection_log.csv', index=False)
print('Saved inputs/tickers.csv and inputs/ticker_selection_log.csv')


# ======================= Step 4: Yahoo expense ratios (cached, resumable) =======================
# The course PDF warns Yahoo's expense ratio is often missing, so it is only a starting point:
# Step 5 picks a random sample to check against the prospectus, and the report states the error rate.
er_file = 'data/raw/yahoo_expense_ratios.csv'
if os.path.exists(er_file):
    er = pd.read_csv(er_file)
else:
    er = pd.DataFrame({'ticker': [], 'exp_ratio_yahoo': []})

todo = [t for t in tickers['ticker'] if t not in list(er['ticker'])]
print('Expense ratios to fetch from Yahoo:', len(todo))

new_rows = []
failed = 0
for i, t in enumerate(todo):
    value = None
    try:
        ops = yf.Ticker(t).funds_data.fund_operations
        if 'Annual Report Expense Ratio' in ops.index:
            value = ops.loc['Annual Report Expense Ratio', t]
        new_rows.append({'ticker': t, 'exp_ratio_yahoo': value})   # Yahoo answered (value may be empty)
    except Exception:
        failed = failed + 1                            # Yahoo error or rate limit: not saved, so the
                                                       # next run tries this fund again
    time.sleep(0.5)
    if (i + 1) % 50 == 0 or i == len(todo) - 1:        # save progress every 50 funds
        er = pd.concat([er, pd.DataFrame(new_rows)], ignore_index=True)
        er.to_csv(er_file, index=False)
        new_rows = []
        print('  saved', len(er), 'expense ratios,', failed, 'Yahoo errors so far')

if failed > 0:
    print('Yahoo did not answer for', failed, 'funds - run this script again later to fill them in')

er = er[er['ticker'].isin(tickers['ticker'])]
print('Funds with a Yahoo expense ratio:', er['exp_ratio_yahoo'].notna().sum(), 'of', len(tickers))

# R4: very cheap "active" funds may really be index funds - look at these by hand
cheap = er[er['exp_ratio_yahoo'] < 0.0015].merge(tickers[['ticker', 'fund_name']], on='ticker')
print('Funds with expense ratio under 0.15% (check if passive):', len(cheap))
print(cheap)

# inputs/expense_ratios.csv in the format 01_build_data.py reads. Never overwrite hand edits.
if os.path.exists('inputs/expense_ratios.csv'):
    print('inputs/expense_ratios.csv already exists - not overwritten')
else:
    out = tickers[['ticker']].merge(er, on='ticker', how='left')
    out = out.rename(columns={'exp_ratio_yahoo': 'exp_ratio'})
    out['named_bench'] = None
    out['manager'] = None
    out['source_url'] = 'yahoo funds_data'
    out['date_checked'] = str(date.today())
    out.to_csv('inputs/expense_ratios.csv', index=False)
    print('Saved inputs/expense_ratios.csv')


# ======================= Step 5: random sample for the prospectus check =======================
if not os.path.exists('inputs/expense_ratio_spotcheck.csv'):
    random.seed(3630)                                  # same sample every time
    sample = random.sample(list(tickers['ticker']), 30)
    check = er[er['ticker'].isin(sample)].copy()
    check['exp_ratio_prospectus'] = None               # fill in by hand
    check['source_url'] = None
    check.to_csv('inputs/expense_ratio_spotcheck.csv', index=False)
    print('Saved inputs/expense_ratio_spotcheck.csv: fill in the prospectus column by hand')
