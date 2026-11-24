# Data dictionary and source log (Member 1)

How the two data files were built, what every column means, which rows each rule removed,
and what the data cannot tell us. All counts are from the build on 2026-11-24.

Rebuild everything from the repo's top folder:

```
python src/00_build_ticker_list.py   # Yahoo screener -> inputs/tickers.csv (+ expense ratios)
python src/01_build_data.py          # prices + French factors -> data/clean/*.parquet
```

Both scripts download once and reuse the saved file afterwards (`data/raw/screen_raw.csv`,
`data/raw/prices.parquet`). Delete a file only to re-download on purpose, and log the date below.

## Source log

| Source | Date | What | Saved to |
|---|---|---|---|
| Yahoo fund screener (`yf.screen`) | 2026-11-24 | 9 Morningstar US equity style categories, Nasdaq-listed: 5,512 share classes | `data/raw/screen_raw.csv` |
| Yahoo prices (`yf.download`, adjusted close) | 2026-11-24 | 1,664 funds + SPY, daily from 1994-12-01, in batches of 50 | `data/raw/prices.parquet` |
| Yahoo prices, retry | 2026-11-24 | the 45 tickers with no prices: 0 recovered | — |
| Yahoo expense ratios (`funds_data`) | 2026-11-24 | 1,603 of the 1,642 funds | `data/raw/yahoo_expense_ratios.csv`, `inputs/expense_ratios.csv` |
| Ken French Data Library | read on every run of 01 | FF3 daily and monthly, 12 industry portfolios daily and monthly (value-weighted, first block); last month 2026-09 | not saved |

The `data/` folder is git-ignored. The final `data/clean/` files are shared on the group drive.

## Fund selection (`src/00_build_ticker_list.py`)

Every share class Yahoo listed is in `inputs/ticker_selection_log.csv` with the reason it was kept or dropped.

| Rule | What it does | Share classes |
|---|---|---|
| R1 | Universe: Yahoo screener, Large/Mid-Cap/Small × Growth/Blend/Value, Nasdaq-listed | 5,512 listed |
| R2 | One share class per fund: class words removed from the end of the name (A, C, I, R6, Inv, Instl, 529-F-2, …), oldest class kept | 3,630 dropped |
| R3 | Passive funds and ETFs: "Index", "Indx", "Idx" or "ETF" in the name | 208 index + 10 ETF dropped |
| R4 | Funds with a Yahoo expense ratio under 0.15% printed for a manual look | (look only) |
| R5 | Manual exclude list after that look, `inputs/exclude_manual.csv`: 8 funds tracking the S&P 500 or a 500-stock index, 14 Fidelity Series funds (sold only inside Fidelity funds-of-funds, 0% fee at fund level) | 22 dropped |
| | **Active funds on the list** (`inputs/tickers.csv`) | **1,642** |

## Cleaning (`src/01_build_data.py`)

| Step | What it does | Removed |
|---|---|---|
| Download failures | Tickers Yahoo returned no prices for (listed in the 01 output), retried once | 45 funds |
| C1 | Days with no price are dropped, never filled | 2,244,907 empty fund-days (mostly days before a fund existed) |
| C2 | A daily return beyond ±20% is a Yahoo price error (the largest real market day in the sample is 9.65%): that fund-day and the fund's whole month are dropped. Days in `data/raw/bad_price_days.csv` | 144 bad days in 71 funds → 129 fund-months, 2,711 fund-days |
| C3 | Only dates that have French factors are kept (French data ends 2026-09) | 3,184 fund-months |
| Flag only | Fund-months more than 10 points away from SPY, for a look by hand. Many are real (Feb 2000 tech rally) | 2,029 flagged in `data/raw/months_to_check.csv`, none removed |

Daily returns use `pct_change(fill_method=None)` so pandas 2 (Colab) and pandas 3 give the same numbers.
Only the two eras are kept: 1995–2002 and 2022 onward. `quant_common.py` drops the partial year 2026.

## Fund counts

| | Dot-com (1995–2002) | AI (2022–2026-09) |
|---|---|---|
| Funds with data | 804 | 1,595 |
| Per year | 394 (1995) → 804 (2002) | 1,506 (2022) → 1,595 (2026) |

All 804 dot-com funds are also in the AI era (see Limitations, survivorship).

## `data/clean/fund_month.parquet` (144,114 rows, one per fund per month)

| Column | Type | Meaning | Source |
|---|---|---|---|
| `ticker` | str | Fund ticker (one share class per fund) | `inputs/tickers.csv` |
| `month` | str `YYYY-MM` | Calendar month | — |
| `ret_net` | float | Monthly total return after expenses, decimal, from month-end adjusted close | Yahoo |
| `bench_ret` | float | SPY monthly total return, decimal | Yahoo |
| `rf` | float | 1-month T-bill return, decimal | French FF3 monthly |
| `mkt_rf`, `smb`, `hml` | float | Market excess return, size and value factors, decimal | French FF3 monthly |
| `tech_ret` | float | Business Equipment industry return (raw, not minus rf), decimal | French 12 industries, value-weighted |
| `exp_ratio` | float | Annual expense ratio, decimal (0.0075 = 0.75%). **From Yahoo, see Limitations** | `inputs/expense_ratios.csv` |
| `named_bench` | empty | Benchmark the fund names. Not filled yet | prospectus (to do) |
| `first_month` | str `YYYY-MM` | First month with a price in our download. The download starts 1994-12, so older funds show `1994-12`, not their launch date | Yahoo |
| `manager` | empty | Current manager and start year, e.g. `Jane Smith (since 1998)`. Not filled yet | prospectus (to do) |

## `data/clean/fund_day.parquet` (3,018,529 rows, one per fund per trading day)

| Column | Type | Meaning | Source |
|---|---|---|---|
| `ticker` | str | Fund ticker | `inputs/tickers.csv` |
| `date` | date | Trading day | — |
| `ret` | float | Daily total return from adjusted close, decimal | Yahoo |
| `rf`, `mkt_rf`, `smb`, `hml` | float | French **daily** factors, decimal | French FF3 daily |
| `tech_ret` | float | French **daily** Business Equipment return (raw), decimal | French 12 industries daily |

## Hand checks (to do)

| File | What to fill in |
|---|---|
| `inputs/expense_ratio_spotcheck.csv` | 30 random funds: prospectus expense ratio and its URL. Report how often Yahoo is wrong |
| `inputs/return_spotcheck.csv` | 5 funds × 5 years (1998, 2000, 2002, 2022, 2024): the issuer's calendar-year return next to ours |

## Limitations (for the report)

- **Survivorship bias.** The screener lists only funds that still exist, so every fund that closed or merged is missing.
  That is why all 804 dot-com funds are still alive in the AI era. Results overstate active performance, most of all
  for 1995–2002. The 45 tickers with no prices are download failures, not a count of dead funds; that count
  needs a dated fund list from the time, or S&P SPIVA survival rates as a fallback.
- **Share classes are matched by name.** Where Yahoo shortens a name differently for two classes of one fund
  ("Gro" vs "Growth"), the fund can still be counted twice.
- **Expense ratios are Yahoo's.** The plan says to use the prospectus. With 1,642 funds that was not feasible, so
  the 30-fund spot check measures Yahoo's error rate instead. 39 funds have no expense ratio.
- **Yahoo adjusted prices.** Rule C2 removes the obvious distribution errors; smaller ones can remain. The 5-fund
  return spot check tests this, and `months_to_check.csv` lists months worth a look.
- **Empty `manager` and `named_bench`.** Sophia's cross-era comparison reads `manager`, so until it is filled in for
  the funds in both eras every fund shows as "not the same manager".
- **Benchmark.** `bench_ret` is SPY for every fund, including mid- and small-cap funds. The FF3 regressions control
  for size and value; Method 1's excess return does not.
