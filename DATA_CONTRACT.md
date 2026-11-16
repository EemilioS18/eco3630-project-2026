# Data contract: what the Quant scripts expect from `01_build_data.py`

Member 1 builds these files; Members 3 and 4 only read them. If a column has to change, open an issue first.
`src/quant_common.py` stops with a clear message if a column is missing or looks like percents.

**All returns are decimals** (0.012 = 1.2%). French files are published in percents: divide by 100.

## `data/clean/fund_month.parquet` (from the plan, page 3)

One row per fund per month.

| Column | Type | Meaning |
|---|---|---|
| `ticker` | str | Fund ticker |
| `month` | str `YYYY-MM` | Month end |
| `ret_net` | float | Monthly total return after expenses, from adjusted close |
| `bench_ret` | float | SPY monthly total return |
| `rf` | float | 1-month T-bill return (French) |
| `mkt_rf`, `smb`, `hml` | float | French monthly factors |
| `tech_ret` | float | French 12-industry "Business Equipment" monthly return (raw, not minus rf) |
| `exp_ratio` | float | Annual expense ratio, decimal (prospectus) |
| `named_bench` | str | Benchmark the fund names (prospectus) |
| `first_month` | str `YYYY-MM` | First month with a price |
| `manager` | str | Current manager **and start year**, e.g. `"Jane Smith (since 1998)"`. The first 4-digit year is read as the start year |

`era` and `phase` are optional: the Quant scripts derive them from the year using the plan's Periods table.

## `data/clean/fund_day.parquet` (new: needed for Methods 2–4)

The plan runs Methods 2–4 on **daily** returns, but `fund_month` is monthly, so a second file is needed.
One row per fund per trading day.

| Column | Type | Meaning |
|---|---|---|
| `ticker` | str | Fund ticker |
| `date` | date | Trading day |
| `ret` | float | Daily total return from adjusted close |
| `rf` | float | French daily risk-free rate |
| `mkt_rf`, `smb`, `hml` | float | French **daily** three-factor file |
| `tech_ret` | float | French 12-industry **daily** Business Equipment return (raw) |

Drop days where a fund has no price; don't fill them with 0.

## Settings decided in week 1 (all in `src/quant_common.py`)

| Setting | Current value | Note |
|---|---|---|
| `LAST_FULL_YEAR` | 2025 | 2026 is partial |
| `MIN_DAYS_PER_YEAR` | 200 | Method 2 minimum |
| `BENCHMARK` | `bench_ret` (SPY) | Set to `mkt` to use the French market instead. The plan contradicts itself here: pick one |
| `HAC_LAGS` | 5 | Newey-West lags |
