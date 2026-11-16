"""Shared settings and loaders for the Quant lead scripts (03 and 04).

Owner: Member 3 (Quant lead). Other members: change these through an issue, not a commit.

Input files (built by Member 1's 01_build_data.py, never edited by hand):
  data/clean/fund_month.parquet  one row per fund per month (columns: see DATA_CONTRACT.md)
  data/clean/fund_day.parquet    one row per fund per trading day (needed for Methods 2-4)

All returns are decimals (0.01 = 1%), never percents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---- Week-1 decisions. Change here, and only here, once the group agrees. ----
LAST_FULL_YEAR = 2025          # last calendar year used; 2026 is partial
MIN_DAYS_PER_YEAR = 200        # Method 2: a fund-year needs this many daily returns
MIN_MONTHS_PER_YEAR = 12       # Method 1: a fund-year needs all 12 monthly returns
BENCHMARK = "bench_ret"        # Method 1 benchmark column: "bench_ret" (SPY) or "mkt" (French market)
HAC_LAGS = 5                   # Newey-West lags for daily regressions
TRADING_DAYS = 252             # annualize daily alpha by this

ERAS = {"dotcom": (1995, 2002), "ai": (2022, LAST_FULL_YEAR)}

# Method 3 year labels (from the plan). 2000 is the turning year and is reported on its own.
BOOM_YEARS = set(range(1995, 2000)) | set(range(2023, LAST_FULL_YEAR + 1))
BUST_YEARS = {2001, 2002, 2022}
TURNING_YEARS = {2000}

MONTH_COLUMNS = ["ticker", "month", "ret_net", "bench_ret", "rf", "mkt_rf", "smb", "hml",
                 "tech_ret", "exp_ratio", "named_bench", "first_month", "manager"]
DAY_COLUMNS = ["ticker", "date", "ret", "rf", "mkt_rf", "smb", "hml", "tech_ret"]


def era_of(year: int) -> str | None:
    for name, (start, end) in ERAS.items():
        if start <= year <= end:
            return name
    return None


def phase_of(year: int) -> str | None:
    if year in BOOM_YEARS:
        return "boom"
    if year in BUST_YEARS:
        return "bust"
    if year in TURNING_YEARS:
        return "turning"
    return None


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", default="data/clean", type=Path,
                        help="folder holding fund_month.parquet and fund_day.parquet")
    parser.add_argument("--out", default="output/tables", type=Path,
                        help="folder for output tables")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def _check_columns(df: pd.DataFrame, needed: list[str], name: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} is missing columns {missing}. See DATA_CONTRACT.md.")


def _check_units(df: pd.DataFrame, cols: list[str], name: str) -> None:
    # French publishes percents. A daily or monthly decimal return above 1.0 (100%) means
    # someone forgot to divide by 100.
    for col in cols:
        if df[col].abs().max() > 1.0:
            raise SystemExit(f"{name}.{col} has values above 1.0: looks like percents, not decimals.")


def load_month(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / "fund_month.parquet")
    _check_columns(df, MONTH_COLUMNS, "fund_month")
    _check_units(df, ["ret_net", "bench_ret", "rf", "mkt_rf", "smb", "hml", "tech_ret"], "fund_month")
    df["month"] = pd.PeriodIndex(df["month"].astype(str), freq="M")
    df["year"] = df["month"].dt.year
    df["mkt"] = df["mkt_rf"] + df["rf"]
    df["era"] = df["year"].map(era_of)
    return df[df["era"].notna()].copy()


def load_day(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / "fund_day.parquet")
    _check_columns(df, DAY_COLUMNS, "fund_day")
    _check_units(df, ["ret", "rf", "mkt_rf", "smb", "hml", "tech_ret"], "fund_day")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["era"] = df["year"].map(era_of)
    df = df[df["era"].notna()].copy()
    df["ex_ret"] = df["ret"] - df["rf"]
    df["tech_rf"] = df["tech_ret"] - df["rf"]
    return df


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
    print(f"  wrote {out / (name + '.csv')} ({len(df)} rows)")
