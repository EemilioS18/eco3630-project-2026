"""Make a fake fund_month.parquet and fund_day.parquet with KNOWN alphas and betas.

Lets the Quant scripts be written and checked before Member 1's real data exists.
If 03_regressions.py recovers the alphas and betas saved in fake_truth.csv, the code works.

  python tests/make_fake_data.py              -> writes data/fake/
  python src/03_regressions.py --data-dir data/fake --out output/fake_tables
  python tests/check_fake_recovery.py

Built in, so each test has something to find:
  - true annual alphas from -3% to +3% (the Method 2 test)
  - "tech-tilted" funds whose FF3 alpha in boom years is really tech exposure (the Method 4 / H4 test)
  - a few funds with persistent skill (the Method 5 test)
  - funds that start late and funds that exist in only one era (realistic gaps)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 3630
N_FUNDS = 40
YEARS = list(range(1995, 2003)) + list(range(2022, 2027))  # 2026 is partial, like the real data
OUT = Path("data/fake")


def trading_days(year: int) -> pd.DatetimeIndex:
    end = f"{year}-10-31" if year == 2026 else f"{year}-12-31"
    return pd.bdate_range(f"{year}-01-01", end)


def make_factors(rng: np.random.Generator) -> pd.DataFrame:
    dates = trading_days(YEARS[0]).append([trading_days(y) for y in YEARS[1:]])
    n = len(dates)
    year = dates.year
    boom = np.isin(year, [1995, 1996, 1997, 1998, 1999, 2023, 2024, 2025, 2026])
    rf = np.where(year < 2003, 0.05, 0.04) / 252
    mkt_rf = rng.normal(0.0004, 0.011, n)
    smb = rng.normal(0.0, 0.005, n)
    hml = rng.normal(0.0, 0.005, n)
    # Tech beats the market in booms and lags it in busts: the source of fake "alpha" for tilted funds.
    tech_extra = rng.normal(np.where(boom, 0.0004, -0.0006), 0.008, n)
    tech_ret = rf + 1.2 * mkt_rf + tech_extra
    return pd.DataFrame({"date": dates, "rf": rf, "mkt_rf": mkt_rf, "smb": smb, "hml": hml,
                         "tech_ret": tech_ret, "tech_extra": tech_extra})


def make_funds(rng: np.random.Generator) -> pd.DataFrame:
    funds = []
    for i in range(N_FUNDS):
        kind = "plain"
        if i < 8:
            kind = "tech_tilt"
        elif i < 12:
            kind = "skilled"
        # Which eras the fund lives in: most in both, some only one, some start late.
        if i % 7 == 3:
            span = (2022, 2026)
        elif i % 9 == 5:
            span = (1995, 2002)
        elif i % 11 == 2:
            span = (1998, 2026)
        else:
            span = (1995, 2026)
        funds.append({
            "ticker": f"FAKE{i:02d}",
            "kind": kind,
            "first_year": span[0], "last_year": span[1],
            "alpha_ann": 0.03 if kind == "skilled" else rng.uniform(-0.03, 0.01),
            "beta": rng.uniform(0.85, 1.15),
            "s": rng.uniform(-0.2, 0.4),
            "h": rng.uniform(-0.3, 0.3),
            "tech_load": rng.uniform(0.5, 0.8) if kind == "tech_tilt" else 0.0,
            "exp_ratio": rng.uniform(0.006, 0.012),
            "manager": f"Manager {chr(65 + i % 26)} (since {span[0] if i % 2 else 2010})",
        })
    return pd.DataFrame(funds)


def main() -> None:
    rng = np.random.default_rng(SEED)
    fac = make_factors(rng)
    funds = make_funds(rng)
    OUT.mkdir(parents=True, exist_ok=True)

    days = []
    for f in funds.itertuples():
        d = fac[(fac.date.dt.year >= f.first_year) & (fac.date.dt.year <= f.last_year)].copy()
        # Tech tilt loads on tech's return beyond the market (orthogonal part), so the true FF3
        # beta stays near f.beta and only the alpha is contaminated.
        d["ret"] = (d.rf + f.alpha_ann / 252 + f.beta * d.mkt_rf + f.s * d.smb + f.h * d.hml
                    + f.tech_load * d.tech_extra + rng.normal(0, 0.004, len(d)))
        d["ticker"] = f.ticker
        days.append(d)
    day = pd.concat(days, ignore_index=True)
    day[["ticker", "date", "ret", "rf", "mkt_rf", "smb", "hml", "tech_ret"]].to_parquet(OUT / "fund_day.parquet")

    # Monthly: compound daily returns. (Real French monthly factors are published, not compounded,
    # but this is close enough for a code test.)
    day["month"] = day.date.dt.to_period("M")
    comp = lambda s: (1 + s).prod() - 1
    m = day.groupby(["ticker", "month"]).agg(ret_net=("ret", comp)).reset_index()
    mk = fac.assign(month=fac.date.dt.to_period("M"), mkt=fac.mkt_rf + fac.rf).groupby("month").agg(
        mkt=("mkt", comp), rf=("rf", comp), smb=("smb", comp), hml=("hml", comp),
        tech_ret=("tech_ret", comp)).reset_index()
    mk["mkt_rf"] = mk.mkt - mk.rf
    # SPY: the market plus a little tracking noise.
    mk["bench_ret"] = mk.mkt + rng.normal(0, 0.002, len(mk))
    m = m.merge(mk[["month", "rf", "mkt_rf", "smb", "hml", "tech_ret", "bench_ret"]], on="month")
    m = m.merge(funds[["ticker", "exp_ratio", "manager"]], on="ticker")
    m["named_bench"] = "S&P 500"
    m["first_month"] = m.groupby("ticker")["month"].transform("min").astype(str)
    m["month"] = m["month"].astype(str)
    m.to_parquet(OUT / "fund_month.parquet")

    funds.to_csv(OUT / "fake_truth.csv", index=False)
    print(f"wrote {OUT}/fund_day.parquet ({len(day)} rows), fund_month.parquet ({len(m)} rows), fake_truth.csv")


if __name__ == "__main__":
    main()
