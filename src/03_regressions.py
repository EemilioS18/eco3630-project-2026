"""Methods 1-4 of the project plan. Owner: Member 3 (Quant lead).

  python src/03_regressions.py                      # real data in data/clean -> output/tables
  python src/03_regressions.py --data-dir data/fake --out output/fake_tables   # fake test data

Writes to --out:
  m1_fund_year_excess.csv     fund x year: net return, benchmark return, excess return (Method 1)
  m1_share_beating.csv        year: share of funds beating the benchmark, mean/median excess
  m1_era_test.csv             H1 on excess returns: is the mean fund-year excess return < 0 in each era?
  m2_alpha_fund_year.csv      fund x year: alpha/beta/R2 for CAPM, FF3 and FF3+tech (Methods 2 and 4)
  m2_mean_alpha_by_year.csv   year: mean alpha per model, t-test against zero
  m2_era_test.csv             H1 on alpha: mean fund-year alpha per era, SEs clustered by year
  m2_flagged_funds.csv        top 5 and bottom 5 funds by FF3 alpha in each era (for Member 4)
  m3_boom_bust.csv            alpha = c + gamma * Bust, SEs clustered by year (Method 3, H2)
  m4_tech_by_year.csv         year: mean alpha with and without the tech factor, mean tech loading
  m4_h4_test.csv              H4: how much positive boom-year alpha survives the tech factor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from quant_common import (BENCHMARK, HAC_LAGS, LAST_FULL_YEAR, MIN_DAYS_PER_YEAR,
                          MIN_MONTHS_PER_YEAR, TRADING_DAYS, load_day, load_month, parse_args,
                          phase_of, save)

MODELS = {
    "capm": ["mkt_rf"],
    "ff3": ["mkt_rf", "smb", "hml"],
    "ff3tech": ["mkt_rf", "smb", "hml", "tech_rf"],
}


# ---------------------------------------------------------------- Method 1
def method1(month: pd.DataFrame, out) -> pd.DataFrame:
    """Compound monthly net returns and benchmark returns to calendar years."""
    m = month[month["year"] <= LAST_FULL_YEAR]
    grow = lambda s: (1 + s).prod() - 1
    fy = (m.groupby(["ticker", "year", "era"])
            .agg(ret_net=("ret_net", grow), bench=(BENCHMARK, grow), n_months=("ret_net", "count"),
                 exp_ratio=("exp_ratio", "last"), manager=("manager", "last"))
            .reset_index())
    fy = fy[fy["n_months"] >= MIN_MONTHS_PER_YEAR].copy()
    fy["excess"] = fy["ret_net"] - fy["bench"]
    fy["beat"] = fy["excess"] > 0
    fy["phase"] = fy["year"].map(phase_of)
    save(fy, out, "m1_fund_year_excess")

    rows = []
    for (year, era), g in fy.groupby(["year", "era"]):
        n, k = len(g), int(g["beat"].sum())
        rows.append({"year": year, "era": era, "phase": phase_of(year), "n_funds": n,
                     "bench_return": g["bench"].iloc[0], "share_beating": k / n,
                     "binom_p_vs_half": stats.binomtest(k, n, 0.5).pvalue,
                     "mean_excess": g["excess"].mean(), "median_excess": g["excess"].median()})
    save(pd.DataFrame(rows), out, "m1_share_beating")
    save(era_mean_test(fy, "excess"), out, "m1_era_test")
    return fy


def era_mean_test(df: pd.DataFrame, col: str, label: str = "") -> pd.DataFrame:
    """Mean of `col` per era, SEs clustered by year (every fund faces the same market in a year).

    With only 4-8 years per era the clustered SE is rough; the per-year table matters as much.
    """
    rows = []
    for era, g in df.dropna(subset=[col]).groupby("era"):
        fit = sm.OLS(g[col].to_numpy(), np.ones(len(g))).fit(
            cov_type="cluster", cov_kwds={"groups": g["year"].to_numpy()}, use_t=True)
        rows.append({"era": era, "model": label or col, "mean": fit.params[0], "se_cluster_year": fit.bse[0],
                     "t": fit.tvalues[0], "p": fit.pvalues[0], "n_fund_years": len(g),
                     "n_years": g["year"].nunique(), "n_funds": g["ticker"].nunique()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Methods 2 and 4
def fit_fund_year(g: pd.DataFrame) -> dict:
    row = {"n_days": len(g)}
    y = g["ex_ret"].to_numpy()
    for name, cols in MODELS.items():
        X = sm.add_constant(g[cols].to_numpy())
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        row[f"alpha_{name}"] = fit.params[0] * TRADING_DAYS
        row[f"alpha_t_{name}"] = fit.tvalues[0]
        row[f"alpha_p_{name}"] = fit.pvalues[0]
        row[f"r2_{name}"] = fit.rsquared
        for col, b, t in zip(cols, fit.params[1:], fit.tvalues[1:]):
            short = {"mkt_rf": "beta", "smb": "s", "hml": "h", "tech_rf": "tech"}[col]
            row[f"{short}_{name}"] = b
            if col == "tech_rf":
                row[f"tech_t_{name}"] = t
    return row


def method2(day: pd.DataFrame, month: pd.DataFrame, out) -> pd.DataFrame:
    d = day[day["year"] <= LAST_FULL_YEAR]
    rows = []
    for (ticker, year, era), g in d.groupby(["ticker", "year", "era"]):
        if len(g) < MIN_DAYS_PER_YEAR:
            continue
        rows.append({"ticker": ticker, "year": year, "era": era, "phase": phase_of(year), **fit_fund_year(g)})
    fa = pd.DataFrame(rows)
    info = month.groupby("ticker")[["manager", "exp_ratio"]].last().reset_index()
    fa = fa.merge(info, on="ticker", how="left")
    # Method 4 flag: FF3 alpha was significantly positive, disappears once tech is controlled for,
    # and the fund loads clearly on tech. "Beat the market by owning tech, not by picking stocks."
    fa["alpha_drop_tech"] = fa["alpha_ff3"] - fa["alpha_ff3tech"]
    fa["tech_driven"] = ((fa["alpha_ff3"] > 0) & (fa["alpha_p_ff3"] < 0.05)
                         & (fa["alpha_p_ff3tech"] >= 0.05) & (fa["tech_t_ff3tech"] > 2))
    save(fa, out, "m2_alpha_fund_year")

    rows = []
    for (year, era), g in fa.groupby(["year", "era"]):
        row = {"year": year, "era": era, "phase": phase_of(year), "n_funds": len(g)}
        for name in MODELS:
            a = g[f"alpha_{name}"]
            row[f"mean_alpha_{name}"] = a.mean()
            row[f"t_{name}"] = stats.ttest_1samp(a, 0).statistic if len(a) > 1 else np.nan
            row[f"share_sig_pos_{name}"] = ((a > 0) & (g[f"alpha_p_{name}"] < 0.05)).mean()
        rows.append(row)
    save(pd.DataFrame(rows), out, "m2_mean_alpha_by_year")
    save(pd.concat([era_mean_test(fa, f"alpha_{m}", m) for m in MODELS]), out, "m2_era_test")

    flagged = []
    for era, g in fa.groupby("era"):
        per_fund = (g.groupby("ticker")
                      .agg(mean_alpha_ff3=("alpha_ff3", "mean"), mean_alpha_ff3tech=("alpha_ff3tech", "mean"),
                           mean_tech_loading=("tech_ff3tech", "mean"), years=("year", "count"),
                           manager=("manager", "last"), exp_ratio=("exp_ratio", "last"))
                      .reset_index())
        # Only rank funds present for at least half the era, so one lucky year can't top the list.
        per_fund = per_fund[per_fund["years"] >= g["year"].nunique() / 2]
        top = per_fund.nlargest(5, "mean_alpha_ff3").assign(group="top5")
        bottom = per_fund.nsmallest(5, "mean_alpha_ff3").assign(group="bottom5")
        flagged.append(pd.concat([top, bottom]).assign(era=era))
    save(pd.concat(flagged, ignore_index=True), out, "m2_flagged_funds")
    return fa


# ---------------------------------------------------------------- Method 3
def method3(fa: pd.DataFrame, out) -> None:
    """alpha_{i,y} = c + gamma * Bust_y + u, SEs clustered by year. 2000 (turning year) excluded."""
    d = fa[fa["phase"].isin(["boom", "bust"])].copy()
    d["bust"] = (d["phase"] == "bust").astype(float)
    rows = []
    for sample, g in [("both_eras", d)] + list(d.groupby("era")):
        for name in MODELS:
            if g["bust"].nunique() < 2:
                continue
            X = sm.add_constant(g["bust"].to_numpy())
            fit = sm.OLS(g[f"alpha_{name}"].to_numpy(), X).fit(
                cov_type="cluster", cov_kwds={"groups": g["year"].to_numpy()}, use_t=True)
            rows.append({"sample": sample, "model": name,
                         "c_boom_mean_alpha": fit.params[0], "gamma_bust_minus_boom": fit.params[1],
                         "gamma_se": fit.bse[1], "gamma_t": fit.tvalues[1], "gamma_p": fit.pvalues[1],
                         "n_fund_years": len(g), "n_years": g["year"].nunique()})
    save(pd.DataFrame(rows), out, "m3_boom_bust")


# ---------------------------------------------------------------- Method 4
def method4(fa: pd.DataFrame, out) -> None:
    by_year = (fa.groupby(["year", "era", "phase"])
                 .agg(n_funds=("ticker", "count"), mean_alpha_ff3=("alpha_ff3", "mean"),
                      mean_alpha_ff3tech=("alpha_ff3tech", "mean"), mean_tech_loading=("tech_ff3tech", "mean"),
                      share_tech_driven=("tech_driven", "mean"))
                 .reset_index())
    save(by_year, out, "m4_tech_by_year")

    # H4: among boom fund-years with positive FF3 alpha, how much alpha is left after adding tech?
    rows = []
    boom = fa[(fa["phase"] == "boom") & (fa["alpha_ff3"] > 0)]
    for era, g in [("both_eras", boom)] + list(boom.groupby("era")):
        if len(g) < 2:
            continue
        before, after = g["alpha_ff3"].mean(), g["alpha_ff3tech"].mean()
        rows.append({"sample": era, "n_fund_years": len(g),
                     "mean_alpha_ff3": before, "mean_alpha_ff3tech": after,
                     "share_of_alpha_explained_by_tech": 1 - after / before,
                     "paired_t": stats.ttest_rel(g["alpha_ff3"], g["alpha_ff3tech"]).statistic,
                     "paired_p": stats.ttest_rel(g["alpha_ff3"], g["alpha_ff3tech"]).pvalue,
                     "share_tech_driven": g["tech_driven"].mean()})
    save(pd.DataFrame(rows), out, "m4_h4_test")


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    month = load_month(args.data_dir)
    day = load_day(args.data_dir)
    print(f"Loaded {month['ticker'].nunique()} funds, {len(month)} fund-months, {len(day)} fund-days")
    print("Method 1: excess returns")
    method1(month, args.out)
    print("Methods 2 and 4: yearly regressions")
    fa = method2(day, month, args.out)
    print("Method 3: boom vs bust")
    method3(fa, args.out)
    print("Method 4: tech factor summary")
    method4(fa, args.out)


if __name__ == "__main__":
    main()
