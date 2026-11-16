"""Method 5 of the project plan: persistence and luck. Owner: Member 3 (Quant lead).

Run 03_regressions.py first; this reads its tables from the same --out folder.

  python src/04_persistence.py
  python src/04_persistence.py --data-dir data/fake --out output/fake_tables

Writes to --out:
  m5_winner_loser.csv     2x2 winner/loser tables, year t -> t+1, per transition and pooled per era,
                          with chi-square, Fisher exact and the cross-product ratio (H3)
  m5_quintiles.csv        where year-t top and bottom quintile funds rank in year t+1
  m5_luck.csv             count of significant alphas vs the 5% expected by chance
  m5_cross_era.csv        funds alive in both eras: alpha in each, and the manager
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from quant_common import parse_args, save

ALPHA_MODEL = "ff3"   # model used for the luck check and cross-era comparison


def next_year_pairs(fy: pd.DataFrame) -> pd.DataFrame:
    """Each fund's year t joined to its year t+1, both in the same era (no 2002 -> 2022 jump)."""
    nxt = fy[["ticker", "year", "era", "excess", "beat"]].copy()
    nxt["year"] -= 1
    return fy.merge(nxt, on=["ticker", "year", "era"], suffixes=("", "_next"))


def two_by_two(g: pd.DataFrame, label: str, era: str) -> dict:
    ww = int((g["beat"] & g["beat_next"]).sum())
    wl = int((g["beat"] & ~g["beat_next"]).sum())
    lw = int((~g["beat"] & g["beat_next"]).sum())
    ll = int((~g["beat"] & ~g["beat_next"]).sum())
    table = np.array([[ww, wl], [lw, ll]])
    row = {"era": era, "transition": label, "n_pairs": len(g), "WW": ww, "WL": wl, "LW": lw, "LL": ll,
           "p_win_next_if_winner": ww / (ww + wl) if ww + wl else np.nan,
           "p_win_next_if_loser": lw / (lw + ll) if lw + ll else np.nan}
    # Cross-product ratio (Brown and Goetzmann 1995): above 1 means winners tend to repeat.
    row["cross_product_ratio"] = (ww * ll) / (wl * lw) if wl * lw else np.nan
    if (table.sum(axis=0) > 0).all() and (table.sum(axis=1) > 0).all():
        chi2, p, _, expected = stats.chi2_contingency(table, correction=False)
        row.update(chi2=chi2, chi2_p=p, min_expected_cell=expected.min(),
                   fisher_p=stats.fisher_exact(table).pvalue)
    return row


def winner_loser(pairs: pd.DataFrame, out) -> None:
    rows = []
    for era, g in pairs.groupby("era"):
        for year, gy in g.groupby("year"):
            rows.append(two_by_two(gy, f"{year}->{year + 1}", era))
        # Pooling years treats fund-years as independent, which they are not (same market each
        # year). Read the pooled p-value as optimistic; the per-year rows show whether it is consistent.
        rows.append(two_by_two(g, "pooled", era))
    save(pd.DataFrame(rows), out, "m5_winner_loser")


def quintiles(pairs: pd.DataFrame, fy: pd.DataFrame, out) -> None:
    def quintile(s: pd.Series) -> pd.Series:
        return pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)  # 5 = best

    counts = fy.groupby("year")["ticker"].transform("count")
    q = fy[counts >= 5].copy()
    q["q"] = q.groupby("year")["excess"].transform(quintile)
    qq = q[["ticker", "year", "q"]]
    p = (pairs.merge(qq, on=["ticker", "year"])
              .merge(qq.assign(year=qq["year"] - 1), on=["ticker", "year"], suffixes=("", "_next")))
    rows = []
    for era, g in p.groupby("era"):
        for start in (5, 1):
            s = g[g["q"] == start]
            row = {"era": era, "start_quintile": "top" if start == 5 else "bottom", "n": len(s),
                   "mean_next_quintile": s["q_next"].mean()}
            for k in range(1, 6):
                row[f"share_next_q{k}"] = (s["q_next"] == k).mean() if len(s) else np.nan
            rows.append(row)
    # With no persistence, each share_next_q is about 0.20 and mean_next_quintile about 3.
    save(pd.DataFrame(rows), out, "m5_quintiles")


def luck(fa: pd.DataFrame, out) -> None:
    rows = []
    groups = [((e, str(y)), g) for (e, y), g in fa.groupby(["era", "year"])]
    groups += [((e, "all_years"), g) for e, g in fa.groupby("era")]
    for (era, year), g in groups:
        n = len(g)
        sig = g[f"alpha_p_{ALPHA_MODEL}"] < 0.05
        pos = sig & (g[f"alpha_{ALPHA_MODEL}"] > 0)
        neg = sig & (g[f"alpha_{ALPHA_MODEL}"] < 0)
        rows.append({"era": era, "year": year, "model": ALPHA_MODEL, "n_tests": n,
                     "n_significant": int(sig.sum()), "expected_by_chance": 0.05 * n,
                     "n_sig_positive": int(pos.sum()), "expected_sig_positive": 0.025 * n,
                     "n_sig_negative": int(neg.sum()),
                     # Binomial p treats funds as independent. They are correlated (same market),
                     # so this p-value overstates the evidence.
                     "binom_p_sig_positive": stats.binomtest(int(pos.sum()), n, 0.025, alternative="greater").pvalue})
    save(pd.DataFrame(rows), out, "m5_luck")


def cross_era(fa: pd.DataFrame, out) -> None:
    per = fa.groupby(["ticker", "era"])[f"alpha_{ALPHA_MODEL}"].mean().unstack("era")
    if not {"dotcom", "ai"} <= set(per.columns):
        print("  no fund has alpha in both eras; skipping m5_cross_era")
        return
    both = per.dropna(subset=["dotcom", "ai"]).reset_index()
    both = both.rename(columns={"dotcom": "alpha_dotcom", "ai": "alpha_ai"})
    both = both.merge(fa.groupby("ticker")["manager"].last().reset_index(), on="ticker", how="left")

    def since(text) -> float:
        m = re.search(r"(19|20)\d{2}", str(text))
        return float(m.group(0)) if m else np.nan

    # The plan's `manager` column holds the CURRENT manager and start year. A start year of 2002 or
    # earlier means the same manager ran the fund in both eras. Check odd cases by hand.
    both["manager_start_year"] = both["manager"].map(since)
    both["same_manager_both_eras"] = both["manager_start_year"] <= 2002
    if len(both) >= 5:
        rho, p = stats.spearmanr(both["alpha_dotcom"], both["alpha_ai"])
        print(f"  cross-era: {len(both)} funds, Spearman rho = {rho:.2f} (p = {p:.3f}). "
              "Small sample: report as a case study, not a test.")
    save(both, out, "m5_cross_era")


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    try:
        fy = pd.read_csv(args.out / "m1_fund_year_excess.csv")
        fa = pd.read_csv(args.out / "m2_alpha_fund_year.csv")
    except FileNotFoundError:
        raise SystemExit(f"Run 03_regressions.py with --out {args.out} first.")
    pairs = next_year_pairs(fy)
    print(f"Method 5: {len(pairs)} fund-year pairs (t, t+1)")
    winner_loser(pairs, args.out)
    quintiles(pairs, fy, args.out)
    luck(fa, args.out)
    cross_era(fa, args.out)


if __name__ == "__main__":
    main()
