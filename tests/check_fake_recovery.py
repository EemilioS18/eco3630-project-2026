"""Check that 03_regressions.py recovers the alphas and betas planted by make_fake_data.py.

  python tests/make_fake_data.py
  python src/03_regressions.py --data-dir data/fake --out output/fake_tables
  python src/04_persistence.py --data-dir data/fake --out output/fake_tables
  python tests/check_fake_recovery.py

Exits with an error if any check fails.
"""

from pathlib import Path

import pandas as pd

TABLES = Path("output/fake_tables")
truth = pd.read_csv("data/fake/fake_truth.csv")
fa = pd.read_csv(TABLES / "m2_alpha_fund_year.csv").merge(truth, on="ticker")
failures = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


# One fund-year's alpha is noisy: with 0.4% daily idiosyncratic noise its standard error is about
# 6% a year (real funds are similar). So test the way a statistician would: the true alpha should sit
# inside the regression's 95% confidence interval about 95% of the time, and errors should average ~0.
fa["alpha_se"] = fa.alpha_ff3tech / fa.alpha_t_ff3tech
z = (fa.alpha_ff3tech - fa.alpha_ann) / fa.alpha_se
coverage = (z.abs() < 1.96).mean()
check("alpha 95% CI covers the true alpha", 0.92 <= coverage <= 0.98,
      f"coverage {coverage:.3f} over {len(fa)} fund-years (HAC SEs)")
bias, bias_se = (fa.alpha_ff3tech - fa.alpha_ann).mean(), fa.alpha_se.mean() / len(fa) ** 0.5
check("alpha unbiased", abs(bias) < 3 * bias_se, f"mean error {bias:+.4f} (SE about {bias_se:.4f})")
per = fa.groupby("ticker").agg(est=("alpha_ff3tech", "mean"), true=("alpha_ann", "first"),
                               beta=("beta_ff3", "mean"), beta_true=("beta", "first"),
                               tech=("tech_ff3tech", "mean"), tech_true=("tech_load", "first"))
print(f"       per fund, averaged over its years: corr(estimated, true alpha) = {per.est.corr(per.true):.2f}")
check("FF3 market beta recovered", (per.beta - per.beta_true).abs().max() < 0.03,
      f"max abs error {(per.beta - per.beta_true).abs().max():.4f}")
check("tech loading recovered", (per.tech - per.tech_true).abs().max() < 0.05,
      f"max abs error {(per.tech - per.tech_true).abs().max():.4f}")

# H4 built in: tech-tilted funds show positive FF3 alpha in booms that the tech factor removes.
tilt_boom = fa[(fa.kind == "tech_tilt") & (fa.phase == "boom")]
check("tech-tilt funds: fake boom alpha under FF3", tilt_boom.alpha_ff3.mean() > 0.03,
      f"mean FF3 alpha {tilt_boom.alpha_ff3.mean():.3f}, true {tilt_boom.alpha_ann.mean():.3f}")
tilt_err = tilt_boom.alpha_ff3tech - tilt_boom.alpha_ann
tilt_se = (tilt_boom.alpha_ff3tech / tilt_boom.alpha_t_ff3tech).mean() / len(tilt_boom) ** 0.5
check("tech-tilt funds: fake alpha gone with tech factor", abs(tilt_err.mean()) < 3 * tilt_se,
      f"mean FF3+tech alpha {tilt_boom.alpha_ff3tech.mean():.3f} vs true {tilt_boom.alpha_ann.mean():.3f} "
      f"(SE about {tilt_se:.3f})")
plain = fa[fa.kind != "tech_tilt"]
check("untilted funds: tech factor changes little",
      (plain.alpha_ff3 - plain.alpha_ff3tech).abs().mean() < 0.01,
      f"mean abs change {(plain.alpha_ff3 - plain.alpha_ff3tech).abs().mean():.4f}")
check("tech_driven flag hits tilted funds, not others",
      fa[fa.kind == "tech_tilt"].tech_driven.mean() > 5 * max(plain.tech_driven.mean(), 0.01),
      f"tilted {fa[fa.kind == 'tech_tilt'].tech_driven.mean():.2f} vs others {plain.tech_driven.mean():.2f}")

# Method 1: excess = compounded fund return minus compounded benchmark return.
fy = pd.read_csv(TABLES / "m1_fund_year_excess.csv")
gap = (fy.excess - (fy.ret_net - fy.bench)).abs().max()
check("excess = ret_net - bench", gap < 1e-5, f"max gap {gap:.1e} (CSV keeps 6 significant digits)")
check("only full years kept", (fy.n_months == 12).all() and fy.year.max() <= 2025,
      f"months {fy.n_months.min()}-{fy.n_months.max()}, last year {fy.year.max()}")

# Method 5: no 2002 -> 2022 pairs, and skilled funds should beat the benchmark more often.
wl = pd.read_csv(TABLES / "m5_winner_loser.csv")
check("no cross-era transitions", not wl.transition.str.startswith("2002->").any(), "none found")
fy = fy.merge(truth, on="ticker")
skill_beat = fy[fy.kind == "skilled"].beat.mean()
plain_beat = fy[fy.kind == "plain"].beat.mean()
check("skilled funds beat more often", skill_beat > plain_beat,
      f"skilled {skill_beat:.2f} vs plain {plain_beat:.2f}")

print(f"\n{'All checks passed.' if not failures else f'{len(failures)} check(s) failed: {failures}'}")
raise SystemExit(1 if failures else 0)
