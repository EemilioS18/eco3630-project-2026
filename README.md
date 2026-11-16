# eco3630-project-2026
Project for eco3630.

## Quant scripts (Member 3)

Methods 1-5 of the project plan. They read `data/clean/fund_month.parquet` and `data/clean/fund_day.parquet`
(columns in [DATA_CONTRACT.md](DATA_CONTRACT.md)) and write CSV tables to `output/tables/`.

```
python src/03_regressions.py     # Methods 1-4: excess returns, alpha/beta by year, boom vs bust, tech factor
python src/04_persistence.py     # Method 5: winner/loser, quintiles, luck check, cross-era
```

Until the real data exists, test on fake data with known alphas:

```
python tests/make_fake_data.py
python src/03_regressions.py --data-dir data/fake --out output/fake_tables
python src/04_persistence.py --data-dir data/fake --out output/fake_tables
python tests/check_fake_recovery.py
```

Run from the repo root with `python src/...`. Week-1 settings (last year, minimum days, benchmark) live in `src/quant_common.py`.
