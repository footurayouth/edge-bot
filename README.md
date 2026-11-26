# Edge Stack - IBKR paper bot (runs on GitHub)

Every US trading day GitHub starts IB Gateway, logs in to the IBKR **paper** account (secrets `IB_USER`, `IB_PASSWORD`),
and trades the conservative Edge Stack with micro futures in the 16:20-16:55 New York window (23:20-23:55 Bulgaria).
No Mac needed. Do not log in to the paper account elsewhere at that time (IBKR allows one login at a time).

- **Results:** [logs/performance.csv](logs/performance.csv) (one row per day: bought / sold / held, equity, day P&L,
  total vs S&P 500), [logs/journal.csv](logs/journal.csv) (every fill with realized P&L and commission),
  and the **Actions** tab (each run's summary).
- **Test without orders:** Actions > Edge Stack paper bot > Run workflow > mode `test`.
- **Pause:** Actions > Edge Stack paper bot > `...` > Disable workflow.

- **QQQ day trade (paper ledger, no orders):** [qqq_daytrade/README.md](qqq_daytrade/README.md) compares it with this bot and the S&P 500 from the same start.
