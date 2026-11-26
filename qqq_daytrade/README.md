# QQQ day trade — paper ledger (started 2026-11-27)

Tracks the "noise-band breakout" day trade on QQQ next to the Edge Stack bot and the S&P 500, all from the same
starting dollars (the Edge bot's equity on 2026-11-25, $109,812). No orders are sent and nothing logs in to IBKR:
after each close GitHub downloads the day's QQQ 5-minute bars from Yahoo and replays the rule (`ledger.py`).

**The rule, in short:** every half hour from 10:00 to 15:30, if QQQ is further from today's open than it
normally is at that time of day (the 14-day average move = "noise band"), follow the move: long above the band, short
below it. Exit when price falls back inside the band or crosses VWAP. Always flat at the close. 3x = the same
trade with 3x the size (e.g. MNQ micro futures); cost 1 bp of price per round trip.

Backtest (Alpaca 5m bars, 1 bp): 2016-26 ~10%/yr at 1x (worst drop -10%), ~31%/yr at 3x (-29%);
2022-26 ~12% / ~38% (-23%). At 3 bp costs 3x falls to ~22%/yr. Unproven: the rule failed on IWM and DIA.
Data check: Yahoo vs Alpaca bars on the same 42 days give the same trades on most days (daily-return corr 0.99),
but single days can differ by up to ~0.35% because the 9:30 opening print differs between the two feeds.

Files: `days.csv` (each day's trades and 1x return), `performance.csv` (equity and % vs Edge and S&P 500).

<!-- table -->
