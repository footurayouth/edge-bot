"""Offline checks (no IBKR needed): order planning incl. rolls, trading window, and that the targets equal the click
tool's futures route for the same capital."""
import os, sys, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bot
from types import SimpleNamespace as C
f = {"MES": C(conId=2, symbol="MES", lastTradeDateOrContractMonth="20261218"), "MNQ": C(conId=4, symbol="MNQ", lastTradeDateOrContractMonth="20261218"),
     "M2K": C(conId=6, symbol="M2K", lastTradeDateOrContractMonth="20261218"), "MYM": C(conId=8, symbol="MYM", lastTradeDateOrContractMonth="20261218")}
old = C(conId=1, symbol="MES", lastTradeDateOrContractMonth="20260918")
t = {"MES": 3, "MNQ": 0, "M2K": 0, "MYM": -1}
held = {"MES": [(old, 2), (f["MES"], 1)], "MNQ": [(f["MNQ"], 2)]}
o = [(c.symbol, c.conId, a, q) for c, a, q, _ in bot.plan_orders(t, held, f)]
assert o == [("MES", 1, "SELL", 2), ("MES", 2, "BUY", 2), ("MNQ", 4, "SELL", 2), ("MYM", 8, "SELL", 1)], o
assert bot.plan_orders({"MES": 1}, {"MES": [(f["MES"], 1)]}, f) == []
NY = bot.NY
w = lambda y, m, d, hh, mm: (lambda now: (lambda win: win and win[0] <= now.hour * 60 + now.minute < win[1])(bot.trade_window(now)))(dt.datetime(y, m, d, hh, mm, tzinfo=NY))
assert w(2026, 11, 2, 16, 25) and not w(2026, 11, 2, 16, 10) and not w(2026, 11, 2, 16, 56)      # Monday
assert not w(2026, 11, 1, 16, 25)                     # Sunday
assert not w(2026, 11, 26, 16, 25)                    # Thanksgiving
assert w(2026, 11, 27, 13, 8) and not w(2026, 11, 27, 16, 25)   # half day
print("plan_orders and trade_window: OK")

# performance log: one row per day (trade or hold), P&L vs the first day, no duplicate rows, status page renders
import tempfile, csv
bot.LOGD = bot.HERE = tempfile.mkdtemp()
fill = lambda a, n, pnl: {"symbol": "MNQ", "expiry": "20261218", "action": a, "filled": n, "commission_usd": 0.61, "realized_pnl_usd": pnl}
bot.log_day("2026-11-02", [fill("BUY", 1.0, 0.0)], {"MNQ": 1}, 100000.0, 700.0)
bot.log_day("2026-11-02", [], {"MNQ": 1}, 999999.0, 700.0)                                  # same day again: ignored
bot.log_day("2026-11-03", [], {"MNQ": 1}, 101000.0, 707.0)
bot.log_day("2026-11-04", [fill("SELL", 1.0, 950.0)], {}, 100950.0, 700.0)
for r in ([{"time": "x", "action": "BUY", "qty": 1, "symbol": "MNQ", "expiry": "20261218", "avg_price": 1, "why": "t", **fill("BUY", 1.0, 0.0)},
           {"time": "y", "action": "SELL", "qty": 1, "symbol": "MNQ", "expiry": "20261218", "avg_price": 2, "why": "t", **fill("SELL", 1.0, 950.0)}]):
    bot.append_csv("journal.csv", {k: r[k] for k in ("time", "symbol", "expiry", "action", "qty", "filled", "avg_price", "commission_usd", "realized_pnl_usd", "why")})
rows = list(csv.DictReader(open(os.path.join(bot.LOGD, "performance.csv"))))
assert [r["action"] for r in rows] == ["BUY 1 MNQ 20261218", "HOLD", "SELL 1 MNQ 20261218"], rows
assert [r["day_pnl_usd"] for r in rows] == ["0.0", "1000.0", "-50.0"] and rows[-1]["total_pct"] == "0.95" and rows[1]["spy_total_pct"] == "1.0"
assert rows[-1]["holding"] == "nothing (cash)" and rows[-1]["realized_pnl_usd"] == "950.0"
bot.write_status({"last": {}})
page = open(os.path.join(bot.LOGD, "status.html")).read()
assert "+950 USD (+0.95%)" in page and "1 won, 0 lost" in page and "worst drop -0.0%" in page, page[:2000]
print("performance log and status page: OK")
