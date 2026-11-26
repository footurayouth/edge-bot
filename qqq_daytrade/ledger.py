"""QQQ noise-band day trade ("F") - paper ledger next to the Edge Stack bot. No orders, no logins.

Each run downloads QQQ 5-minute bars from Yahoo (free, no key), replays the rule on every finished trading day
since START and writes:
  days.csv         one row per day: trades, 1x day return after costs
  performance.csv  1x / 3x equity vs the Edge Stack bot and the S&P 500, all starting from the same dollars
  README.md        the rule in plain words + the latest comparison

Rule (strategy-search/orb_vwap, backtest 2016-26: ~10-12%/yr at 1x, worst drop ~-8..-10%):
  * "Noise band" for each 5-minute slot = average absolute move from the open at that time of day, last 14 days.
  * Upper band = max(today's open, yesterday's close) x (1 + noise); lower band = min(open, prev close) x (1 - noise).
  * Every half hour from 10:00 to 15:30 (at the bar close): above the upper band -> long, below the lower -> short.
  * Exit a long when price closes below max(upper band, VWAP); exit a short above min(lower band, VWAP).
  * Always flat at the close. Cost 1 bp of price per round trip (MNQ fees + slippage ~0.5 bp, padded).
"""
import json, math, urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
NY = ZoneInfo("America/New_York")
START = "2026-11-27"            # first day traded; base = Edge bot equity and SPY close on the last day before it
COST_BP = 1.0
LEV = (1, 3)
LOOK = 14


def yahoo_5m():
    """QQQ 5m bars + each session's regular hours (Yahoo meta.tradingPeriods: real half days included)."""
    u = "https://query1.finance.yahoo.com/v8/finance/chart/QQQ?interval=5m&range=60d&includePrePost=false"
    for a in range(5):
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())
            break
        except Exception as e:
            print("yahoo retry", e)
    else:
        raise SystemExit("Yahoo unavailable")
    r = j["chart"]["result"][0]; q = r["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(NY))
    df = df.dropna(subset=["open", "high", "low", "close"])    # zero-volume bars with real prices are kept (as engine.load)
    df["volume"] = df["volume"].fillna(0.0)
    sessions = {}
    for per in r["meta"]["tradingPeriods"]:
        p0 = per[0] if isinstance(per, list) else per
        a = pd.Timestamp(p0["start"], unit="s", tz="UTC").tz_convert(NY); b = pd.Timestamp(p0["end"], unit="s", tz="UTC").tz_convert(NY)
        sessions[a.date()] = (a.hour * 60 + a.minute, b.hour * 60 + b.minute)
    return df, sessions


def build_days(df, sessions):
    """Same day arrays as orb_vwap/engine.py load() + add_sigma(): RTH bars from 09:30 to the session's real end
    (15:55 last bar, 12:55 on half days), VWAP from typical price, prev close, 14-day noise per 5m slot.
    Every session in `sessions` gets an entry; `complete` = no missing 5m bar, `prev_ok` = previous session complete."""
    df = df.copy()
    df["mins"] = df.index.hour * 60 + df.index.minute
    df["date"] = df.index.date
    G = dict(tuple(df.groupby("date", sort=True)))
    days = []
    for d in sorted(sessions):
        s0, s1 = sessions[d]
        g = G.get(d)
        if g is not None:
            g = g[(g["mins"] >= s0) & (g["mins"] < s1)]
        if g is None or len(g) < 3:
            days.append(dict(date=pd.Timestamp(d), complete=False, empty=True)); continue
        o, h, l, c, v = (g[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        tp = (h + l + c) / 3
        vw = np.cumsum(tp * v) / np.maximum(np.cumsum(v), 1e-9)
        complete = s0 == 570 and len(g) == (s1 - s0) // 5 and g["mins"].iloc[0] == 570
        days.append(dict(date=pd.Timestamp(d), o=o, h=h, l=l, c=c, v=v, vwap=vw, mins=g["mins"].to_numpy(),
                         complete=bool(complete), empty=False, starts_930=bool(g["mins"].iloc[0] == 570),
                         ends_ok=bool(g["mins"].iloc[-1] == s1 - 5)))
    for i, x in enumerate(days):
        p = days[i - 1] if i > 0 else None
        x["prev_ok"] = bool(p is not None and p["complete"])
        x["prevC"] = p["c"][-1] if p is not None and not p["empty"] else np.nan
    mv = np.full((len(days), 78), np.nan)
    for i, x in enumerate(days):
        if not x["empty"]:
            mv[i, (x["mins"] - 570) // 5] = np.abs(x["c"] / x["o"][0] - 1)
    for i, x in enumerate(days):
        x["sigma"] = np.nanmean(mv[i - LOOK:i], axis=0) if i >= LOOK else None
    return days


def trade_day(day, cost_bp=COST_BP):
    """orb_vwap/engine.py noise_breakout, plus a list of the position changes. Returns (net 1x return, trades)."""
    sg = day["sigma"]
    if sg is None or not np.isfinite(day["prevC"]):
        return None, []
    o, c, vw, m = day["o"], day["c"], day["vwap"], day["mins"]
    n = len(c)
    sgk = sg[(m - 570) // 5]
    ub = max(o[0], day["prevC"]) * (1 + sgk); lb = min(o[0], day["prevC"]) * (1 - sgk)
    pos = np.zeros(n); p = 0; log = []
    for k in range(n - 1):
        if (m[k] + 5) % 30 == 0 and m[k] + 5 >= 600 and np.isfinite(ub[k]):
            before = p
            if p > 0 and c[k] < max(ub[k], vw[k]): p = 0
            if p < 0 and c[k] > min(lb[k], vw[k]): p = 0
            if p == 0:
                if c[k] > ub[k]: p = 1
                elif c[k] < lb[k]: p = -1
            if p != before:
                t = f"{(m[k] + 5) // 60:02d}:{(m[k] + 5) % 60:02d}"
                log.append(f"{t} {'LONG' if p > 0 else 'SHORT' if p < 0 else 'FLAT'} @{c[k]:.2f}")
        pos[k] = p
    pos[-1] = 0
    if pos[-2] != 0:
        log.append(f"close FLAT @{c[-1]:.2f}")
    if not np.any(pos):
        return 0.0, []
    r = np.diff(c) / c[:-1]
    sides = np.abs(np.diff(np.concatenate([[0], pos]))).sum()
    return float(np.sum(pos[:-1] * r)) - sides * cost_bp / 2 / 1e4, log


def edge_rows():
    p = ROOT / "logs" / "performance.csv"
    e = pd.read_csv(p, parse_dates=["date"]).drop_duplicates("date", keep="last")
    return e.set_index("date")[["equity_usd", "spy_close"]]


def main():
    now = datetime.now(NY)
    days = build_days(*yahoo_5m())
    dp = HERE / "days.csv"
    old = pd.read_csv(dp, parse_dates=["date"]) if dp.exists() else pd.DataFrame(columns=["date", "ret_1x", "trades"])
    have = set(old.date)
    new = []
    for x in days:
        d = x["date"]
        finished = d.date() < now.date() or (now.hour * 60 + now.minute) >= 16 * 60 + 30
        if d < pd.Timestamp(START) or d in have or not finished or x["empty"] or not x["starts_930"] or not x["ends_ok"]:   # never book a day without its close
            continue
        old_enough = (pd.Timestamp(now.date()) - d).days >= 2        # Yahoo gaps that never fill: accept, flagged
        if not (x["complete"] and x["prev_ok"]) and not old_enough:
            print(d.date(), "data incomplete, retry next run"); continue
        ret, log = trade_day(x)
        if not (x["complete"] and x["prev_ok"]):
            log = ["[incomplete Yahoo data]"] + log
        if ret is None:
            continue
        new.append(dict(date=d, ret_1x=round(ret, 6), trades="; ".join(log) or "no trade"))
        print(d.date(), f"{ret:+.3%}", "; ".join(log) or "no trade")
    D = (pd.concat([old, pd.DataFrame(new)], ignore_index=True) if len(old) else pd.DataFrame(new)) if new else old
    if D.empty:
        print("no finished trading day since", START); return
    D = D.sort_values("date"); D.to_csv(dp, index=False)

    E = edge_rows()
    base_rows = E[E.index < pd.Timestamp(START)]
    base_eq, base_spy = float(base_rows.equity_usd.iloc[-1]), float(base_rows.spy_close.iloc[-1])
    rows = []; eq = {L: base_eq for L in LEV}
    for r in D.itertuples():
        for L in LEV:
            eq[L] *= 1 + L * r.ret_1x
        e = E.loc[r.date] if r.date in E.index else None
        rows.append(dict(date=r.date.date(), qqq_day_1x_pct=round(100 * r.ret_1x, 3),
                         **{f"qqq_{L}x_usd": round(eq[L], 2) for L in LEV},
                         edge_usd=None if e is None else float(e.equity_usd),
                         spy_close=None if e is None else float(e.spy_close),
                         **{f"qqq_{L}x_pct": round(100 * (eq[L] / base_eq - 1), 2) for L in LEV},
                         edge_pct=None if e is None else round(100 * (float(e.equity_usd) / base_eq - 1), 2),
                         spy_pct=None if e is None else round(100 * (float(e.spy_close) / base_spy - 1), 2),
                         trades=r.trades))
    P = pd.DataFrame(rows); P.to_csv(HERE / "performance.csv", index=False)
    last = P.iloc[-1]; le = P.dropna(subset=["edge_pct"])
    le = le.iloc[-1] if len(le) else None
    fmt = lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.2f}%"
    table = (f"| since {START} (start ${base_eq:,.0f} each) | total |\n|---|---|\n"
             f"| QQQ day trade 1x | {fmt(last.qqq_1x_pct)} |\n| QQQ day trade 3x | {fmt(last.qqq_3x_pct)} |\n"
             f"| Edge Stack bot (as of {le.date if le is not None else '-'}) | {fmt(None if le is None else le.edge_pct)} |\n"
             f"| S&P 500 (SPY price, same date) | {fmt(None if le is None else le.spy_pct)} |\n")
    readme = (HERE / "README.md").read_text().split("<!-- table -->")[0]
    (HERE / "README.md").write_text(readme + "<!-- table -->\n" + table + f"\nLast day {last.date} (Edge/SPY columns fill in after the Edge bot's run).\n")
    print(table)


if __name__ == "__main__":
    main()
