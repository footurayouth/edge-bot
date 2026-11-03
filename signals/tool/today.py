"""EDGE STACK - daily signal tool.

Run:  python3 tool/today.py            (or double-click "Edge Stack.command")
Edit: tool/config.json                  (risk preset, account sizes, how you execute)

Prints today's target positions/orders for the three sleeves and writes tool/dashboard.html.
Best times: futures sleeves 20-50 min after the US close (16:20-16:50 New York = normally 23:20-23:50 in
Bulgaria/Greece time); crypto sleeve any time after 00:00 UTC (03:00 local), ideally in the morning."""
import os, sys, json, datetime as dt, warnings, base64, io
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mpl"))
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from lab import data as D
import edge

D.CACHE = os.path.join(HERE, "cache"); os.makedirs(D.CACHE, exist_ok=True)   # live data never touches research cache
PRESETS = json.load(open(os.path.join(HERE, "presets.json")))


def load_config():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    p = PRESETS[cfg["risk"]]
    return cfg, p


CUT = None   # set in main: drops today's still-forming US bar when run before 16:15 New York time


def fresh_ohlc(t):
    d = D.ohlc(t, refresh=True)
    return d.loc[:CUT] if CUT is not None else d


def us_cutoff():
    from zoneinfo import ZoneInfo
    ny = dt.datetime.now(ZoneInfo("America/New_York"))
    ready = 13 * 60 + 5 if str(ny.date()) in HALF_DAYS else 16 * 60 + 15
    if ny.hour * 60 + ny.minute < ready:
        return pd.Timestamp(ny.date()) - pd.Timedelta(days=1)
    return None


def futures_orders(cfg, p, out):
    eq = float(cfg["futures_account_usd"])
    spy_o = fresh_ohlc("SPY")
    o = {e: (spy_o if e == "SPY" else fresh_ohlc(e)) for e in edge.IBS_UNIVERSE}
    spy = spy_o["Close"]
    ndx = fresh_ohlc("^NDX")["Close"]
    last_day = spy.index[-1]
    alerts = out.setdefault("alerts", [])
    idx_px = {e: fresh_ohlc(v[3]) for e, v in edge.FUTURES.items()}
    ends = {**{e: o[e].index[-1] for e in o}, "^NDX": ndx.index[-1], **{v[3]: idx_px[e].index[-1] for e, v in edge.FUTURES.items()}}
    if len(set(ends.values())) > 1:
        alerts.append("STOP: Yahoo data ends on different dates " + str({k: str(v.date()) for k, v in ends.items()})
                      + " - some bars are not published yet. Re-run in 10 minutes; do NOT trade these targets.")
    # sleeve 1
    td = edge.trend_detail(ndx)
    trend_notional = p["L_trend"] * td["f"]
    # sleeve 2
    edge.SLOT_ON, edge.SLOT_OFF = p["slot_on"], p["slot_off"]
    W, log, held = edge.ibs_positions(o, spy)
    ibs_today = W.iloc[-1]
    regime_on = spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]
    ibs_now = {e: float(v) for e, v in ibs_today.items() if v > 0}
    today_log = log[log.date == last_day]
    # combine per instrument (same edge.stack() as the backtest, incl. the gross cap)
    f_series = pd.Series(td["f"], index=[last_day])
    Ws = edge.stack(W.iloc[[-1]], f_series, p["L_trend"], p["max_gross"])
    # month-end overlay (sleeve 2b)
    tlt = fresh_ohlc("TLT")["Close"]
    ov, s_me, dl_me = edge.monthend_weights(spy, tlt, p.get("monthend", 0.0))
    Wt = edge.add_overlay(Ws, ov.reindex(Ws.index).fillna(0.0), p["max_gross"]).iloc[-1]
    me_today = float(Wt["SPY"] - Ws.iloc[-1]["SPY"])
    me = {"days_left": int(dl_me.iloc[-1]), "s_pct": round(float(s_me.iloc[-1]) * 100, 2), "mes_x": round(me_today, 2),
          "provisional": "SHORT" if s_me.iloc[-1] > 0 else "LONG"}
    target = {e: float(Wt.get(e, 0.0)) for e in edge.IBS_UNIVERSE}
    contract_usd, lvl = {}, {}
    for e, (fut, fname, m, idx_t) in edge.FUTURES.items():
        lvl[e] = idx_px[e]["Close"].iloc[-1]; contract_usd[e] = m * lvl[e]
    n, exact, gross_rounded = edge.round_contracts(target, eq, contract_usd, p["max_gross"])
    lost = [edge.FUTURES[e][0] for e in target if target[e] >= 0.25 and n[e] == 0]
    if cfg.get("route", "ibkr_futures") != "t212" and (abs(gross_rounded - sum(target.values())) > 0.25 or lost):
        alerts.append(f"SIZE: whole contracts give {gross_rounded:.2f}x vs the backtest's {sum(target.values()):.2f}x"
                      + (f"; {', '.join(lost)} rounds to 0" if lost else "") + ". The account is too small for this preset "
                      "(README: Setup) - expect results well below the backtest, or use the small-account route.")
    rows = []
    for e, (fut, fname, m, idx_t) in edge.FUTURES.items():
        rows.append({"etf": e, "future": fut, "name": fname, "x_equity": round(target[e], 2),
                     "notional_usd": round(target[e] * eq), "contract_usd": round(contract_usd[e]),
                     "contracts_exact": round(exact[e], 2), "contracts": n[e],
                     "cfd": {"SPY": "US 500 index CFD", "QQQ": "US Tech 100 index CFD", "IWM": "US 2000 / Russell 2000 CFD",
                             "DIA": "US 30 / Wall Street CFD"}[e]})
    t_capped = min(trend_notional, p["max_gross"])                    # same split as edge.stack (trend first)
    ibs_capped = {e: float(Ws.iloc[-1][e]) - (t_capped if e == "QQQ" else 0.0) for e in edge.IBS_UNIVERSE}
    if me_today != 0:
        ibs_capped["SPY"] = ibs_capped.get("SPY", 0.0) + me_today
    cfd_name = {"SPY": "USA500", "QQQ": "TECH100", "IWM": "USA2000", "DIA": "USA30"}
    small = {"trend_etp_2x_pct": round(t_capped / 2 * 100, 1),
             "qqq3_usd": round(t_capped / 3 * eq), "qqq3_pct": round(t_capped / 3 * 100, 1),
             "cfd_cash_usd": round(eq - t_capped / 3 * eq),
             "ibs_cfd_notional": {cfd_name[e]: round(v * eq) for e, v in ibs_capped.items() if abs(v) > 1e-9}}
    ibs_vals = {e: float(edge.ibs(o[e]).iloc[-1]) for e in edge.IBS_UNIVERSE}
    # ---- today vs yesterday targets (both from the same rules) -> concrete actions
    f_all = edge.trend_fraction(ndx).reindex(spy.index).ffill().fillna(0.0)
    ovr = ov.reindex(spy.index).fillna(0.0)
    def route_targets(k):
        d = spy.index[k]
        Sk = edge.stack(W.iloc[[k]], pd.Series(f_all.iloc[k], index=[d]), p["L_trend"], p["max_gross"])
        Tk = edge.add_overlay(Sk, ovr.iloc[[k]], p["max_gross"]).iloc[0]
        t = min(p["L_trend"] * f_all.iloc[k], p["max_gross"])
        cfd = {e: float(Tk[e]) - (t if e == "QQQ" else 0.0) for e in edge.IBS_UNIVERSE}
        return {"qqq3": t / 3, "cfd": cfd, "fut": {e: float(Tk[e]) for e in edge.IBS_UNIVERSE}}
    T0, T1 = route_targets(-1), route_targets(-2)
    why = {}
    for r_ in today_log.to_dict("records"):
        why[r_["etf"]] = ("IBS dip-buy signal (" + r_["why"] + ")") if r_["action"] == "BUY" else ("IBS exit: " + r_["why"])
    if me["mes_x"] != 0:
        why["SPY"] = (why.get("SPY", "") + "; " if "SPY" in why else "") + "month-end trade (1 day)"
    elif me["days_left"] == 0 and abs(T1["cfd"]["SPY"] - T0["cfd"]["SPY"]) > 1e-9 and "SPY" not in why:
        why["SPY"] = "month-end trade ends today"
    regime_prev = spy.iloc[-2] > spy.rolling(200).mean().iloc[-2]
    cfd_px = {e: float(idx_px[e]["Close"].iloc[-1]) for e in edge.IBS_UNIVERSE}
    actions, holds = [], []
    if cfg.get("route", "ibkr_futures") == "t212":
        dq = (T0["qqq3"] - T1["qqq3"]) * eq
        if abs(T0["qqq3"] - T1["qqq3"]) >= 0.10 or ((T0["qqq3"] > 0) != (T1["qqq3"] > 0)):
            actions.append(f"T212 Invest: {'BUY' if dq > 0 else 'SELL'} QQQ3 for ${abs(dq):,.0f} -> total QQQ3 ${T0['qqq3'] * eq:,.0f} "
                           f"(do it tomorrow 16:30-18:25; reason: Nasdaq trend score now {td['f']:.2f})")
        holds.append(f"T212 Invest: QQQ3 worth about ${T0['qqq3'] * eq:,.0f}. If your QQQ3 value is off by more than "
                     f"${0.10 * eq:,.0f}, buy/sell the difference. (No stop-loss: you exit when the trend score falls.)")
        for e in edge.IBS_UNIVERSE:
            a, b = T0["cfd"][e], T1["cfd"][e]
            nm = cfd_name[e]
            units = abs(a) * eq / cfd_px[e]
            if abs(a - b) * eq >= 0.05 * eq or ((abs(a) > 1e-9) != (abs(b) > 1e-9)) or (a * b < 0):
                if abs(a) < 1e-9:
                    actions.append(f"T212 CFD: CLOSE your whole {nm} position ({why.get(e, 'rule exit')})")
                elif abs(b) < 1e-9 or a * b < 0:
                    actions.append(f"T212 CFD: {'BUY' if a > 0 else 'SELL (short)'} {nm} {units:.2f} units (~${abs(a) * eq:,.0f})"
                                   f" ({why.get(e, 'new position')})")
                else:
                    reason = why.get(e, "SPY regime change -> position size changes" if regime_prev != regime_on
                                     else "positions share the exposure cap -> size changes")
                    actions.append(f"T212 CFD: CHANGE {nm} to {'long' if a > 0 else 'short'} {units:.2f} units (~${abs(a) * eq:,.0f}) ({reason})")
            if abs(a) > 1e-9:
                dh = held.get(e)
                extra = (f"; held {dh} days, auto-exit at the first close above the previous day's high or after day "
                         f"{edge.IBS_MAXHOLD}") if dh is not None else ""
                if e == "SPY" and me["mes_x"] != 0:
                    extra += "; month-end part closes at tomorrow evening's run"
                holds.append(f"T212 CFD: {nm} {'long' if a > 0 else 'short'} ~{units:.2f} units (~${abs(a) * eq:,.0f}){extra}")
        holds.append(f"T212 CFD account cash: about ${(1 - T0['qqq3']) * eq:,.0f} (move cash between Invest and CFD to keep this).")
    else:
        n1, _, _ = edge.round_contracts(T1["fut"], eq, contract_usd, p["max_gross"])
        for e in edge.IBS_UNIVERSE:
            fut = edge.FUTURES[e][0]
            if n[e] != n1[e]:
                actions.append(f"IBKR: {'BUY' if n[e] > n1[e] else 'SELL'} {abs(n[e] - n1[e])} {fut} -> hold {n[e]} ({why.get(e, 'trend/size change')})")
            if n[e]:
                holds.append(f"IBKR: {n[e]} {fut}")
    out["plan"] = {"actions": actions, "holds": holds}
    out["_raw"] = {"eq": eq, "date": str(last_day.date()), "T0": T0, "why": why, "held": dict(held), "cfd_px": cfd_px,
                   "f": float(td["f"]), "me": me_today}
    out["futures"] = {"date": str(last_day.date()), "equity_usd": eq, "rows": rows,
                      "trend": {"f": round(td["f"], 3), "x": round(trend_notional, 2), "ndx": round(td["close"], 1),
                                "rv20": round(td["rv20"] * 100, 1), "above_smas": f"{td['above']}/7"},
                      "monthend": me,
                      "ibs": {"held": {e: f"{n} days" for e, n in held.items()}, "regime_on": bool(regime_on),
                              "today_ibs": {e: round(v, 3) for e, v in ibs_vals.items()},
                              "actions": today_log[["etf", "action", "why"]].to_dict("records")},
                      "gross_x": round(sum(target.values()), 2), "gross_rounded_x": round(gross_rounded, 2),
                      "max_gross": p["max_gross"], "small": small, "mode": cfg.get("mode", "futures")}


def crypto_orders(cfg, p, out):
    eq = float(cfg["crypto_account_usd"])
    if eq <= 0:                                   # conservative copy: crypto switched off (crypto_account_usd = 0)
        out["crypto"] = {"date": "off", "equity_usd": 0, "coins": {}, "cash_pct": 100.0}
        out["crypto_share_now"] = 0.0; out["_rawc"] = None
        return
    btc = D.close_complete("BTC-USD", refresh=True); eth = D.close_complete("ETH-USD", refresh=True)
    Wall = edge.crypto_weights(btc, eth).fillna(0.0)
    W, Wy = Wall.iloc[-1], Wall.iloc[-2]
    age_h = (pd.Timestamp.now("UTC").tz_localize(None) - (btc.index[-1] + pd.Timedelta(days=1))).total_seconds() / 3600
    if age_h > 12:
        out.setdefault("alerts", []).append(
            f"CRYPTO: these targets are from a close {age_h:.0f}h ago. Do not trade them now; re-run after the next "
            "00:00 UTC (03:00 EEST) and trade that output (a day-late crypto signal cost ~4-5%/yr in the backtest).")
    out["crypto_share_now"] = round(eq / (eq + float(cfg["futures_account_usd"])), 3)
    det = {}
    for name, c in (("BTC", btc), ("ETH", eth)):
        vol = c.pct_change().rolling(edge.CRYPTO_VOLWIN).std().iloc[-1] * np.sqrt(365)
        det[name] = {"price": round(c.iloc[-1], 2), "trend": round(edge.crypto_trend(c).iloc[-1], 2),
                     "vol30": round(vol * 100, 1), "weight": round(float(W[name]), 3),
                     "usd": round(float(W[name]) * eq)}
    out["_rawc"] = {"eq": eq, "date": str(btc.index[-1].date()), "W": {k: float(W[k]) for k in ("BTC", "ETH")},
                    "px": {"BTC": btc, "ETH": eth}, "age_h": age_h,
                    "det": {k: {"trend": round(float(edge.crypto_trend(c_).iloc[-1]), 2),
                                "vol30": round(float(c_.pct_change().rolling(edge.CRYPTO_VOLWIN).std().iloc[-1] * np.sqrt(365) * 100), 1)}
                            for k, c_ in (("BTC", btc), ("ETH", eth))}}
    plan = out.setdefault("plan", {"actions": [], "holds": []})
    big = sum(abs(float(W[n]) - float(Wy[n])) for n in ("BTC", "ETH")) >= 0.10 or any((W[n] > 0) != (Wy[n] > 0) for n in ("BTC", "ETH"))
    for name in ("BTC", "ETH"):
        a, b = float(W[name]), float(Wy[name])
        if big and abs(a - b) > 0.005:
            plan["actions"].append(f"Revolut X: {'BUY' if a > b else 'SELL'} {name} for ${abs(a - b) * eq:,.0f} -> hold ${a * eq:,.0f} "
                                   f"(trend {det[name]['trend']}, 30d vol {det[name]['vol30']}%)")
        plan["holds"].append(f"Revolut X: {name} worth about ${a * eq:,.0f} ({a * 100:.0f}% of the crypto account)")
    plan["holds"].append(f"Revolut X: rebalance only if |BTC off| + |ETH off| > ${0.10 * eq:,.0f}; rest stays in cash (EUR/USD).")
    out["crypto"] = {"date": str(btc.index[-1].date()), "equity_usd": eq, "coins": det,
                     "cash_pct": round((1 - W.sum()) * 100, 1)}


HALF_DAYS = edge.HALF_DAYS


def freshness_warning(us_date):
    from zoneinfo import ZoneInfo
    ny = dt.datetime.now(ZoneInfo("America/New_York"))
    d, m = pd.Timestamp(us_date).date(), ny.hour * 60 + ny.minute
    close, cme_stop, ready = (13 * 60, 13 * 60 + 15, 13 * 60 + 5) if str(ny.date()) in HALF_DAYS else (16 * 60, 17 * 60, 16 * 60 + 15)
    if d == ny.date():
        if m < ready:
            return f"WARNING: today's bar is not final yet. Re-run at {ready // 60}:{ready % 60:02d} New York time."
        if m >= cme_stop:
            return (f"LATE: CME micros stopped at {cme_stop // 60}:{cme_stop % 60:02d} ET. The backtest trades at today's close; "
                    "trading these in the next session is NOT what was tested.")
        return ""
    if ny.weekday() < 5 and m >= ready:
        if pd.Timestamp(ny.date()) not in edge.nyse_holidays(ny.year):
            return (f"STOP: today's US close ({ny.date()}) is not in the data yet (Yahoo delay) - re-run in 10 minutes. "
                    "If the US market was closed today, ignore this.")
        return f"NOTE: today's US close is not in the data (holiday) - signals are from {d}."
    return (f"NOTE: signals are from the {d} close (market closed or not yet closed today). If you traded them then, "
            "do nothing; the BUY/SELL lines below are that day's, not new ones.")


def local_window():
    """Tonight's evening window (16:20-16:55 New York) in the user's local time zone (Bulgaria)."""
    from zoneinfo import ZoneInfo
    ny, loc = ZoneInfo("America/New_York"), ZoneInfo("Europe/Sofia")
    d = dt.datetime.now(ny).date()
    a = dt.datetime(d.year, d.month, d.day, 16, 20, tzinfo=ny).astimezone(loc)
    b = dt.datetime(d.year, d.month, d.day, 16, 55, tzinfo=ny).astimezone(loc)
    return f"{a:%H:%M}-{b:%H:%M}"


def print_report(out, cfg, p):
    f = out["futures"]; c = out["crypto"]
    for w in out["alerts"]:
        print("!" * 78 + "\n" + w + "\n" + "!" * 78)
    pl = out.get("plan", {"actions": [], "holds": []})
    print("=" * 78)
    print(f" TODAY'S ACTIONS   (evening trades: tonight {local_window()} Bulgaria time, Mon-Fri; crypto: mornings, any day)")
    print("=" * 78)
    for a in pl["actions"] or ["Nothing new today - keep your positions (first time? set up everything listed under WHAT YOU SHOULD HOLD NOW)."]:
        print("  * " + a)
    print("\n WHAT YOU SHOULD HOLD NOW")
    for h in pl["holds"]:
        print("  - " + h)
    print("\n No stop-loss / take-profit: every exit is a rule and the tool tells you when (tested: SL/TP made results worse).")
    print(" IBS dip-buys last 1-10 trading days (avg ~4); month-end trades exactly 1 day; QQQ3/crypto follow the trend.\n")
    print("=" * 78)
    print(f" DETAILS  |  preset: {cfg['risk']}  |  data: US close {f['date']}, crypto UTC close {c['date']}")
    print("=" * 78)
    print(f"\n[1+2] INDEX ACCOUNT (${f['equity_usd']:,.0f}) - trade 20-50 min after the US close (normally 23:20-23:50 your time; T212 CFDs pause at 23:59)")
    t = f["trend"]
    print(f"  Nasdaq trend: NDX {t['ndx']} is above {t['above_smas']} of its SMAs, 20d vol {t['rv20']}% -> f={t['f']} -> {t['x']}x equity in Nasdaq")
    ib = f["ibs"]
    print(f"  IBS swing: regime {'ON (SPY>SMA200)' if ib['regime_on'] else 'OFF (SPY<SMA200, small size)'}; "
          f"today's IBS {ib['today_ibs']}")
    me = f["monthend"]
    if me["mes_x"] != 0:
        print(f"  MONTH-END: tomorrow is the last trading day -> MES {me['mes_x']:+.2f}x equity from this close to tomorrow's close (s={me['s_pct']}%)")
    elif me["days_left"] == 0:
        print("  MONTH-END: today was the last trading day -> month-end trade must be FLAT now")
    else:
        print(f"  Month-end trade: {me['days_left']} trading days left; entry at the close 1 day before month-end (now {me['provisional']}, s={me['s_pct']}%)")
    for a in ib["actions"]:
        print(f"     -> {a['action']} {a['etf']} ({a['why']})")
    if ib["held"]:
        print(f"     holding: {ib['held']}")
    t212 = cfg.get("route", "ibkr_futures") == "t212"
    if not t212:
        print(f"\n  TARGET POSITIONS (target {f['gross_x']}x equity, after rounding {f['gross_rounded_x']}x, cap {f['max_gross']}x):")
        print(f"  {'future':7}{'x equity':>9}{'notional $':>13}{'1 contract $':>14}{'contracts':>11}")
        for r in f["rows"]:
            print(f"  {r['future']:7}{r['x_equity']:>9}{r['notional_usd']:>13,}{r['contract_usd']:>14,}{r['contracts']:>7} ({r['contracts_exact']})")
        print("  -> adjust your futures holdings to these contract counts (negative = short; only the month-end trade can be short).")
    sm = f["small"]
    if cfg.get("route", "ibkr_futures") == "t212":
        print(f"\n  >>> TRADING 212 ROUTE (your setup) - index equity ${f['equity_usd']:,.0f}")
        print(f"   T212 Invest: hold QQQ3 (3x Nasdaq-100) worth ${sm['qqq3_usd']:,} ({sm['qqq3_pct']}% of index equity).")
        print(f"      Change it only if your QQQ3 value differs by >10% of index equity; trade next day 16:30-18:25 your time.")
        print(f"   T212 CFD account: keep ${sm['cfd_cash_usd']:,} there (= index equity minus QQQ3).")
        if sm["ibs_cfd_notional"]:
            for k, v in sm["ibs_cfd_notional"].items():
                print(f"      CFD {k}: {'LONG' if v > 0 else 'SHORT'} notional ${abs(v):,}  (units = notional / index price)")
        else:
            print("      CFDs: no positions (close any open USA500/TECH100/USA2000/USA30 CFDs)")
    else:
        print(f"  SMALL-ACCOUNT alternative (no futures): hold {sm['trend_etp_2x_pct']}% of the account in a 2x Nasdaq-100 UCITS ETP")
        print(f"     (buy/sell next EU morning), and IBS via index CFDs with notional {sm['ibs_cfd_notional'] or 'none today'}")
    print(f"\n[3] CRYPTO ACCOUNT (${c['equity_usd']:,.0f}) - any time after 03:00 EEST, ideally mornings")
    for k, v in c["coins"].items():
        print(f"  {k}: price {v['price']:,}  trend {v['trend']}  vol30 {v['vol30']}%  -> target {v['weight']*100:.1f}% = ${v['usd']:,}")
    print(f"  cash/T-bills: {c['cash_pct']}%  (trade only if |BTC-target|+|ETH-target| > 10% of the crypto account, or a target goes to/from 0)")
    print(f"  crypto share of total: {out['crypto_share_now']:.0%} vs preset {p['crypto_share']:.0%} - re-balance the two accounts to the preset each quarter")
    print("  Update futures_account_usd / crypto_account_usd to today's balances before every run (sizes and the cap use them).")
    print("\nDashboard: tool/dashboard.html")


def equity_png():
    fn = os.path.join(ROOT, "research", "final", "_preset_curves.pkl")
    if not os.path.exists(fn):
        return ""
    curves = pd.read_pickle(fn)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    for k, s in curves.items():
        ax.plot((1 + s).cumprod(), label=k, lw=1.4)
    ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(frameon=False); ax.set_title("Backtest: what $1 grew to (after costs; futures cost model - your T212 setup is a few %/yr lower)", fontsize=10)
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}" if y >= 1 else f"${y:.1f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ""))
    b = io.BytesIO(); fig.tight_layout(); fig.savefig(b, format="png"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def b64(name):
    """Embed an image so the dashboard is one self-contained file (works when emailed/shared)."""
    try:
        return "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(HERE, name), "rb").read()).decode()
    except Exception:
        return name


def write_dashboard(out, cfg, p):
    f = out["futures"]; c = out["crypto"]
    st = PRESETS[cfg["risk"]].get("stats", {})
    rows = "".join(f"<tr><td>{r['future']}</td><td>{r['name']}</td><td>{r['x_equity']}x</td><td>${r['notional_usd']:,}</td>"
                   f"<td>${r['contract_usd']:,}</td><td><b>{r['contracts']}</b> <small>({r['contracts_exact']})</small></td></tr>"
                   for r in f["rows"])
    acts = "".join(f"<li><b>{a['action']}</b> {a['etf']} – {a['why']}</li>" for a in f["ibs"]["actions"]) or "<li>No IBS entries/exits today</li>"
    pl = out.get("plan", {"actions": [], "holds": []})
    plan_html = ('<div class="card" style="border-width:2px;border-color:var(--acc)"><h2>Today\'s actions</h2><ol>'
                 + "".join(f"<li><b>{a}</b></li>" for a in (pl["actions"] or ["Nothing to trade today – keep your positions."]))
                 + '</ol><h2>What you should hold now</h2><ul>' + "".join(f"<li>{h}</li>" for h in pl["holds"])
                 + '</ul><p class="mut">No stop-loss / take-profit: every exit is a rule and the tool tells you when (tested: SL/TP made results worse). '
                   'IBS dip-buys last 1–10 trading days (avg ~4); the month-end trade exactly 1 day; QQQ3 and crypto follow the trend.</p></div>'
                 + '<h2 style="margin-top:22px">Details and reasoning</h2>')
    fut_table = ("" if cfg.get("route", "ibkr_futures") == "t212" else
                 '<div class="scroll"><table><tr><th>Future</th><th>Name</th><th>Exposure</th><th>Notional</th><th>1 contract</th>'
                 '<th>Contracts</th></tr>' + rows + '</table></div>')
    coins = "".join(f"<tr><td>{k}</td><td>{v['price']:,}</td><td>{v['trend']}</td><td>{v['vol30']}%</td><td><b>{v['weight']*100:.1f}%</b></td><td>${v['usd']:,}</td></tr>"
                    for k, v in c["coins"].items())
    stats_rows = "".join(f"<tr><td>{k}</td>" + "".join(f"<td>{x}</td>" for x in v) + "</tr>" for k, v in st.items())
    mm = f["monthend"]
    me_html = (f"<b>MES {mm['mes_x']:+.2f}x equity until tomorrow&#39;s close</b>" if mm["mes_x"]
               else f"{mm['days_left']} trading days left (now {mm['provisional']}, s={mm['s_pct']}%)")
    try:
        pr = json.load(open(os.path.join(HERE, "proof.json")))
        tiles = "".join(f"<div class='tile'><div class='tn'>{a}</div><div class='tl'>{b}</div></div>" for a, b in pr["headline"])
        areas = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{t}</td><td>{c}</td><td><b>{d}</b></td></tr>" for a, b, t, c, d in pr["areas"])
        res = "".join(f"<tr><td>{a}</td><td><b>{b}</b></td><td>{c}</td></tr>" for a, b, c in pr["results"])
        cri = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in pr["crises"])
        rules_ = "".join(f"<li>{r}</li>" for r in pr["rules"])
        proof_html = (f"<details class='card'><summary style='cursor:pointer;font-weight:700;font-size:17px'>Strategy proof - how this was researched and tested (click)</summary>"
                      f"<div class='tiles'>{tiles}</div>"
                      f"<h2>Research areas</h2><div class='scroll'><table><tr><th>Area</th><th>Variants</th><th>Trades simulated</th><th>Data period</th><th>Result</th></tr>{areas}</table></div>"
                      f"<h2>Testing rules</h2><ul>{rules_}</ul>"
                      f"<h2>Results of the chosen setup (after all costs)</h2><div class='scroll'><table><tr><th>Test</th><th>Average yearly return</th><th>Worst drop</th></tr>{res}</table></div>"
                      f"<h2>Crisis periods (loss from the S&amp;P's peak to its low)</h2><div class='scroll'><table><tr><th>Crisis</th><th>Edge conservative (T212)</th><th>S&amp;P 500</th></tr>{cri}</table></div>"
                      f"<p class='mut'>Backtests are not guarantees; markets change and real results are usually lower than backtests.</p></details>")
    except Exception:
        proof_html = ""
    png = equity_png()
    img = f'<img alt="Backtest equity curves" src="data:image/png;base64,{png}">' if png else ""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge Stack Conservative</title><style>
:root{{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e3de;--acc:#0b6e4f;--warn:#a4472d}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151514;--fg:#ecece8;--mut:#a3a39c;--card:#1f1f1d;--line:#33332f;--acc:#4cc39a;--warn:#e08a6d}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:16px;position:relative}}
html{{--bimg:url({b64("banner_ai.jpg")});--bgimg:url({b64("bg_ai.jpg")})}}
body::before{{content:"";position:fixed;inset:0;background:var(--bgimg) center/cover no-repeat;opacity:.32;z-index:-1;pointer-events:none}}
.card{{backdrop-filter:blur(2px)}}
.hero{{position:relative;border-radius:16px;overflow:hidden;min-height:300px;display:flex;align-items:flex-end;margin:0 0 14px;
  background:#2b2b2b var(--bimg) 30% 32%/cover no-repeat;box-shadow:0 6px 24px rgba(0,0,0,.18)}}
.hero::after{{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,.05) 0%,rgba(0,0,0,.55) 100%)}}
.hero div{{position:relative;z-index:1;padding:18px 20px;color:#fff;text-shadow:0 1px 6px rgba(0,0,0,.5)}}
.hero h1{{margin:0;font-size:28px;letter-spacing:.3px}} .hero p{{margin:4px 0 0;opacity:.92}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.tile{{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}
.tn{{font-size:22px;font-weight:800;color:var(--acc)}} .tl{{font-size:13px;color:var(--mut)}}
main{{max-width:900px;margin:auto}} h1{{font-size:22px;margin:4px 0}} h2{{font-size:17px;margin:0 0 8px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} td,th{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}}
.mut{{color:var(--mut)}} .big{{font-size:18px;color:var(--acc)}} img{{max-width:100%;height:auto}} .scroll{{overflow-x:auto}}
</style></head><body><main>
<div class="hero"><div><h1>Edge Stack Conservative</h1><p>{cfg['risk']} · {dt.datetime.now():%A %d %B %Y} · stay patient, follow the rules</p></div></div>{out.get('simple_html', '')}{proof_html}<details><summary class="mut" style="cursor:pointer;margin:14px 0">Details and reasoning (click to open)</summary>{"".join(f'<div class="card" style="border-color:var(--warn);color:var(--warn)"><b>{a}</b></div>' for a in out["alerts"])}<div class="mut">US close {f['date']} · crypto UTC close {c['date']} · generated {dt.datetime.now():%Y-%m-%d %H:%M}</div>
<div class="card"><h2>1+2 · Index account (${f['equity_usd']:,.0f}) – trade 20–50 min after the US close (≈23:20–23:50 your time)</h2>
<p>Nasdaq trend: NDX {f['trend']['ndx']}, above {f['trend']['above_smas']} SMAs, 20-day vol {f['trend']['rv20']}% → <b>{f['trend']['x']}x</b> Nasdaq exposure.</p>
<p>Month-end trade: {me_html}</p>
<p>IBS swing: regime <b>{'ON' if f['ibs']['regime_on'] else 'OFF'}</b>; today's IBS {f['ibs']['today_ibs']}</p><ul>{acts}</ul>
{fut_table}
<p class="big">Target exposure {f['gross_x']}x equity (after rounding {f['gross_rounded_x']}x; cap {f['max_gross']}x)</p>
<p class="big">Trading 212 route: QQQ3 ${f['small']['qqq3_usd']:,} ({f['small']['qqq3_pct']}% of index equity) · CFD account cash ${f['small']['cfd_cash_usd']:,} · CFDs: {f['small']['ibs_cfd_notional'] or 'none'}</p></div>
<div class="card"><h2>3 · Crypto account (${c['equity_usd']:,.0f}) – mornings EEST</h2>
<div class="scroll"><table><tr><th>Coin</th><th>Price</th><th>Trend</th><th>Vol 30d</th><th>Target</th><th>USD</th></tr>{coins}</table></div>
<p>Cash / T-bills {c['cash_pct']}%. Trade only when holdings drift &gt;10% from target.</p></div>
<div class="card"><h2>Backtest of this preset (after costs)</h2><div class="scroll"><table>
<tr><th>Period</th><th>CAGR %</th><th>Max DD %</th><th>Sharpe</th><th>Worst 12m %</th></tr>{stats_rows}</table></div>{img}
<p class="mut">Backtests are not guarantees. Expect long flat periods and drawdowns as deep as shown, or deeper.</p></div></details>
</main></body></html>"""
    open(os.path.join(HERE, "dashboard.html"), "w").write(html)


if __name__ == "__main__":
    import simple
    CUT = us_cutoff()
    cfg, p = load_config()
    out = {}
    futures_orders(cfg, p, out)
    crypto_orders(cfg, p, out)
    w = freshness_warning(out["futures"]["date"])
    out["alerts"][:0] = [w] if w else []
    qc = fresh_ohlc("QQQ")["Close"]
    try:
        rf = D.rf_daily(qc.index)                                    # T-bill, for QQQ3's 2x financing drag
    except Exception:
        rf = None
    rows, info = simple.build(out, cfg, p, qc, rf)
    simple.print_simple(rows, info, color=sys.stdout.isatty())
    if "--details" in sys.argv:
        print("\n\n----- DETAILS (for the curious) -----")
        print_report(out, cfg, p)
    out["simple_html"] = simple.html_simple(rows, info)
    write_dashboard(out, cfg, p)
    json.dump({k: v for k, v in out.items() if not k.startswith("_")}, open(os.path.join(HERE, "last_signal.json"), "w"), indent=1, default=str)
