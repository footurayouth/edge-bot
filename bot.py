"""EDGE STACK - automated IBKR PAPER trading bot (micro-futures route; preset from config.json, default conservative).

What it trades (same rules and code as the Edge Stack tool, futures route - signals/tool/today.py + edge.py):
  - Nasdaq trend  -> MNQ micro futures, notional = L x trend score
  - IBS dip-buys  -> MES / MNQ / M2K / MYM (S&P 500, Nasdaq-100, Russell 2000, Dow); exit at the first close above the
                     prior day's high or after 10 days
  - month-end     -> MES for one day
When: once per US trading day, 16:20-16:55 New York time (13:05-13:14 on half days), right after the close, as the
backtest assumes. It computes today's target contracts from the final daily bars and moves the paper account to them
with market orders, rolling to the next quarterly contract `roll_days` before expiry.
Safety: trades ONLY if every account the API reports is a paper account (IDs start with 'D'); never trades on data that
is not final; aborts if a target exceeds max_contracts; dry_run=true logs the orders without sending them.
launchd runs this every 5 minutes (Install autopilot.command); outside the window it exits at once.
Manual: .venv/bin/python bot.py --now [--dry-run]   (--now ignores the time window, for testing)"""
import os, sys, json, csv, time, argparse, logging, datetime as dt, html as H
from zoneinfo import ZoneInfo
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "signals", "tool")); sys.path.insert(0, os.path.join(HERE, "signals"))
NY = ZoneInfo("America/New_York")
CFG = json.load(open(os.path.join(HERE, "config.json")))
LOGD = os.path.join(HERE, "logs"); os.makedirs(LOGD, exist_ok=True)
STATE_F = os.path.join(HERE, "state.json")
FUT = {"SPY": ("MES", "CME"), "QQQ": ("MNQ", "CME"), "IWM": ("M2K", "CME"), "DIA": ("MYM", "CBOT")}
EXCH = {s: x for s, x in FUT.values()}
MAX_ATTEMPTS = 3
log = logging.getLogger("edge-bot")


def setup_logging(verbose):
    if log.handlers:                                   # main() can run several times in one process (cloud_run.py)
        return
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(LOGD, "bot.log")); fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(logging.Formatter("%(message)s")); log.addHandler(sh)


def load_state():
    try:
        return json.load(open(STATE_F))
    except Exception:
        return {}


def save_state(s):
    tmp = STATE_F + ".tmp"; json.dump(s, open(tmp, "w"), indent=1, default=str); os.replace(tmp, STATE_F)


def trade_window(now):
    """(start, end) minutes after midnight New York for today's trading window, or None if the market is closed."""
    import pandas as pd, edge
    d = now.date()
    if now.weekday() >= 5 or pd.Timestamp(d) in edge.nyse_holidays(d.year):
        return None
    return (13 * 60 + 5, 13 * 60 + 14) if str(d) in edge.HALF_DAYS else (16 * 60 + 20, 16 * 60 + 55)


def compute_targets(capital):
    """Target contracts per micro future from today's final bars (same function the click tool uses)."""
    import today as T
    T.CUT = T.us_cutoff()                          # before 16:15 NY: use yesterday's bar (only happens with --now)
    p = T.PRESETS[CFG["preset"]]
    out = {}
    T.futures_orders({"risk": CFG["preset"], "futures_account_usd": capital, "route": "ibkr_futures"}, p, out)
    f = out["futures"]
    alerts = [a for a in [T.freshness_warning(f["date"])] if a] + out.get("alerts", [])
    tgt = {r["future"]: int(r["contracts"]) for r in f["rows"]}
    exact = {r["future"]: float(r["contracts_exact"]) for r in f["rows"]}
    return f["date"], tgt, exact, alerts, f


def plan_orders(targets, held, fronts):
    """targets {sym: n}; held {sym: [(contract, qty)]}; fronts {sym: contract to hold}. Returns [(contract, action, qty, why)].
    Positions in any other expiry of the same symbol are closed (roll), the rest is moved to the target."""
    orders = []
    for sym, tgt in targets.items():
        f = fronts[sym]; cur = 0
        for c, q in held.get(sym, []):
            if c.conId == f.conId:
                cur += q
            elif q:
                orders.append((c, "SELL" if q > 0 else "BUY", abs(q), f"roll: close {c.lastTradeDateOrContractMonth}"))
        d = tgt - cur
        if d:
            orders.append((f, "BUY" if d > 0 else "SELL", abs(d), f"target {tgt}, held {cur}"))
    return orders


# ---------------------------------------------------------------- IBKR
class NotOpen(Exception):
    """Neither TWS nor IB Gateway is accepting connections (app closed or not logged in)."""


def connect():
    from ib_async import IB
    ib = IB()
    for port in dict.fromkeys([int(CFG["port"]), 7497, 4002]):     # TWS paper 7497, IB Gateway paper 4002
        try:
            ib.connect(CFG["host"], port, clientId=int(CFG["client_id"]), timeout=20)
            break
        except (ConnectionRefusedError, OSError, TimeoutError):
            ib.disconnect()
    else:
        raise NotOpen("TWS / IB Gateway not open or not logged in (tried ports 7497 and 4002)")
    ib.RequestTimeout = 60                                   # no request may hang the run
    accts = ib.managedAccounts()
    if not accts or not all(a.startswith("D") for a in accts):
        ib.disconnect()
        raise RuntimeError(f"REFUSED: the API reports accounts {accts}; this bot only trades paper accounts (IDs start with 'D'). "
                           "Log TWS / IB Gateway in with your PAPER login.")
    acct = CFG.get("account") or accts[0]
    if acct not in accts:
        ib.disconnect(); raise RuntimeError(f"account {acct} not in {accts}")
    return ib, acct


def equity_usd(ib, acct):
    # values arrive with the connection (one-account login); re-requesting them can block forever, so read what is there
    vals = {(v.tag, v.currency): v.value for v in ib.accountValues(acct)}
    if not any(t == "NetLiquidation" for t, _ in vals):
        vals.update({(v.tag, v.currency): v.value for v in ib.accountSummary(acct)})
    base = next((c for (t, c) in vals if t == "NetLiquidation" and c not in ("", "BASE")), None)
    if base is None:
        raise RuntimeError("no NetLiquidation value from IBKR")
    nl = float(vals[("NetLiquidation", base)])
    if base == "USD":
        return nl, nl, base
    rate = float(vals.get(("ExchangeRate", "USD"), "nan"))       # base-currency units per 1 USD
    if not rate == rate or rate <= 0:                             # IBKR only sends it while the account holds USD
        rate = 1 / usd_per(ib, base)
    return nl / rate, nl, base


def usd_per(ib, base):
    """USD per 1 unit of the base currency (Yahoo; the paper account has no IBKR forex data)."""
    import yfinance as yf
    px = yf.Ticker(f"{base}USD=X").history(period="5d")["Close"].dropna()
    if not len(px):
        raise RuntimeError(f"no {base}USD exchange rate")
    return float(px.iloc[-1])


def front_contract(ib, sym, roll_days):
    from ib_async import Future
    cds = ib.reqContractDetails(Future(symbol=sym, exchange=EXCH[sym], currency="USD"))
    today = dt.datetime.now(NY).date()
    for c in sorted({cd.contract.conId: cd.contract for cd in cds}.values(), key=lambda c: c.lastTradeDateOrContractMonth):
        ltd = dt.datetime.strptime(c.lastTradeDateOrContractMonth[:8], "%Y%m%d").date()
        if (ltd - today).days > roll_days:
            return c
    raise RuntimeError(f"no {sym} contract found")


def held_positions(ib, acct):
    held = {}
    for p in ib.positions(acct):
        c = p.contract
        if c.secType == "FUT" and c.symbol in EXCH and p.position:
            c.exchange = c.exchange or EXCH[c.symbol]
            held.setdefault(c.symbol, []).append((c, int(round(p.position))))
    return held


def cancel_open(ib):
    """Cancel this bot's still-open orders on our symbols, so a retry can never double up."""
    n = 0
    for t in ib.openTrades():
        if t.contract.symbol in EXCH and not t.isDone():
            ib.cancelOrder(t.order); n += 1
    if n:
        ib.sleep(3)
    return n


def execute(ib, acct, orders):
    from ib_async import MarketOrder
    trades = []
    for c, act, q, why in orders:
        t = ib.placeOrder(c, MarketOrder(act, q, account=acct, tif="DAY"))
        trades.append((t, why))
    t0 = time.time()
    while time.time() - t0 < 120 and not all(t.isDone() for t, _ in trades):
        ib.sleep(1)
    ib.sleep(3)                                              # IBKR sends commission / realized P&L just after the fill
    ok = lambda x: x is not None and abs(x) < 1e9            # IBKR marks "not set" with a huge number
    res = []
    for t, why in trades:
        s = t.orderStatus
        cr = [f.commissionReport for f in t.fills if f.commissionReport]
        res.append({"symbol": t.contract.symbol, "expiry": t.contract.lastTradeDateOrContractMonth, "action": t.order.action,
                    "qty": t.order.totalQuantity, "status": s.status, "filled": s.filled, "avg_price": s.avgFillPrice,
                    "commission_usd": round(sum(c.commission for c in cr if ok(c.commission)), 2),
                    "realized_pnl_usd": round(sum(c.realizedPNL for c in cr if ok(c.realizedPNL)), 2), "why": why})
    return res


# ---------------------------------------------------------------- records
def append_csv(name, row):
    fn = os.path.join(LOGD, name); new = not os.path.exists(fn)
    with open(fn, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader() if new else None; w.writerow(row)


def log_day(date, res, after, eq, spy):
    """performance.csv: one row per trading day (trade or hold) - what it did, what it holds, equity and P&L."""
    fn = os.path.join(LOGD, "performance.csv")
    prev = list(csv.DictReader(open(fn))) if os.path.exists(fn) else []
    if prev and prev[-1]["date"] == date:
        return
    eq0, spy0 = (float(prev[0]["equity_usd"]), float(prev[0]["spy_close"])) if prev else (eq, spy)
    last = float(prev[-1]["equity_usd"]) if prev else eq
    did = [r for r in res if r.get("filled")]
    append_csv("performance.csv", {
        "date": date,
        "action": "; ".join(f"{r['action']} {r['filled']:g} {r['symbol']} {r['expiry']}" for r in did) or "HOLD",
        "holding": ", ".join(f"{n:+d} {s}" for s, n in sorted(after.items()) if n) or "nothing (cash)",
        "equity_usd": round(eq, 2), "day_pnl_usd": round(eq - last, 2), "total_pnl_usd": round(eq - eq0, 2),
        "total_pct": round(100 * (eq / eq0 - 1), 2), "spy_close": round(spy, 2), "spy_total_pct": round(100 * (spy / spy0 - 1), 2),
        "realized_pnl_usd": round(sum(float(r.get("realized_pnl_usd") or 0) for r in did), 2),
        "commissions_usd": round(sum(float(r.get("commission_usd") or 0) for r in did), 2)})


def write_status(st):
    """status.html: today's signal, targets vs positions, orders, and the paper account vs the S&P 500 since start."""
    rd = lambda f: list(csv.DictReader(open(os.path.join(LOGD, f)))) if os.path.exists(os.path.join(LOGD, f)) else []
    rows, fills = rd("performance.csv"), rd("journal.csv")
    perf = ""
    if rows:
        a, b = rows[0], rows[-1]
        eqs = [float(r["equity_usd"]) for r in rows]
        peak, mdd = eqs[0], 0.0
        for e in eqs:
            peak = max(peak, e); mdd = min(mdd, e / peak - 1)
        real = [float(f.get("realized_pnl_usd") or 0) for f in fills]
        won, lost = sum(x > 0 for x in real), sum(x < 0 for x in real)
        perf = (f"<p><b>Since {a['date']}: ${float(b['equity_usd']):,.0f} = {float(b['total_pnl_usd']):+,.0f} USD "
                f"({float(b['total_pct']):+.2f}%) vs S&amp;P 500 {float(b['spy_total_pct']):+.2f}%</b></p>"
                f"<p class=mut>{len(rows)} trading days · worst drop {mdd:.1%} · {len(fills)} fills · closed trades: {won} won, {lost} lost, "
                f"realized ${sum(real):+,.0f} · commissions ${sum(float(f.get('commission_usd') or 0) for f in fills):,.0f}</p>")
    d = st.get("last", {})
    fut = d.get("signal", {})
    tr = "".join(f"<tr><td>{s}</td><td>{d.get('targets', {}).get(s, '')}</td><td>{d.get('exact', {}).get(s, '')}</td><td>{d.get('held_before', {}).get(s, 0)}</td></tr>"
                 for s in ("MES", "MNQ", "M2K", "MYM"))
    od = "".join(f"<tr><td>{o['action']}</td><td>{o['qty']}</td><td>{o['symbol']} {o['expiry']}</td><td>{o['status']}</td><td>{o['avg_price']}</td><td>{H.escape(o['why'])}</td></tr>"
                 for o in d.get("orders", [])) or "<tr><td colspan=6>no orders</td></tr>"
    eq = "".join(f"<tr><td>{r['date']}</td><td>{H.escape(r['action'])}</td><td>{H.escape(r['holding'])}</td><td>${float(r['equity_usd']):,.0f}</td>"
                 f"<td>{float(r['day_pnl_usd']):+,.0f}</td><td>{float(r['total_pct']):+.2f}%</td><td>{float(r['spy_total_pct']):+.2f}%</td></tr>"
                 for r in rows[::-1])
    fl = "".join(f"<tr><td>{f['time']}</td><td>{f['action']} {f['qty']}</td><td>{f['symbol']} {f['expiry']}</td><td>{f['avg_price']}</td>"
                 f"<td>{f.get('realized_pnl_usd', '')}</td><td>{f.get('commission_usd', '')}</td><td>{H.escape(f['why'])}</td></tr>"
                 for f in fills[::-1][:300])
    al = "".join(f"<li>{H.escape(a)}</li>" for a in d.get("alerts", []))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge Stack paper bot</title><style>body{{font:15px -apple-system,Helvetica,Arial;margin:0;background:#f7f7f5;color:#1d1d1b}}
main{{max-width:900px;margin:auto;padding:16px}}.card{{background:#fff;border:1px solid #e3e3de;border-radius:12px;padding:14px 18px;margin:12px 0}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #eee;padding:6px;text-align:left}}.mut{{color:#6b6b66}}</style></head><body><main>
<h1>Edge Stack - IBKR paper bot</h1><p class="mut">preset {CFG['preset']} · {'DRY RUN (no orders sent)' if CFG.get('dry_run') else 'live paper orders'} · last run {st.get('last_run', '-')} · status: <b>{H.escape(str(d.get('status', '-')))}</b></p>
<div class="card">{perf}<p>Signal date {d.get('date', '-')} · Nasdaq trend score {fut.get('trend', {}).get('f', '-')} · S&amp;P above 200-day: {fut.get('ibs', {}).get('regime_on', '-')} ·
IBS positions held: {H.escape(str(fut.get('ibs', {}).get('held', {})))} · month-end: {H.escape(str(fut.get('monthend', {})))}</p>
<p>Account used for sizing: ${d.get('capital_usd', 0):,.0f}</p>{'<ul>' + al + '</ul>' if al else ''}</div>
<div class="card"><h2>Targets (contracts)</h2><table><tr><th>Future</th><th>Target</th><th>Exact</th><th>Held before</th></tr>{tr}</table></div>
<div class="card"><h2>Orders this run</h2><table><tr><th>Side</th><th>Qty</th><th>Contract</th><th>Status</th><th>Fill</th><th>Why</th></tr>{od}</table></div>
<div class="card"><h2>Every day (newest first)</h2><table><tr><th>Date</th><th>Did</th><th>Holding after</th><th>Equity</th><th>Day P&amp;L $</th><th>Total</th><th>S&amp;P total</th></tr>{eq}</table></div>
<div class="card"><h2>Every fill (newest first)</h2><p class=mut>Realized P&amp;L = profit or loss IBKR books when a position is closed or reduced.</p><table><tr><th>Time NY</th><th>Side</th><th>Contract</th><th>Price</th><th>Realized P&amp;L $</th><th>Commission $</th><th>Why</th></tr>{fl}</table></div>
</main></body></html>"""
    open(os.path.join(HERE, "status.html"), "w").write(page)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="ignore the time window (testing)")
    ap.add_argument("--dry-run", action="store_true", help="compute and log orders, send nothing")
    ap.add_argument("--capital", type=float, help="account size in USD for sizing (overrides IBKR / config)")
    a = ap.parse_args()
    setup_logging(verbose=a.now)
    dry = a.dry_run or bool(CFG.get("dry_run"))
    now = dt.datetime.now(NY)
    st = load_state()
    if not a.now:
        w = trade_window(now)
        m = now.hour * 60 + now.minute
        if w is None or not (w[0] <= m < w[1]):
            return
        day = st.get("days", {}).get(str(now.date()), {})
        if day.get("status") in ("done", "dry_run") or day.get("attempts", 0) >= MAX_ATTEMPTS:
            return
    key = str(now.date())
    rec = st.setdefault("days", {}).setdefault(key, {"attempts": 0})      # attempts = failed runs (waiting for data is free)
    st["last_run"] = now.strftime("%Y-%m-%d %H:%M NY")
    ib = acct = None
    try:
        try:
            ib, acct = connect()
            cap, nl, base = equity_usd(ib, acct)
            log.info(f"connected: paper account {acct}, net liquidation {nl:,.0f} {base} = ${cap:,.0f}")
        except NotOpen as e:
            if not dry:                                       # not a failure: keep trying every 5 minutes in the window
                rec["status"] = "waiting: open IB Gateway (or TWS) and log in"; log.warning(str(e))
                return
            log.warning(f"IBKR not reachable ({e}) - dry run continues with capital_usd from config")
            cap = nl = None; base = None
        except Exception as e:
            if not dry:
                raise
            log.warning(f"IBKR not reachable ({e}) - dry run continues with capital_usd from config")
            cap = nl = None; base = None
        if a.capital or CFG.get("capital_usd"):
            cap = float(a.capital or CFG["capital_usd"])
        if not cap:
            raise RuntimeError("no account size: IBKR not connected and capital_usd not set")
        date, tgt, exact, alerts, fut = compute_targets(cap)
        rec.update(date=date, targets=tgt, exact={k: round(v, 2) for k, v in exact.items()}, alerts=alerts, capital_usd=cap,
                   signal={k: fut[k] for k in ("trend", "ibs", "monthend", "gross_x", "gross_rounded_x")})
        log.info(f"signal date {date}; targets {tgt} (exact {rec['exact']}); trend {fut['trend']['f']}; alerts {alerts}")
        if any(x.startswith(("STOP", "LATE", "WARNING")) for x in alerts) and not dry:
            rec["status"] = "waiting (data not final)"; log.warning("data not final - will retry in 5 minutes")
            return
        bad = {s: n for s, n in tgt.items() if abs(n) > int(CFG["max_contracts"].get(s, 0))}
        if bad:
            raise RuntimeError(f"targets {bad} exceed max_contracts {CFG['max_contracts']} - nothing sent")
        if ib is None:
            rec.update(status="dry_run", orders=[{"symbol": s, "expiry": "front", "action": "TARGET", "qty": n, "status": "not sent",
                                                  "avg_price": "", "why": "IBKR not connected"} for s, n in tgt.items()])
            return
        fronts = {s: front_contract(ib, s, int(CFG["roll_days"])) for s in tgt}
        cancelled = cancel_open(ib)
        held = held_positions(ib, acct)
        rec["held_before"] = {s: sum(q for _, q in v) for s, v in held.items()}
        orders = plan_orders(tgt, held, fronts)
        log.info(f"held {rec['held_before']}; cancelled {cancelled} open orders; orders: "
                 + ("; ".join(f"{act} {q} {c.symbol} {c.lastTradeDateOrContractMonth} ({why})" for c, act, q, why in orders) or "none"))
        if dry:
            rec.update(status="dry_run", orders=[{"symbol": c.symbol, "expiry": c.lastTradeDateOrContractMonth, "action": act, "qty": q,
                                                  "status": "not sent (dry run)", "avg_price": "", "why": why} for c, act, q, why in orders])
            return
        res = execute(ib, acct, orders)
        rec["orders"] = res
        for r in res:
            append_csv("journal.csv", {"time": now.strftime("%Y-%m-%d %H:%M"), **r})
        held2 = held_positions(ib, acct)
        after = {s: sum(q for _, q in v) for s, v in held2.items()}
        stale = {s: [c.lastTradeDateOrContractMonth for c, q in v if q and c.conId != fronts[s].conId] for s, v in held2.items() if s in fronts}
        ok = all(after.get(s, 0) == n for s, n in tgt.items()) and not any(stale.values())   # right size AND all in the front month
        rec["held_after"] = after
        rec["status"] = "done" if ok else f"MISMATCH after orders: held {after} vs target {tgt}"
        log.info(f"after: {after} -> {rec['status']}")
        if ok:
            import today as T
            ib.sleep(2)
            cap2, _, _ = equity_usd(ib, acct)
            log_day(date, res, after, cap2, float(T.D.close("SPY").iloc[-1]))
        else:
            rec["attempts"] += 1
    except Exception as e:
        rec["status"] = f"ERROR: {e}"; rec["attempts"] += 1; log.exception("run failed")
    finally:
        st["last"] = {**rec}
        save_state(st)
        write_status(st)
        if ib is not None:
            ib.disconnect()


if __name__ == "__main__":
    main()
