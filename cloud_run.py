"""GitHub Actions entry point for the Edge Stack paper bot (see .github/workflows/bot.yml).

  python cloud_run.py --check [--force]   prints go=1 if today's trading window is open or opens within 25 minutes
  python cloud_run.py --wait              waits up to 6 minutes for IB Gateway (docker) to log in and accept the API
  python cloud_run.py                     runs bot.main() every 4 minutes inside the window until the day is done,
                                          has failed 3 times, or the window closes (same rules as the Mac launchd job)
  python cloud_run.py --test              one run now with no orders sent (state and logs go to a temp folder, so the
                                          real run later that day is not blocked), plus a what-if order to prove that
                                          orders would be accepted"""
import os, sys, time, json, tempfile, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bot

EARLY = 25          # minutes before the window opens that a run may start (GitHub's schedule is often a few minutes late)


def minutes(now):
    return now.hour * 60 + now.minute


def check(force):
    now = dt.datetime.now(bot.NY)
    w = bot.trade_window(now)
    go = force or (w is not None and w[0] - EARLY <= minutes(now) < w[1])
    print(f"go={1 if go else 0}")
    print(f"New York time {now:%Y-%m-%d %H:%M}, window {w}", file=sys.stderr)


def wait():
    from ib_async import IB
    t0 = time.time()
    while time.time() - t0 < 360:
        ib = IB()
        try:
            ib.connect(bot.CFG["host"], int(bot.CFG["port"]), clientId=99, timeout=15)
            print(f"IB Gateway ready after {time.time() - t0:.0f}s, accounts {ib.managedAccounts()}")
            ib.disconnect()
            return 0
        except Exception:
            ib.disconnect()
            time.sleep(10)
    print("IB Gateway did not log in within 6 minutes (wrong paper login, or IBKR refused the login)")
    return 1


def run_window():
    while True:
        now = dt.datetime.now(bot.NY)
        w = bot.trade_window(now)
        m = minutes(now)
        if w is None or m >= w[1]:
            break
        if m >= w[0]:
            sys.argv = ["bot.py"]
            bot.main()
            day = bot.load_state().get("days", {}).get(str(now.date()), {})
            print(f"{now:%H:%M} NY: {day.get('status')}")
            if day.get("status") in ("done", "dry_run") or day.get("attempts", 0) >= bot.MAX_ATTEMPTS:
                break
            time.sleep(240)
        else:
            time.sleep(20)


def test():
    tmp = tempfile.mkdtemp()
    bot.LOGD, bot.STATE_F, bot.HERE = tmp, os.path.join(tmp, "state.json"), tmp
    sys.argv = ["bot.py", "--now", "--dry-run"]
    bot.main()
    print(open(os.path.join(tmp, "bot.log")).read())
    from ib_async import MarketOrder
    ib, acct = bot.connect()
    c = bot.front_contract(ib, "MNQ", int(bot.CFG["roll_days"]))
    st = ib.whatIfOrder(c, MarketOrder("BUY", 1, account=acct, tif="DAY"))
    ok = bool(st and st.initMarginChange not in ("", None))
    print(f"what-if order 1 {c.localSymbol}: {'accepted, margin ' + str(st.initMarginChange) if ok else 'NO ANSWER'}")
    ib.disconnect()
    return 0 if ok else 1


def summary():
    """Markdown for the GitHub run page: today's status and the latest days."""
    st = bot.load_state().get("last", {})
    out = [f"**Status:** {st.get('status', 'no run')} · signal date {st.get('date', '-')} · targets {st.get('targets', {})}"]
    for o in st.get("orders", []):
        out.append(f"- {o['action']} {o['qty']} {o['symbol']} {o['expiry']}: {o['status']} at {o['avg_price']} ({o['why']})")
    fn = os.path.join(bot.LOGD, "performance.csv")
    if os.path.exists(fn):
        import csv
        rows = list(csv.reader(open(fn)))
        out += ["", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        out += ["| " + " | ".join(r) + " |" for r in rows[1:][-10:]]
    print("\n".join(out))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--check" in a:
        check("--force" in a)
    elif "--wait" in a:
        sys.exit(wait())
    elif "--test" in a:
        sys.exit(test())
    elif "--summary" in a:
        summary()
    else:
        run_window()
