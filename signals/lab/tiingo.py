"""Tiingo daily prices (incl. many DELISTED US stocks -> reduces survivorship bias).

Free-plan limits: 50 requests/hour, 1000/day. This module enforces <=45/h and <=900/day across
all processes via a shared request log, and caches every response forever in data/tiingo/.
Key is read from trading-lab/.env (TIINGO_KEY=...). Never print or log the key.

    from lab import tiingo
    tiingo.cached('YHOO')            # -> DataFrame or None (no network)
    tiingo.prices('YHOO')            # network if not cached (blocks/waits if over the rate limit when wait=True)
    tiingo.budget()                  # remaining requests this hour / today
    tiingo.queue_add(['YHOO','CELG'])  # add to the background fetch queue (see tool/tiingo_worker.py)
"""
import os, json, time, fcntl, urllib.request, urllib.error
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "tiingo")
os.makedirs(DIR, exist_ok=True)
LOG = os.path.join(DIR, "_requests.log")
LOCK = os.path.join(DIR, "_lock")
QUEUE = os.path.join(DIR, "_queue.txt")
MISSING = os.path.join(DIR, "_missing.txt")
SYMS = os.path.join(DIR, "_symbols.log")
PER_HOUR, PER_DAY = 45, 900
MONTH_SYMBOLS, MONTH_BYTES = 450, 1_500_000_000   # plan: 500 unique symbols/month, 2 GB/month


def _key():
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if line.startswith("TIINGO_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("TIINGO_KEY missing in .env")


def _fn(t):
    return os.path.join(DIR, t.upper().replace("/", "_").replace(".", "-") + ".parquet")


def _stamps():
    if not os.path.exists(LOG):
        return []
    now = time.time(); out = []
    with open(LOG) as f:
        for x in f.read().split():
            try:
                v = float(x)
            except ValueError:
                v = now  # corrupt line counts as a fresh request (conservative)
            if now - v < 86400:
                out.append(v)
    return out


def month_usage():
    """Unique symbols and bytes requested in the last 31 days (>= any calendar month)."""
    now, syms, by = time.time(), set(), 0
    if os.path.exists(SYMS):
        for ln in open(SYMS):
            p = ln.split()
            try:
                if len(p) == 3 and now - float(p[0]) < 31 * 86400:
                    syms.add(p[1]); by += int(p[2])
            except ValueError:
                continue
    return syms, by


def budget():
    s = _stamps(); now = time.time(); syms, by = month_usage()
    return {"hour_left": PER_HOUR - sum(1 for x in s if now - x < 3600), "day_left": PER_DAY - len(s),
            "month_symbols_left": MONTH_SYMBOLS - len(syms), "month_MB_left": int((MONTH_BYTES - by) / 1e6)}


def cached(t):
    fn = _fn(t)
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    return None


def is_missing(t):
    return os.path.exists(MISSING) and t.upper() in open(MISSING).read().split()


def prices(t, start="1900-01-01", wait=False):
    """Adjusted daily OHLCV for ticker t (Tiingo naming, e.g. BRK-B). Returns None if unavailable.
    Always fetches full history (cached forever); returns rows from `start`."""
    t = t.upper()
    c = cached(t)
    if c is not None:
        return c.loc[start:]
    if is_missing(t):
        return None
    while True:
        with open(LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            c = cached(t)                      # another process may have fetched it meanwhile
            if c is not None:
                return c.loc[start:]
            syms, by = month_usage()
            if (t not in syms and len(syms) >= MONTH_SYMBOLS) or by >= MONTH_BYTES:
                raise RuntimeError(f"Tiingo MONTHLY budget exhausted (unique symbols {len(syms)}, bytes {by})")
            b = budget()
            if b["hour_left"] > 0 and b["day_left"] > 0:
                with open(LOG, "a") as f:
                    f.write(f"{time.time()}\n")
                with open(SYMS, "a") as f:
                    f.write(f"{time.time()} {t} 2000000\n")   # provisional size, corrected below
                break
        if not wait:
            raise RuntimeError(f"Tiingo rate budget exhausted: {b}")
        time.sleep(60)
    url = f"https://api.tiingo.com/tiingo/daily/{t}/prices?startDate=1900-01-01&resampleFreq=daily"
    req = urllib.request.Request(url, headers={"Authorization": f"Token {_key()}", "Content-Type": "application/json"})
    try:
        body_bytes = urllib.request.urlopen(req, timeout=60).read()
        with open(SYMS, "a") as f:
            f.write(f"{time.time()} {t} {max(0, len(body_bytes) - 2000000)}\n")
        raw = json.loads(body_bytes.decode())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:300]
        except Exception:
            pass
        if e.code == 404 and "not found" in body.lower():
            _mark_missing(t)
            return None
        # 429 / 5xx / anything else: do NOT cache, do NOT mark missing -> caller retries later
        raise RuntimeError(f"Tiingo HTTP {e.code} for {t}: {body[:120]}")
    if isinstance(raw, dict):  # error payloads come back as {"detail": "..."}
        msg = str(raw.get("detail", raw))
        if "not found" in msg.lower():
            _mark_missing(t)
            return None
        raise RuntimeError(f"Tiingo error for {t}: {msg[:120]}")
    if not isinstance(raw, list):
        raise RuntimeError(f"Tiingo unexpected payload for {t}")
    if len(raw) == 0:
        _mark_missing(t)
        return None
    d = pd.DataFrame(raw)
    need = {"date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume", "close"}
    if not need.issubset(d.columns):
        raise RuntimeError(f"Tiingo payload missing columns for {t}: {need - set(d.columns)}")
    d.index = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    d = d.rename(columns={"adjOpen": "Open", "adjHigh": "High", "adjLow": "Low", "adjClose": "Close",
                          "adjVolume": "Volume", "close": "RawClose"})
    d = d[["Open", "High", "Low", "Close", "Volume", "RawClose"]].sort_index().dropna(subset=["Close"])
    d = d[~d.index.duplicated(keep="last")]
    if d["Close"].isna().all() or (d["Close"] <= 0).any():
        raise RuntimeError(f"Tiingo bad prices for {t}")
    tmp = _fn(t) + f".{os.getpid()}.tmp"
    d.to_parquet(tmp)
    os.replace(tmp, _fn(t))  # atomic: readers never see a half-written file
    return d.loc[start:]


def _mark_missing(t):
    with open(MISSING, "a") as f:
        f.write(t.upper() + "\n")


def clean(d):
    """Drop stale rows after a stock stopped trading (Tiingo pads delisted names with flat, zero-volume
    bars). Returns the frame truncated at the last bar with volume > 0."""
    if d is None or len(d) == 0:
        return d
    traded = d.index[d["Volume"] > 0]
    return d.loc[: traded[-1]] if len(traded) else d.iloc[:0]


def queue_add(tickers):
    have = set(open(QUEUE).read().split()) if os.path.exists(QUEUE) else set()
    new = [t.upper() for t in tickers if t.upper() not in have and cached(t) is None and not is_missing(t)]
    with open(QUEUE, "a") as f:
        for t in new:
            f.write(t + "\n")
    return len(new)


def queue_pending():
    if not os.path.exists(QUEUE):
        return []
    return [t for t in dict.fromkeys(open(QUEUE).read().split()) if cached(t) is None and not is_missing(t)]


def supported():
    """Tiingo master list (ticker, exchange, assetType, priceCurrency, startDate, endDate)."""
    return pd.read_csv(os.path.join(ROOT, "data", "tiingo_supported_tickers.csv"))
