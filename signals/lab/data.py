"""Data loading with on-disk cache. Free sources only: Yahoo (yfinance) + FRED."""
import os, time, io, warnings
import pandas as pd, numpy as np
import yfinance as yf
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data")
os.makedirs(CACHE, exist_ok=True)
try:  # keep yfinance's cache inside this project folder
    os.makedirs(os.path.join(CACHE, "yf_cache"), exist_ok=True)
    yf.set_tz_cache_location(os.path.join(CACHE, "yf_cache"))
except Exception:
    pass
MAX_AGE_H = 12  # refresh cache after this many hours


def _fresh(path):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < MAX_AGE_H * 3600


def ohlc(ticker, refresh=False):
    """Daily OHLCV, split/dividend adjusted (auto_adjust=True) + RawClose. Index tz-naive dates.
    Cache-safe: empty or truncated downloads never overwrite a good cache; zero closes dropped.
    NOTE crypto: Yahoo's bar for *today* (UTC) is incomplete -> use close_complete() for signals."""
    fn = os.path.join(CACHE, ticker.replace("^", "IDX_").replace("=", "_").replace("/", "_") + ".parquet")
    old = pd.read_parquet(fn) if os.path.exists(fn) else None
    if old is not None and len(old) == 0:
        old = None
    if not refresh and old is not None and _fresh(fn):
        return old
    d = None
    for attempt in range(3):
        try:
            d = yf.download(ticker, period="max", auto_adjust=True, progress=False, threads=False)
            if d is not None and len(d):
                break
        except Exception:
            pass
        time.sleep(2 + attempt * 3)
    if d is None or len(d) == 0:
        if old is not None:
            return old
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d[d["Close"] > 0]
    if old is not None and (d.index[0] > old.index[0] + pd.Timedelta(days=7) or len(d) < 0.9 * len(old)):
        warnings.warn(f"{ticker}: truncated Yahoo history ({len(d)} vs {len(old)} rows) - keeping cache")
        os.utime(fn)
        return old
    tmp = fn + f".{os.getpid()}.tmp"
    d.to_parquet(tmp); os.replace(tmp, fn)
    return d


def close(ticker, refresh=False):
    return ohlc(ticker, refresh)["Close"].rename(ticker)


def closes(tickers, refresh=False):
    return pd.concat([close(t, refresh) for t in tickers], axis=1)


def fred(series_id, refresh=False):
    fn = os.path.join(CACHE, f"FRED_{series_id}.parquet")
    if not refresh and _fresh(fn):
        return pd.read_parquet(fn)[series_id]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read().decode()
        s = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True, na_values=".")
        s.columns = [series_id]
        s.to_parquet(fn)
        return s[series_id]
    except Exception:
        if os.path.exists(fn):
            return pd.read_parquet(fn)[series_id]
        raise


def rf_daily(index, refresh=False):
    """Per-bar T-bill return aligned to `index`, no look-ahead (bar t accrues the yield known at t-1).
    DTB3 (daily, 1954+, bank-discount basis) spliced with TB3MS (monthly, 1934+); 1928-33 uses an
    approximate 3%/yr (flagged). Calendar-gap accrual, so works on 5-day, 7-day, weekly calendars."""
    d = fred("DTB3", refresh).dropna() / 100.0
    m = fred("TB3MS", refresh).dropna() / 100.0
    m.index = m.index + pd.offsets.MonthEnd(0)          # monthly avg known at month end
    m = m[m.index < d.index[0]]
    pre = pd.Series(0.03, index=pd.date_range("1920-01-01", m.index[0] - pd.Timedelta(days=1), freq="ME"))
    y = pd.concat([pre, m, d]).sort_index()
    y = y[~y.index.duplicated(keep="last")]
    growth_91d = 1 / (1 - y * 91 / 360)                 # discount quote -> 91-day growth factor
    g = growth_91d.reindex(growth_91d.index.union(index)).ffill().reindex(index)
    gaps = pd.Series(index, index=index).diff().dt.days.fillna(1)
    out = g.shift(1).fillna(g) ** (gaps / 91.0) - 1
    return out.fillna(0.0).rename("rf")


def close_complete(ticker, refresh=False):
    """Close series without a possibly-incomplete bar dated today (UTC) - use for crypto signals."""
    c = close(ticker, refresh)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    return c[c.index < today]


def shiller_div_yield():
    """Monthly S&P 500 dividend yield (D/P) from Shiller's data (data/shiller_ie_data.xls), 1871-2023."""
    x = pd.read_excel(os.path.join(CACHE, "shiller_ie_data.xls"), sheet_name="Data", skiprows=7)
    x = x[pd.to_numeric(x["Date"], errors="coerce").notna()]
    x = x[pd.to_numeric(x["D"], errors="coerce").notna()]
    yr = x["Date"].astype(float)
    idx = pd.to_datetime([f"{int(v)}-{int(round((v - int(v)) * 100)):02d}-01" for v in yr]) + pd.offsets.MonthEnd(0)
    return pd.Series(x["D"].astype(float).values / x["P"].astype(float).values, index=idx).rename("dy")


def sp500_tr_returns():
    """Daily S&P 500 TOTAL return from 1928: ^GSPC price + Shiller dividend yield accrual until 1988,
    then ^SP500TR. Use for long-history synthetic leverage (price-only proxies understate 3x by 9-15%/yr pre-1990)."""
    px = close("^GSPC")
    r = px.pct_change()
    dy = shiller_div_yield().reindex(shiller_div_yield().index.union(px.index)).ffill().reindex(px.index)
    gaps = pd.Series(px.index, index=px.index).diff().dt.days.fillna(1)
    r_tr = r + dy.fillna(dy.dropna().iloc[-1]) * gaps / 365.0
    tr = close("^SP500TR").pct_change().dropna()
    r_tr.loc[tr.index[0]:] = tr.reindex(r_tr.loc[tr.index[0]:].index)
    return r_tr.dropna().rename("SPX_TR")


def ndx_tr_returns(div_yield_pre1999=0.006):
    """Daily Nasdaq-100 TOTAL return from 1985: ^NDX price + ~0.6%/yr dividends until QQQ (1999-03-10),
    then QQQ adjusted (includes the 0.20% fee -> slightly conservative)."""
    px = close("^NDX")
    r = px.pct_change()
    gaps = pd.Series(px.index, index=px.index).diff().dt.days.fillna(1)
    r = r + div_yield_pre1999 * gaps / 365.0
    q = close("QQQ").pct_change().dropna()
    r.loc[q.index[0]:] = q.reindex(r.loc[q.index[0]:].index)
    return r.dropna().rename("NDX_TR")


def synthetic_leveraged(underlying_ret, L, rf, expense=0.0095, borrow_spread=0.006):
    """Daily-rebalanced leveraged ETF simulation (calendar-time fee accrual, any calendar).
    ret = L*r - (L-1)*(rf + spread) - expense.  Validated within ~0.3%/yr of TQQQ/QLD/UPRO/SSO/SOXL
    ONLY with TOTAL-RETURN underlyings (QQQ, SPY, SOXX, sp500_tr_returns, ndx_tr_returns)."""
    rf = rf.reindex(underlying_ret.index).fillna(0.0)
    gy = pd.Series(_gap_years(underlying_ret.index), index=underlying_ret.index)
    return (L * underlying_ret - (L - 1) * (rf + borrow_spread * gy) - expense * gy).rename(f"{L}x")


def _gap_years(idx):
    return pd.Series(idx, index=idx).diff().dt.days.fillna(1).values / 365.0


PRICE_INDEXES = {"^GSPC", "^NDX", "^SOX", "^IXIC", "^RUT", "^GDAXI", "^N225", "^DJI"}


def spliced_leveraged(proxy_ret, actual_ticker, L, rf, **kw):
    """Real leveraged ETF returns where available, synthetic (from proxy_ret) before inception.
    ETF returns are computed on the proxy calendar from forward-filled closes (no lost returns)."""
    if getattr(proxy_ret, "name", None) in PRICE_INDEXES:
        warnings.warn(f"spliced_leveraged: {proxy_ret.name} is a PRICE index - use a total-return proxy")
    syn = synthetic_leveraged(proxy_ret, L, rf, **kw)
    px = close(actual_ticker)
    px = px.reindex(px.index.union(syn.index)).ffill().reindex(syn.index)
    act = px.pct_change()
    out = syn.copy()
    m = act.notna() & px.notna()
    m &= m.index > px.first_valid_index()
    out[m] = act[m]
    return out.rename(actual_ticker)
