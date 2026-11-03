"""EDGE STACK - final strategy core (used by the daily tool AND by the backtests, so they can't diverge).

Three independent sleeves (each researched, then independently re-implemented and audited by a critic):

  1. NASDAQ TREND   : leveraged Nasdaq-100 exposure only in calm up-trends.
                      35-cell ensemble: SMA n in {100..250 step 25} x 20d-realised-vol threshold {25..35%}.
                      f = share of cells where (NDX close > SMA n) and (rv20 < thr).  Exposure = L_TREND x f.
  2. IBS SWING      : buy US index dips that close near the day's low, sell on the first close above the
                      prior day's high (max 10 days). Signals SPY/QQQ/IWM/DIA; 2 slots; per-slot exposure
                      SLOT_ON x equity when SPY > SMA200 else SLOT_OFF. Executed right AFTER the US close (futures).
  2b. MONTH-END    : on the second-to-last NYSE trading day, s = month-to-date SPY minus month-to-date TLT total
                      return; hold MES at -sign(s) x slot size for one day (flat at the last day's close).
                      Uses only gross room left under the cap (after trend, then IBS). (Harvey-Mazzoleni-Melone 2025)
  3. CRYPTO TREND   : BTC and ETH, each 0.5 x trend x min(1, 40%/vol30) where trend = share of SMA
                      {20,50,100,150,200} below the UTC close.  Daily, rebalance when drift > 10%.

All signals use only data up to the decision close (no look-ahead)."""
import numpy as np, pandas as pd

# ---------------- parameters (frozen; changing them = new backtest required) ----------------
TREND_SMAS = (100, 125, 150, 175, 200, 225, 250)
TREND_VOLS = (0.25, 0.275, 0.30, 0.325, 0.35)
FAST_SMAS = (20, 50)                 # faster re-entry cells: f = max(f35, 0.5 x share of these 10 cells)
IBS_UNIVERSE = ("SPY", "QQQ", "IWM", "DIA")
IBS_ENTRY, IBS_MAXHOLD, IBS_SLOTS = 0.10, 10, 2
SLOT_ON, SLOT_OFF = 1.5, 0.5          # x equity notional per slot (= 50% / 16.7% of equity in a 3x ETF)
CRYPTO_SMAS = (20, 50, 100, 150, 200)
CRYPTO_VT, CRYPTO_VOLWIN = 0.40, 30

FUTURES = {"SPY": ("MES", "Micro E-mini S&P 500", 5.0, "^GSPC"),
           "QQQ": ("MNQ", "Micro E-mini Nasdaq-100", 2.0, "^NDX"),
           "IWM": ("M2K", "Micro E-mini Russell 2000", 5.0, "^RUT"),
           "DIA": ("MYM", "Micro E-mini Dow", 0.5, "^DJI")}
ETF3X = {"SPY": "UPRO", "QQQ": "TQQQ", "IWM": "TNA", "DIA": "UDOW"}
HALF_DAYS = {"2026-11-27", "2026-12-24", "2027-11-26"}   # NYSE 13:00 closes (CME 13:15 ET) - check nyse.com yearly


# ---------------- sleeve 1 ----------------
def trend_fraction(ndx_close):
    """Share (0..1) of the 35 ensemble cells that are 'on' at each close."""
    c = ndx_close.dropna()
    rv = c.pct_change().rolling(20).std() * np.sqrt(252)
    votes = []
    for n in TREND_SMAS:
        above = c > c.rolling(n).mean()
        for thr in TREND_VOLS:
            votes.append((above & (rv < thr)).astype(float))
    f = sum(votes) / len(votes)
    fast = []
    for n in FAST_SMAS:
        above = c > c.rolling(n).mean()
        for thr in TREND_VOLS:
            fast.append((above & (rv < thr)).astype(float))
    f = np.maximum(f, 0.5 * sum(fast) / len(fast))
    f[c.rolling(max(TREND_SMAS)).mean().isna()] = np.nan
    return f.rename("trend_f")


def trend_detail(ndx_close):
    c = ndx_close.dropna()
    rv = c.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    smas = {n: c.rolling(n).mean().iloc[-1] for n in TREND_SMAS}
    return {"close": c.iloc[-1], "rv20": rv, "smas": smas,
            "above": sum(c.iloc[-1] > v for v in smas.values()), "f": trend_fraction(c).iloc[-1]}


# ---------------- sleeve 2 ----------------
def ibs(o):
    rng = (o["High"] - o["Low"]).replace(0, np.nan)
    return ((o["Close"] - o["Low"]) / rng).rename("ibs")


def ibs_positions(ohlc_by_etf, spy_close):
    """Daily state machine. Returns (weights DataFrame [x equity notional, decided at each close], trade log).
    Order of operations at each close t: exits first (C[t] > H[t-1] or held >= MAXHOLD days), then entries
    (IBS[t] < ENTRY, lowest IBS first, into free slots). Open positions are re-sized daily to the regime size."""
    idx = spy_close.index
    C = pd.DataFrame({k: v["Close"] for k, v in ohlc_by_etf.items()}).reindex(idx)
    H = pd.DataFrame({k: v["High"] for k, v in ohlc_by_etf.items()}).reindex(idx)
    I = pd.DataFrame({k: ibs(v) for k, v in ohlc_by_etf.items()}).reindex(idx)
    regime_on = (spy_close > spy_close.rolling(200).mean())
    ok = C.notna() & H.shift(1).notna()
    held = {}   # etf -> bars held
    W = np.zeros((len(idx), C.shape[1])); cols = list(C.columns); log = []
    Cv, Hv, Iv, okv = C.values, H.values, I.values, ok.values
    for t in range(1, len(idx)):
        for e in list(held):
            j = cols.index(e); held[e] += 1
            if (Cv[t, j] > Hv[t - 1, j]) or held[e] >= IBS_MAXHOLD:
                log.append((idx[t], e, "SELL", "close>prior high" if Cv[t, j] > Hv[t - 1, j] else "time stop"))
                del held[e]
        free = IBS_SLOTS - len(held)
        if free > 0:
            cands = [(Iv[t, j], cols[j]) for j in range(len(cols))
                     if cols[j] not in held and okv[t, j] and not np.isnan(Iv[t, j]) and Iv[t, j] < IBS_ENTRY]
            for v, e in sorted(cands)[:free]:
                held[e] = 0; log.append((idx[t], e, "BUY", f"IBS {v:.3f}"))
        size = SLOT_ON if regime_on.iloc[t] else SLOT_OFF
        for e in held:
            W[t, cols.index(e)] = size
    return pd.DataFrame(W, index=idx, columns=cols), pd.DataFrame(log, columns=["date", "etf", "action", "why"]), held


# ---------------- futures account: stack sleeves 1+2 under a gross cap ----------------
def stack(ibs_w, trend_f, L_trend, max_gross):
    """Combine IBS slot weights (DataFrame, x equity per ETF) with trend exposure L_trend*f on QQQ.
    If total notional exceeds max_gross, IBS slots are shrunk first (trend keeps priority)."""
    W = ibs_w.copy()
    t = (L_trend * trend_f).reindex(W.index).fillna(0.0)
    if max_gross is not None:
        room = (max_gross - t).clip(lower=0)
        ib = W.sum(axis=1)
        W = W.mul(np.minimum(1.0, room / ib.where(ib > 0, 1.0)), axis=0)
        t = t.clip(upper=max_gross)
    W["QQQ"] = W["QQQ"] + t
    return W


def round_contracts(target_x, equity, contract_usd, max_gross):
    """Nearest whole contracts, then trim the most over-sized position until gross <= max_gross."""
    exact = {k: target_x[k] * equity / contract_usd[k] for k in target_x}
    n = {k: int(round(v)) for k, v in exact.items()}
    gross = lambda: sum(abs(n[k]) * contract_usd[k] for k in n) / equity
    while max_gross is not None and gross() > max_gross + 1e-9:
        k = max((k for k in n if n[k] != 0), key=lambda k: abs(n[k]) - abs(exact[k]))
        n[k] -= int(np.sign(n[k]))
    return n, exact, gross()


# ---------------- month-end overlay (sleeve 2b) ----------------
def _easter(y):
    a = y % 19; b, c = divmod(y, 100); d, e = divmod(b, 4); f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i, k = divmod(c, 4); l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451; mo = (h + l - 7 * m + 114) // 31; dd = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(y, mo, dd)


def nyse_holidays(y):
    """Regular NYSE full-day holidays (unscheduled closures are not predictable)."""
    def obs(d):
        return d - pd.Timedelta(days=1) if d.weekday() == 5 else d + pd.Timedelta(days=1) if d.weekday() == 6 else d
    def nth(month, weekday, n):
        d = pd.Timestamp(y, month, 1); d += pd.Timedelta(days=(weekday - d.weekday()) % 7)
        return d + pd.Timedelta(weeks=n - 1)
    def last(month, weekday):
        d = pd.Timestamp(y, month, 1) + pd.offsets.MonthEnd(0)
        return d - pd.Timedelta(days=(d.weekday() - weekday) % 7)
    hs = [obs(pd.Timestamp(y, 1, 1)), nth(1, 0, 3), nth(2, 0, 3), _easter(y) - pd.Timedelta(days=2), last(5, 0),
          obs(pd.Timestamp(y, 7, 4)), nth(9, 0, 1), nth(11, 3, 4), obs(pd.Timestamp(y, 12, 25))]
    if y >= 2022:
        hs.append(obs(pd.Timestamp(y, 6, 19)))
    return set(hs)


def trading_days_left_in_month(day):
    """NYSE trading days after `day` in the same month (0 = `day` is the month's last trading day)."""
    end = day + pd.offsets.MonthEnd(0)
    hol = nyse_holidays(day.year)
    return sum(1 for d in pd.bdate_range(day + pd.Timedelta(days=1), end) if d not in hol)


def monthend_weights(spy_close, bond_close, size):
    """Signed MES weight (x equity) decided at each close: -sign(s) x size on the second-to-last trading day.
    History uses the realised trading calendar; the current (unfinished) month uses the holiday calendar."""
    idx = spy_close.dropna().index
    px = pd.DataFrame({"S": spy_close, "B": bond_close}).reindex(idx).ffill()
    r = px.pct_change()
    mtd = (1 + r.fillna(0)).groupby([idx.year, idx.month]).cumprod() - 1
    s = mtd["S"] - mtd["B"]
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    dl = ym.groupby(ym).cumcount(ascending=False)
    cur = ym == ym.iloc[-1]
    dl[cur] = [trading_days_left_in_month(d) for d in idx[cur]]
    w = pd.Series(np.where(dl == 1, -np.sign(s) * size, 0.0), index=idx, name="SPY")
    w[s.isna() | r["B"].isna()] = 0.0
    return w, s, dl


def add_overlay(W, ov, max_gross):
    """Add a signed SPY/MES overlay, shrunk so that gross (on net per-instrument exposure) stays <= max_gross."""
    W = W.copy()
    ov = ov.reindex(W.index).fillna(0.0)
    if max_gross is None:
        W["SPY"] = W["SPY"] + ov
        return W
    a = W["SPY"]; rest = W.abs().sum(axis=1) - a.abs()
    R = (max_gross - rest).clip(lower=0)
    sg = np.sign(ov); m = ov.abs()
    room = (R - sg * a).clip(lower=0)
    W["SPY"] = a + sg * np.minimum(m, room)
    return W


# ---------------- sleeve 3 ----------------
def crypto_trend(c):
    c = c.dropna()
    votes = sum((c > c.rolling(n).mean()).astype(float) for n in CRYPTO_SMAS) / len(CRYPTO_SMAS)
    votes[c.rolling(max(CRYPTO_SMAS)).mean().isna()] = np.nan
    return votes


def crypto_weights(btc, eth):
    out = {}
    for name, c in (("BTC", btc), ("ETH", eth)):
        c = c.dropna()
        vol = c.pct_change().rolling(CRYPTO_VOLWIN).std() * np.sqrt(365)
        out[name] = 0.5 * crypto_trend(c) * np.minimum(1.0, CRYPTO_VT / vol)
    return pd.DataFrame(out)
