"""Vectorised daily backtester (v2, critic-audited 2026-10-11, see research/_critic/ENGINE_AUDIT.md).

Convention (no look-ahead):
  * `weights` row t = target weights decided with information available at the CLOSE of bar t.
  * lag=1  -> trade executed at the close of bar t (needs signal ~minutes before close / MOC order).
  * lag=2  -> trade executed at the close of bar t+1 (fully conservative, one full day delay).
  * Asset return on bar t = close[t]/close[t-1]-1.  Portfolio return on bar t = sum(w[t-lag] * r[t]).
  * Orders only fill on bars where the asset trades (`tradable`, default = return not NaN). A weight held on
    a NaN return raises (compute returns from forward-filled prices on mixed calendars).
  * Uninvested fraction earns the risk-free rate; leverage >1 pays rf + borrow spread (calendar-time accrual).
  * Costs: cost_bps per unit of turnover (one-way). Default: target weights are re-established every bar
    (free daily rebalancing). Pass drift=True to let weights drift between trades and only trade when
    |target - drifted| > band (costs charged on actual traded turnover).
  * Mixed calendars: crypto (Yahoo UTC close, 7 days) driving stock trades -> lag the crypto signal by 1 day.
"""
import warnings
import numpy as np, pandas as pd

TRADING_DAYS = 252


def _gap_years(idx):
    return pd.Series(idx, index=idx).diff().dt.days.fillna(1).values / 365.0


def _prep(weights, rets, rf, lag, tradable):
    if lag < 1:
        raise ValueError("lag must be >= 1 (lag=0 trades at the close it observes = look-ahead)")
    if isinstance(weights, pd.Series):
        weights = weights.to_frame(rets.name if isinstance(rets, pd.Series) else "asset")
    if isinstance(rets, pd.Series):
        rets = rets.to_frame(weights.columns[0])
    idx = rets.index
    unknown = weights.columns.difference(rets.columns)
    if len(unknown):
        raise KeyError(f"weights given for assets without returns: {list(unknown)}")
    on_cal = weights.index.isin(idx).mean()
    if on_cal < 0.5:
        warnings.warn(f"only {on_cal:.0%} of weight dates lie on the return calendar; carried to next bar")
    w = weights.astype(float).reindex(idx.union(weights.index)).ffill().reindex(idx)
    w = w.reindex(columns=rets.columns)
    r = rets.astype(float)
    if np.isinf(r.values).any():
        raise ValueError("non-finite returns (zero/negative prices?)")
    if tradable is None:
        tradable = r.notna()
    tradable = tradable.reindex(index=idx, columns=rets.columns).fillna(False).astype(bool)
    w = w.where(tradable).ffill().fillna(0.0)
    if rf is None:
        rf = pd.Series(0.0, index=idx)
    elif not rf.index.equals(idx):
        cum = (1 + rf.fillna(0.0)).cumprod()
        rf = cum.reindex(cum.index.union(idx)).ffill().reindex(idx).pct_change().fillna(0.0)
    rf = rf.reindex(idx).fillna(0.0)
    return w, r, rf, idx


def backtest(weights, rets, rf=None, cost_bps=5.0, lag=1, borrow_spread=0.01, tradable=None,
             drift=False, band=0.0):
    w, r, rf, idx = _prep(weights, rets, rf, lag, tradable)
    target = w.shift(lag).fillna(0.0)          # weights whose trade has been executed before bar t
    bad = (target != 0) & r.isna()
    if bad.any().any():
        first = {c: str(bad.index[bad[c]][0].date()) for c in bad.columns if bad[c].any()}
        raise ValueError(f"non-zero held weight on NaN returns (forgot to ffill prices?): {first}")
    r0 = r.fillna(0.0)
    spread = borrow_spread * _gap_years(idx)
    if not drift:
        held = target
        gross = (held * r0).sum(axis=1)
        cash = 1 - held.sum(axis=1)
        cash_ret = np.where(cash >= 0, cash * rf, cash * (rf + spread))
        turnover = held.diff().abs().sum(axis=1)
        turnover.iloc[0] = held.iloc[0].abs().sum()
        port = gross + cash_ret - turnover * cost_bps / 1e4
        return port.rename("ret"), held, turnover
    # drift-aware path loop
    T = target.values; R = r0.values; RF = rf.values; n, k = T.shape
    cur = np.zeros(k); out = np.zeros(n); to = np.zeros(n); H = np.zeros((n, k))
    prev_t = np.zeros(k)
    for i in range(n):
        tgt = T[i]
        changed_sign = np.any((np.abs(tgt) > 0) != (np.abs(prev_t) > 0))
        if changed_sign or np.abs(tgt - cur).sum() > band or (band == 0 and np.any(tgt != prev_t)):
            to[i] = np.abs(tgt - cur).sum(); cur = tgt.copy()
        prev_t = tgt
        H[i] = cur
        c = 1 - cur.sum()
        cr = c * RF[i] if c >= 0 else c * (RF[i] + spread[i])
        g = (cur * R[i]).sum()
        out[i] = g + cr - to[i] * cost_bps / 1e4
        # drift weights to end of bar
        val = 1 + g + cr
        if val > 0:
            cur = cur * (1 + R[i]) / val
    held = pd.DataFrame(H, index=idx, columns=target.columns)
    return pd.Series(out, index=idx, name="ret"), held, pd.Series(to, index=idx)


def stats(ret, rf=None, name=None):
    ret = ret.dropna()
    if len(ret) < 20:
        return {}
    n = len(ret)
    span = (ret.index[-1] - ret.index[0]).days / 365.25
    years = span * n / (n - 1)
    ppy = n / years
    eq = np.concatenate([[1.0], (1 + ret).cumprod().values])
    cagr = eq[-1] ** (1 / years) - 1 if eq[-1] > 0 else -1.0
    vol = ret.std() * np.sqrt(ppy)
    ex = ret - (rf.reindex(ret.index).fillna(0) if rf is not None else 0)
    sharpe = ex.mean() / ex.std() * np.sqrt(ppy) if ex.std() > 0 else np.nan
    dd_dev = np.sqrt((ex.clip(upper=0) ** 2).mean())
    sortino = ex.mean() / dd_dev * np.sqrt(ppy) if dd_dev > 0 else np.nan
    dd = eq / np.maximum.accumulate(eq) - 1
    mdd = dd.min()
    uw = pd.Series(dd[1:] < 0, index=ret.index).astype(int)
    grp = (uw.diff() != 0).cumsum()
    longest = int(uw.groupby(grp).sum().max()) if uw.any() else 0
    yearly = (1 + ret).groupby(ret.index.year).prod() - 1
    roll12 = (1 + ret).rolling(int(round(ppy))).apply(np.prod, raw=True) - 1 if n > ppy else pd.Series([np.nan])
    return {
        "name": name or getattr(ret, "name", ""),
        "start": ret.index[0].date().isoformat(), "end": ret.index[-1].date().isoformat(),
        "years": round(years, 1),
        "CAGR": round(cagr * 100, 2), "vol": round(vol * 100, 1),
        "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
        "maxDD": round(mdd * 100, 1), "calmar": round(cagr / -mdd, 2) if mdd < 0 else np.nan,
        "worst_year": round(yearly.min() * 100, 1), "best_year": round(yearly.max() * 100, 1),
        "worst_12m": round(roll12.min() * 100, 1) if roll12.notna().any() else np.nan,
        "pct_pos_years": round((yearly > 0).mean() * 100, 0),
        "longest_underwater_bars": longest,
        "longest_underwater_days": longest,  # kept for backward compatibility (counts BARS)
        "final_x": round(eq[-1], 2),
    }


def trades(held_series, asset_ret, cost_bps=0.0):
    """Round trips of a single-asset position (long or short). entry/exit = first/last bar held;
    execution happened at the close before `entry` and at the close of the last held bar."""
    h = held_series.fillna(0)
    sgn = np.sign(h)
    new = (sgn != 0) & (sgn != sgn.shift(1, fill_value=0))
    out = []
    starts = list(h.index[new])
    for s in starts:
        seg_idx = h.loc[s:].index
        s_sign = sgn.loc[s]
        stop = sgn.loc[s:] != s_sign
        end_pos = stop.values.argmax() if stop.any() else len(seg_idx)
        seg = seg_idx[:end_pos]
        pnl = (1 + h.loc[seg] * asset_ret.loc[seg].fillna(0)).prod() - 1 - 2 * cost_bps / 1e4 * abs(h.loc[s])
        out.append({"entry": seg[0], "exit": seg[-1], "bars": len(seg), "side": int(s_sign), "ret": pnl})
    return pd.DataFrame(out)


def trade_stats(tr):
    if len(tr) == 0:
        return {}
    wins = tr[tr.ret > 0].ret; losses = tr[tr.ret <= 0].ret
    pf = wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else np.inf
    return {"n_trades": len(tr), "win_rate": round(len(wins) / len(tr) * 100, 1),
            "avg_win": round(wins.mean() * 100, 2) if len(wins) else 0,
            "avg_loss": round(losses.mean() * 100, 2) if len(losses) else 0,
            "profit_factor": round(pf, 2), "avg_bars": round(tr.bars.mean(), 1)}


def yearly_table(ret_dict):
    df = pd.DataFrame({k: (1 + v.dropna()).groupby(v.dropna().index.year).prod() - 1 for k, v in ret_dict.items()})
    return (df * 100).round(1)


def table(rows):
    return pd.DataFrame(rows).set_index("name")
