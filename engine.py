"""
Backtest core: turns a daily target-asset series into an equity curve + metrics.

Timing convention (no lookahead anywhere):
  signals are built from data up to and including day t's close;
  the resulting target is held over day t+1, so it earns return_{t+1}.
  Implemented as `weights.shift(1) * returns`.

Cost model: COST_BPS charged on the traded notional each time the target changes.
Cash is never assumed to be free -- idle capital earns BIL's actual return, and
Sharpe is computed on return in EXCESS of that same BIL series.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_COST_BPS = 10.0  # per side, on the notional switched


def run(returns, target, cost_bps=DEFAULT_COST_BPS, start=None, end=None):
    """Backtest a single-asset-at-a-time strategy.

    returns : DataFrame of daily simple returns, one column per tradable
    target  : Series of column names (the asset to hold over the NEXT day)
    """
    idx = returns.index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    target = target.reindex(idx).ffill().dropna()
    idx = target.index

    assets = sorted(set(target.unique()))
    w = pd.DataFrame(0.0, index=idx, columns=assets)
    for a in assets:
        w.loc[target == a, a] = 1.0

    held = w.shift(1).fillna(0.0)
    r = returns.reindex(idx)[assets].fillna(0.0)
    gross = (held * r).sum(axis=1)

    # notional switched on the day the target changes (charged when it takes effect)
    turn = (w - w.shift(1)).abs().sum(axis=1).shift(1).fillna(0.0) / 2.0
    cost = turn * (cost_bps / 1e4)
    net = gross - cost

    equity = (1 + net).cumprod()
    return pd.DataFrame({"ret": net, "gross": gross, "cost": cost,
                         "equity": equity, "target": target, "turn": turn})


def metrics(bt, cash_ret, label=""):
    r = bt["ret"]
    eq = bt["equity"]
    n = len(r)
    if n < 30:
        return {}
    years = n / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()

    rf = cash_ret.reindex(r.index).fillna(0.0)
    ex = r - rf
    sharpe = ex.mean() / ex.std() * np.sqrt(TRADING_DAYS) if ex.std() > 0 else np.nan
    downside = ex[ex < 0].std()
    sortino = ex.mean() / downside * np.sqrt(TRADING_DAYS) if downside and downside > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)

    tgt = bt["target"]
    expo = {f"pct_{a}": float((tgt == a).mean()) for a in sorted(tgt.unique())}
    switches = int((tgt != tgt.shift(1)).sum())

    m = {"label": label, "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
         "years": round(years, 2), "CAGR": cagr, "maxDD": maxdd, "Sharpe": sharpe,
         "Sortino": sortino, "vol": vol, "Calmar": cagr / abs(maxdd) if maxdd else np.nan,
         "final_x": eq.iloc[-1], "switches": switches,
         "trades_per_yr": switches / years,
         "cost_drag": bt["cost"].sum() / years}
    m.update(expo)
    return m


def buy_hold(returns, ticker, cash_ret, start=None, end=None):
    t = pd.Series(ticker, index=returns.index)
    bt = run(returns, t, cost_bps=0.0, start=start, end=end)
    return metrics(bt, cash_ret, label=f"B&H {ticker}")


def fmt(rows):
    """Compact table for a list of metric dicts."""
    if not rows:
        return "(no rows)"
    cols = ["label", "CAGR", "maxDD", "Sharpe", "Calmar", "vol", "final_x", "trades_per_yr"]
    pcts = {"CAGR", "maxDD", "vol"}
    out = [f"{'strategy':<34}{'CAGR':>8}{'maxDD':>8}{'Shrp':>7}{'Clmr':>7}{'vol':>8}{'mult':>10}{'tr/yr':>7}"]
    out.append("-" * 89)
    for m in rows:
        line = f"{str(m.get('label',''))[:33]:<34}"
        for c in cols[1:]:
            v = m.get(c, np.nan)
            if c in pcts:
                line += f"{v:>7.1%} " if pd.notna(v) else f"{'-':>8}"
            elif c == "final_x":
                line += f"{v:>9.1f}x" if pd.notna(v) else f"{'-':>10}"
            else:
                line += f"{v:>7.2f}" if pd.notna(v) else f"{'-':>7}"
        out.append(line)
    return "\n".join(out)


def run_weights(returns, weights, cost_bps=DEFAULT_COST_BPS, start=None, end=None):
    """Backtest arbitrary daily weights (columns = tickers, rows sum to <= 1).

    Same timing rule as run(): weights dated t are held over t+1. Any unallocated
    fraction sits in BIL, so cash always earns the T-bill return.
    """
    idx = weights.index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    w = weights.reindex(idx).fillna(0.0)

    resid = 1.0 - w.sum(axis=1)
    if "BIL" not in w.columns:
        w["BIL"] = 0.0
    w["BIL"] = w["BIL"] + resid

    held = w.shift(1).fillna(0.0)
    r = returns.reindex(idx)[w.columns].fillna(0.0)
    gross = (held * r).sum(axis=1)
    turn = (w - w.shift(1)).abs().sum(axis=1).shift(1).fillna(0.0) / 2.0
    cost = turn * (cost_bps / 1e4)
    net = gross - cost

    dominant = w.drop(columns=[]).idxmax(axis=1)
    return pd.DataFrame({"ret": net, "gross": gross, "cost": cost,
                         "equity": (1 + net).cumprod(), "target": dominant,
                         "turn": turn})
