"""
Shared data loading + repair for the TECL / XLK / TECS timing project.

Two EODHD defects matter here and both are repaired at load time:

1. TECS `adjusted_close` is unusable. Its ratio to `close` moves on ~900 days,
   so its daily returns are not the fund's returns (regressed vs XLK it gives
   beta -2.40 / corr -0.75 instead of the -3.0 / -0.99 a 3x inverse must show).
   `close` carries the correct daily returns but is NOT back-adjusted for the
   fund's eight reverse splits, so the price jumps 4-10x on those ex-dates.
   Repair: take returns from `close` and divide out the split factor on the
   eight detected split days. Verified below to restore beta -3.00 / corr -0.99.

2. TECS reverse splits are detected, not hardcoded blindly: a split day is one
   where the realised gross return divided by the -3x XLK expectation lands
   within 2% of a clean integer ratio (4, 5 or 10). The detected set is asserted
   against the known Direxion schedule so a data change trips the check.

TECL and every other ticker use `adjusted_close`, which validates clean.

Because `close` excludes distributions, the repaired TECS series is a price
series: it slightly understates TECS's true total return. TECS is a short-side
hedge held a minority of the time, and understating its return is conservative.
"""

import json
import os

import numpy as np
import pandas as pd

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")

# Direxion TECS reverse splits (ex-date -> shares-in ratio). Detected from the
# data and cross-checked against this schedule.
TECS_SPLITS = {
    "2010-07-08": 5,
    "2013-04-02": 5,
    "2015-05-20": 4,
    "2018-03-29": 5,
    "2020-04-23": 10,
    "2021-10-25": 10,
    "2024-11-04": 10,
    "2026-07-15": 10,
}


def _raw(ticker):
    with open(os.path.join(DIR, f"{ticker}.json")) as f:
        df = pd.DataFrame(json.load(f))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if df.index.duplicated().any():
        raise ValueError(f"{ticker}: duplicate dates")
    return df


def tecs_returns(xlk_ret):
    """Daily total-return-proxy series for TECS, with reverse splits divided out."""
    close = _raw("TECS")["close"]
    r = close.pct_change()

    j = pd.concat([xlk_ret, r], axis=1, join="inner").dropna()
    j.columns = ["x", "y"]
    factor = (1 + j["y"]) / (1 - 3 * j["x"])
    cand = factor[(factor > 1.5) | (factor < 1 / 1.5)]

    detected, bad = {}, []
    for d, v in cand.items():
        nearest = min((4, 5, 10), key=lambda k: abs(v - k))
        if abs(v / nearest - 1) > 0.02:
            bad.append((d.date(), round(float(v), 3)))
        else:
            detected[str(d.date())] = nearest
    if bad:
        raise ValueError(f"TECS: discontinuities that are not clean splits: {bad}")
    if detected != TECS_SPLITS:
        raise ValueError(
            f"TECS split schedule changed.\n detected={detected}\n expected={TECS_SPLITS}"
        )

    repaired = r.copy()
    for d, k in detected.items():
        ts = pd.Timestamp(d)
        repaired.loc[ts] = (1 + r.loc[ts]) / k - 1
    return repaired


def load_all():
    """Return (prices, returns) frames indexed on the common trading calendar.

    `prices` holds a synthetic total-return index for every traded ticker
    (base 100 at each ticker's first observation) so backtests only ever
    compound returns, never a raw price that a split could dislocate.
    """
    tickers = ["XLK", "TECL", "TECS", "BIL", "QQQ", "SPY", "TLT", "HYG", "SHY"]
    idx_tickers = ["VIX", "VXN", "IRX"]

    raw = {t: _raw(t) for t in tickers + idx_tickers}
    rets = {}
    for t in tickers:
        if t == "TECS":
            continue
        rets[t] = raw[t]["adjusted_close"].pct_change()
    rets["TECS"] = tecs_returns(rets["XLK"])

    returns = pd.DataFrame(rets).sort_index()
    prices = (1 + returns.fillna(0)).cumprod() * 100
    prices[returns.isna().all(axis=1)] = np.nan

    # keep each ticker NaN before its own inception rather than flat at 100
    for t in returns.columns:
        first = returns[t].first_valid_index()
        prices.loc[prices.index < first, t] = np.nan

    for t in idx_tickers:
        returns[t + "_LVL"] = raw[t]["close"]
        prices[t + "_LVL"] = raw[t]["close"]

    return prices, returns


def sanity_report():
    prices, returns = load_all()
    x = returns["XLK"]
    print("leverage check on repaired data (2009+):")
    for t, want in [("TECL", 3.0), ("TECS", -3.0)]:
        j = pd.concat([x, returns[t]], axis=1, join="inner").dropna()
        j = j[j.index >= "2009-01-01"]
        j.columns = ["x", "y"]
        beta = np.polyfit(j["x"], j["y"], 1)[0]
        print(f"  {t:<5} beta={beta:+.3f} (want {want:+.1f})  corr={j['x'].corr(j['y']):+.4f}  n={len(j)}")
    print("\nspan / worst day per ticker:")
    for t in ["XLK", "TECL", "TECS", "BIL"]:
        s = returns[t].dropna()
        print(f"  {t:<5} {s.index[0].date()} -> {s.index[-1].date()}  "
              f"n={len(s):>5}  min={s.min():+.2%} max={s.max():+.2%}")
    return prices, returns


if __name__ == "__main__":
    sanity_report()
