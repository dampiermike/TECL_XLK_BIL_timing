"""
Regime classification: XLK (up) / TECL (breakout) / TECS (crash) / BIL (sideways).

Every indicator is trailing-only -- a value dated t uses closes up to and
including t, and engine.run() then holds the resulting target over t+1.

The classifier is a hierarchical gate, which is the literal reading of the
strategy brief:

    trend gate      is the market above its long trend?
      risk-on   ->  is it ALSO breaking out (near highs, calm, strong)?  TECL : XLK
      risk-off  ->  is it ALSO crashing (falling fast, vol spiking)?     TECS : BIL

Alternative classifiers (score-bucket, dual-MA, vol-target) live alongside it so
the sweep can compare rule shapes, not just parameter values.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- indicators
def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def roc(s, n):
    return s.pct_change(n)


def realized_vol(s, n):
    return s.pct_change().rolling(n, min_periods=n).std() * np.sqrt(252)


def pct_rank(s, n):
    """Trailing percentile rank of the current value within the last n obs."""
    return s.rolling(n, min_periods=max(20, n // 4)).rank(pct=True)


def proximity_to_high(s, n):
    """1.0 = at the n-day high; 0.9 = 10% below it."""
    return s / s.rolling(n, min_periods=n).max()


def drawdown_from_high(s, n):
    return s / s.rolling(n, min_periods=n).max() - 1


def zscore(s, n):
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std()
    return (s - m) / sd


def efficiency_ratio(s, n):
    """Kaufman efficiency ratio: |net move| / sum of |daily moves| over n days.

    1.0 = a perfectly straight line, 0.0 = pure chop that ends where it began.
    This is the direct measure of the trend quality that decides whether a 3x
    fund compounds or bleeds, which is what the drawdown diagnosis pointed at.
    """
    net = s.diff(n).abs()
    path = s.diff().abs().rolling(n, min_periods=n).sum()
    return net / path


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def macd_hist(s, fast=12, slow=26, sig=9):
    line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return (line - line.ewm(span=sig, adjust=False).mean()) / s


def build_features(prices, signal_ticker="XLK"):
    """Precompute every indicator the classifiers can reference."""
    px = prices[signal_ticker].dropna()
    f = pd.DataFrame(index=px.index)
    f["px"] = px
    for n in (10, 20, 30, 50, 80, 90, 100, 110, 125, 150, 200, 250):
        f[f"sma{n}"] = sma(px, n)
        f[f"above{n}"] = (px > f[f"sma{n}"]).astype(float)
    for n in (5, 10, 20, 30, 45, 60, 90, 120, 200):
        f[f"roc{n}"] = roc(px, n)
    for n in (10, 20, 30, 60):
        f[f"vol{n}"] = realized_vol(px, n)
        f[f"volpct{n}"] = pct_rank(f[f"vol{n}"], 252)
    for n in (20, 40, 50, 60, 75, 90, 120, 250):
        f[f"prox{n}"] = proximity_to_high(px, n)
        f[f"dd{n}"] = drawdown_from_high(px, n)
    for n in (20, 50, 100):
        f[f"z{n}"] = zscore(px, n)
    for n in (10, 20, 30, 60, 90):
        f[f"er{n}"] = efficiency_ratio(px, n)
    for n in (5, 14, 30):
        f[f"rsi{n}"] = rsi(px, n)
    f["macd"] = macd_hist(px)
    # slope of the long trend, in % per day, from an OLS fit over the window
    for n in (50, 80, 90, 100, 110, 125, 150, 200):
        f[f"slope{n}"] = px.rolling(n, min_periods=n).apply(
            lambda w: np.polyfit(np.arange(len(w)), np.log(w), 1)[0], raw=True)

    for col, src in (("vix", "VIX_LVL"), ("vxn", "VXN_LVL")):
        if src in prices:
            v = prices[src].reindex(px.index).ffill()
            f[col] = v
            f[col + "_sma20"] = sma(v, 20)
            f[col + "_pct"] = pct_rank(v, 252)
            f[col + "_chg20"] = v.pct_change(20)
    if "HYG" in prices and "TLT" in prices:
        credit = (prices["HYG"] / prices["TLT"]).reindex(px.index).ffill()
        f["credit_roc20"] = credit.pct_change(20)
        f["credit_above50"] = (credit > sma(credit, 50)).astype(float)
    return f


# ---------------------------------------------------------------- classifiers
def classify_hierarchical(f, p):
    """The brief's own logic: trend gate, then breakout / crash sub-gates."""
    trend = f[f"sma{p['trend_n']}"]
    risk_on = f["px"] > trend * (1 + p.get("trend_buffer", 0.0))

    breakout = (
        (f[f"prox{p['prox_n']}"] >= p["prox_thr"])
        & (f[f"roc{p['mom_n']}"] > p["mom_thr"])
    )
    if p.get("vol_max") is not None:
        breakout &= f[f"volpct{p['vol_n']}"] <= p["vol_max"]

    crash = f[f"roc{p['crash_mom_n']}"] < p["crash_mom_thr"]
    if p.get("crash_vol_min") is not None:
        crash &= f[f"volpct{p['vol_n']}"] >= p["crash_vol_min"]
    if p.get("crash_dd") is not None:
        crash &= f[f"dd{p['crash_dd_n']}"] <= p["crash_dd"]

    tgt = pd.Series("BIL", index=f.index, dtype=object)
    tgt[risk_on] = "XLK"
    tgt[risk_on & breakout] = "TECL"
    tgt[(~risk_on) & crash] = "TECS"
    tgt[f[f"sma{p['trend_n']}"].isna()] = np.nan
    return tgt


def classify_score(f, p):
    """Composite z-score of trend+momentum+vol, cut into four buckets."""
    trend_z = np.sign(f["px"] / f[f"sma{p['trend_n']}"] - 1) * \
        (f["px"] / f[f"sma{p['trend_n']}"] - 1).abs().pipe(lambda s: s / s.rolling(252, min_periods=60).std())
    mom_z = f[f"roc{p['mom_n']}"] / f[f"roc{p['mom_n']}"].rolling(252, min_periods=60).std()
    calm = 1 - 2 * f[f"volpct{p['vol_n']}"]
    score = p["w_trend"] * trend_z + p["w_mom"] * mom_z + p["w_calm"] * calm

    tgt = pd.Series(np.nan, index=f.index, dtype=object)
    tgt[score <= p["s_crash"]] = "TECS"
    tgt[(score > p["s_crash"]) & (score <= p["s_side"])] = "BIL"
    tgt[(score > p["s_side"]) & (score <= p["s_break"])] = "XLK"
    tgt[score > p["s_break"]] = "TECL"
    tgt[score.isna()] = np.nan
    return tgt


def classify_dualma(f, p):
    """Fast/slow MA cross sets the regime; vol percentile picks the gear."""
    fast, slow = f[f"sma{p['fast_n']}"], f[f"sma{p['slow_n']}"]
    up = fast > slow
    calm = f[f"volpct{p['vol_n']}"] <= p["vol_max"]
    deep = f[f"roc{p['crash_mom_n']}"] < p["crash_mom_thr"]

    tgt = pd.Series("BIL", index=f.index, dtype=object)
    tgt[up] = "XLK"
    tgt[up & calm & (f["px"] > fast)] = "TECL"
    tgt[(~up) & deep] = "TECS"
    tgt[slow.isna()] = np.nan
    return tgt


CLASSIFIERS = {
    "hier": classify_hierarchical,
    "score": classify_score,
    "dualma": classify_dualma,
}


def apply_confirm(tgt, days):
    """Require `days` consecutive identical signals before switching.

    Kills one-day whipsaws. Trailing-only: today's confirmed target depends on
    today's and earlier raw targets.
    """
    if not days or days <= 1:
        return tgt
    s = tgt.dropna()
    stable = (s != s.shift(1)).cumsum()
    run_len = s.groupby(stable).cumcount() + 1
    confirmed = s.where(run_len >= days)
    return confirmed.reindex(tgt.index).ffill()


def make_target(f, kind, p):
    tgt = CLASSIFIERS[kind](f, p)
    tgt = apply_confirm(tgt, p.get("confirm", 0))
    return tgt.dropna()
