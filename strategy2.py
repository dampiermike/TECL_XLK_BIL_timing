"""
Classifier v2. Three changes that target what the diagnosis actually showed.

The stage-1 drawdowns were NOT crash losses -- they were chop losses. 2022 lost
44% while sitting in BIL 75% of the time, and every one of the ten worst days
was TECL held into a -4% XLK day. So v2 attacks whipsaw and volatility decay:

1. ASYMMETRIC CONFIRMATION. De-risking is immediate; re-risking needs `confirm_up`
   consecutive days. On a risk ladder (BIL=1, XLK=2, TECL=3) that is exactly a
   trailing rolling minimum of the raw level, which stays vectorised.

2. ABSOLUTE VOLATILITY GATE. A percentile gate lets TECL be held in a calm-for-2022
   tape that is still 35% vol. `abs_vol_max` caps realised vol in real units, and
   `vix_max` does the same on implied.

3. CIRCUIT BREAKER. A one-day escape to BIL when the tape dislocates (vol spike or
   a sharp drop below the fast MA), independent of the ladder's own confirmation.

TECS is layered on top with its own gate and confirmation so its contribution can
be measured in isolation rather than tangled with the long side.
"""

import numpy as np
import pandas as pd

LEVELS = {"BIL": 1, "XLK": 2, "TECL": 3}
INV = {v: k for k, v in LEVELS.items()}


def long_ladder(f, p):
    """Raw (unconfirmed) long-side level: 1=BIL, 2=XLK, 3=TECL."""
    trend = f[f"sma{p['trend_n']}"]
    risk_on = f["px"] > trend * (1 + p.get("trend_buffer", 0.0))

    breakout = (f[f"prox{p['prox_n']}"] >= p["prox_thr"]) & (f[f"roc{p['mom_n']}"] > p["mom_thr"])
    if p.get("vol_max") is not None:
        breakout &= f[f"volpct{p['vol_n']}"] <= p["vol_max"]
    if p.get("abs_vol_max") is not None:
        breakout &= f[f"vol{p['vol_n']}"] <= p["abs_vol_max"]
    if p.get("vix_max") is not None and "vix" in f:
        breakout &= f["vix"] <= p["vix_max"]
    if p.get("require_slope") and f"slope{p['trend_n']}" in f:
        breakout &= f[f"slope{p['trend_n']}"] > 0
    if p.get("er_min") is not None:
        breakout &= f[f"er{p.get('er_n', 20)}"] >= p["er_min"]
    if p.get("rsi_max") is not None:
        breakout &= f[f"rsi{p.get('rsi_n', 14)}"] <= p["rsi_max"]
    if p.get("macd_pos"):
        breakout &= f["macd"] > 0

    # a chop filter can also demote the XLK state to cash, not just block TECL
    if p.get("er_cash") is not None:
        risk_on &= f[f"er{p.get('er_n', 20)}"] >= p["er_cash"]

    lvl = pd.Series(1.0, index=f.index)
    lvl[risk_on] = 2.0
    lvl[risk_on & breakout] = 3.0
    lvl[trend.isna()] = np.nan
    return lvl


def circuit_breaker(f, p):
    """True on days the tape dislocates -- force BIL regardless of the ladder."""
    cb = pd.Series(False, index=f.index)
    if p.get("cb_vol_pct") is not None:
        cb |= f[f"volpct{p['vol_n']}"] >= p["cb_vol_pct"]
    if p.get("cb_drop") is not None:
        cb |= f[f"roc{p['cb_drop_n']}"] <= p["cb_drop"]
    if p.get("cb_vix") is not None and "vix" in f:
        cb |= f["vix"] >= p["cb_vix"]
    return cb


def crash_gate(f, p):
    """True when the tape is in a confirmed decline worth being short."""
    if p.get("crash_mom_thr") is None:
        return pd.Series(False, index=f.index)
    g = f[f"roc{p['crash_mom_n']}"] < p["crash_mom_thr"]
    if p.get("crash_below_n") is not None:
        g &= f["px"] < f[f"sma{p['crash_below_n']}"]
    if p.get("crash_vol_min") is not None:
        g &= f[f"volpct{p['vol_n']}"] >= p["crash_vol_min"]
    if p.get("crash_dd") is not None:
        g &= f[f"dd{p.get('crash_dd_n', 120)}"] <= p["crash_dd"]
    if p.get("crash_vix_min") is not None and "vix" in f:
        g &= f["vix"] >= p["crash_vix_min"]
    if p.get("crash_slope_n") is not None:
        g &= f[f"slope{p['crash_slope_n']}"] < 0
    if p.get("crash_confirm", 0) > 1:
        g = g.rolling(int(p["crash_confirm"]), min_periods=int(p["crash_confirm"])).min().astype(bool)
    return g.fillna(False)


def _sym_confirm(lvl, k):
    """Symmetric run-length confirmation: hold the last level until a new one
    has persisted k days in a row. Slower to de-risk than the rolling-min rule."""
    s = lvl.dropna()
    run = s.groupby((s != s.shift(1)).cumsum()).cumcount() + 1
    return s.where(run >= k).ffill().reindex(lvl.index)


def make_target(f, p):
    lvl = long_ladder(f, p)

    k = int(p.get("confirm_up", 1))
    if p.get("confirm_style", "asym") == "sym":
        held = _sym_confirm(lvl, k) if k > 1 else lvl
    else:
        held = lvl.rolling(k, min_periods=k).min() if k > 1 else lvl

    if p.get("cb_vol_pct") or p.get("cb_drop") or p.get("cb_vix"):
        held = held.where(~circuit_breaker(f, p), 1.0)

    tgt = held.map(INV)
    tgt[crash_gate(f, p)] = "TECS"
    return tgt.dropna()


def vol_target_weights(f, p, returns):
    """Alternative sizing: hold the ladder's asset but scale exposure to a vol budget.

    Instead of 100% TECL, hold min(1, target_vol / recent_vol_of_TECL) and leave the
    rest in BIL. Directly attacks the decay that chop inflicts on a 3x fund.
    """
    tgt = make_target(f, p)
    real = {"TECL": f[f"vol{p['vol_n']}"] * 3.0, "TECS": f[f"vol{p['vol_n']}"] * 3.0,
            "XLK": f[f"vol{p['vol_n']}"], "BIL": None}
    w = pd.DataFrame(0.0, index=tgt.index, columns=["TECL", "TECS", "XLK", "BIL"])
    fixed = {"TECL": p.get("tecl_w", 1.0), "TECS": p.get("tecs_w", 1.0),
             "XLK": p.get("xlk_w", 1.0)}
    for a in ["TECL", "TECS", "XLK"]:
        m = tgt == a
        if not m.any():
            continue
        if p.get("sizing", "voltarget") == "fixed":
            scale = pd.Series(fixed[a], index=tgt.index)
        else:
            scale = (p["target_vol"] / real[a]).clip(upper=p.get("max_leverage", 1.0))
            scale = scale.clip(lower=p.get("min_weight", 0.0)) * fixed[a]
            scale = scale.reindex(tgt.index)  # features span more dates than the target
        w.loc[m, a] = scale.reindex(tgt.index).fillna(0.0)[m]
    w["BIL"] = 1.0 - w[["TECL", "TECS", "XLK"]].sum(axis=1)
    return w


# --------------------------------------------------------------- voting rule
def vote_ladder(f, p):
    """Score-based alternative to the AND-chain gate.

    An AND-chain fails the whole breakout if any single condition misses, which
    makes it brittle to one noisy indicator. Here each bullish condition casts a
    vote and the level is set by the tally, so no single indicator can veto.
    """
    votes = pd.Series(0.0, index=f.index)
    votes += (f["px"] > f[f"sma{p['trend_n']}"]).astype(float)
    votes += (f[f"prox{p['prox_n']}"] >= p["prox_thr"]).astype(float)
    votes += (f[f"roc{p['mom_n']}"] > p["mom_thr"]).astype(float)
    votes += (f[f"volpct{p['vol_n']}"] <= p.get("vol_max_v", 0.7)).astype(float)
    votes += (f[f"slope{p['trend_n']}"] > 0).astype(float)
    votes += (f["macd"] > 0).astype(float)
    if p.get("use_credit") and "credit_roc20" in f:
        votes += (f["credit_roc20"] > 0).astype(float)

    n_cond = 6 + (1 if (p.get("use_credit") and "credit_roc20" in f) else 0)
    frac = votes / n_cond

    lvl = pd.Series(1.0, index=f.index)
    lvl[frac >= p["v_xlk"]] = 2.0
    lvl[frac >= p["v_tecl"]] = 3.0
    if p.get("rsi_max") is not None:
        lvl[(lvl == 3.0) & (f[f"rsi{p.get('rsi_n', 14)}"] > p["rsi_max"])] = 2.0
    lvl[f[f"sma{p['trend_n']}"].isna()] = np.nan
    return lvl


def make_target_vote(f, p):
    lvl = vote_ladder(f, p)
    k = int(p.get("confirm_up", 1))
    if p.get("confirm_style", "sym") == "sym":
        held = _sym_confirm(lvl, k) if k > 1 else lvl
    else:
        held = lvl.rolling(k, min_periods=k).min() if k > 1 else lvl
    if p.get("cb_vol_pct") or p.get("cb_drop") or p.get("cb_vix"):
        held = held.where(~circuit_breaker(f, p), 1.0)
    tgt = held.map(INV)
    tgt[crash_gate(f, p)] = "TECS"
    return tgt.dropna()


def vote_weights(f, p, returns):
    tgt = make_target_vote(f, p)
    w = pd.DataFrame(0.0, index=tgt.index, columns=["TECL", "TECS", "XLK", "BIL"])
    fixed = {"TECL": p.get("tecl_w", 1.0), "TECS": p.get("tecs_w", 1.0), "XLK": p.get("xlk_w", 1.0)}
    for a in ["TECL", "TECS", "XLK"]:
        m = tgt == a
        if m.any():
            w.loc[m, a] = fixed[a]
    w["BIL"] = 1.0 - w[["TECL", "TECS", "XLK"]].sum(axis=1)
    return w
