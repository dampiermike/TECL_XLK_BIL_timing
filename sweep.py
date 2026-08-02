"""
Parallel parameter sweep harness.

Splits every candidate into in-sample (2009-2019) and out-of-sample (2020-2026)
and reports both, so a config that only works on the fitted half is visible as
such. Nothing here selects on the OOS half -- selection happens on IS, and OOS
is read afterwards as a check.
"""

import itertools
import multiprocessing as mp
import os

import pandas as pd

from data import load_all
from engine import run, run_weights, metrics
from strategy import build_features, make_target
import strategy2

IS_START, IS_END = "2009-01-02", "2019-12-31"
OOS_START, OOS_END = "2020-01-01", "2026-07-31"
FULL_START = "2009-01-02"

_G = {}


def _init(signal_ticker):
    prices, returns = load_all()
    _G["returns"] = returns
    _G["cash"] = returns["BIL"]
    _G["f"] = build_features(prices, signal_ticker)


def grid(spec):
    """spec: dict of name -> list of values. Yields dicts of every combination."""
    keys = list(spec)
    for combo in itertools.product(*(spec[k] for k in keys)):
        yield dict(zip(keys, combo))


def _eval(job):
    kind, p, cost_bps = job
    f, returns, cash = _G["f"], _G["returns"], _G["cash"]
    try:
        weights = None
        if kind == "v2":
            tgt = strategy2.make_target(f, p)
        elif kind == "vote":
            weights = strategy2.vote_weights(f, p, returns)
            tgt = weights.index.to_series()
        elif kind == "v2vol":
            weights = strategy2.vol_target_weights(f, p, returns)
            tgt = weights.index.to_series()  # length check only
        else:
            tgt = make_target(f, kind, p)
        if len(tgt) < 500:
            return None
        out = {"kind": kind, **{k: v for k, v in p.items()}}
        for tag, (a, b) in (("IS", (IS_START, IS_END)),
                            ("OOS", (OOS_START, OOS_END)),
                            ("ALL", (FULL_START, OOS_END))):
            bt = (run_weights(returns, weights, cost_bps=cost_bps, start=a, end=b)
                  if weights is not None
                  else run(returns, tgt, cost_bps=cost_bps, start=a, end=b))
            m = metrics(bt, cash, label="")
            if not m:
                return None
            out[f"{tag}_CAGR"] = m["CAGR"]
            out[f"{tag}_maxDD"] = m["maxDD"]
            out[f"{tag}_Sharpe"] = m["Sharpe"]
            out[f"{tag}_Calmar"] = m["Calmar"]
            if tag == "ALL":
                out["trades_per_yr"] = m["trades_per_yr"]
                out["vol"] = m["vol"]
                for k, v in m.items():
                    if k.startswith("pct_"):
                        out[k] = v
        return out
    except Exception:
        return None


def sweep(kind, spec, cost_bps=10.0, signal_ticker="XLK", workers=None, chunk=200):
    jobs = [(kind, p, cost_bps) for p in grid(spec)]
    workers = workers or max(1, mp.cpu_count() - 2)
    with mp.Pool(workers, initializer=_init, initargs=(signal_ticker,)) as pool:
        rows = pool.map(_eval, jobs, chunksize=chunk)
    rows = [r for r in rows if r]
    return pd.DataFrame(rows), len(jobs)


def report(df, n_jobs, title, sort_by="IS_Calmar", top=15, extra_cols=()):
    print(f"\n{'='*100}\n{title}   ({len(df)} valid of {n_jobs} combos)\n{'='*100}")
    if df.empty:
        print("no valid results")
        return df
    d = df.sort_values(sort_by, ascending=False)
    cols = ["IS_CAGR", "IS_maxDD", "IS_Sharpe", "OOS_CAGR", "OOS_maxDD", "OOS_Sharpe",
            "ALL_CAGR", "ALL_maxDD", "ALL_Sharpe", "trades_per_yr"]
    pcols = [c for c in d.columns if c.startswith("pct_")]
    show = list(extra_cols) + cols + pcols
    with pd.option_context("display.width", 250, "display.max_columns", 60,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print(d[show].head(top).to_string(index=False))
    return d


def save(df, name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nsaved -> results/{name}  ({len(df)} rows)")
